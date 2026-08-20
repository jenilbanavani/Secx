"""
Persistence Stage for Decisio Pipeline.

Saves verified decisions, evidence, and governed files to the SQLite/PostgreSQL
database using the project's SQLAlchemy models.

Key design details:
- Creates a default system user if no users exist in the database, allowing the
  Playground Live Ingestion to run successfully without an OAuth login.
- Looks up or creates the Repository record.
- Persists DecisionCandidate, Evidence, and GovernedFile records.
- Returns list of persisted Decision Candidate UUIDs.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    DecisionCandidate,
    DecisionCategory,
    DecisionStatus,
    Evidence,
    GovernedFile,
    Repository,
    SourceType,
    User,
)
from app.services.pipeline.models import PipelineResult, VerifiedDecision

logger = logging.getLogger(__name__)


class PipelinePersistence:
    """Handles writing the PipelineResult to the database."""

    async def persist(self, db: AsyncSession, result: PipelineResult) -> list[str]:
        """Save decisions from PipelineResult to the database.

        Returns a list of persisted decision candidate UUIDs as strings.
        """
        if not result.has_significant_decision or not result.decisions:
            logger.info("No decisions to persist.")
            return []

        try:
            # 1. Ensure we have a User
            user_stmt = select(User).limit(1)
            user_res = await db.execute(user_stmt)
            user = user_res.scalar_one_or_none()

            if not user:
                # Create a default system user so the playground runs out-of-the-box
                user = User(
                    github_id=999999,
                    github_login="decisio-system-user",
                    github_name="Decisio System User",
                    github_token_encrypted="system-default-token-not-encrypted",
                )
                db.add(user)
                await db.flush()
                logger.info("Created default decisio-system-user for repository mapping.")

            # 2. Ensure we have the Repository
            owner, repo_name = result.repository.split("/", 1)
            repo_stmt = select(Repository).where(Repository.full_name == result.repository)
            repo_res = await db.execute(repo_stmt)
            repo = repo_res.scalar_one_or_none()

            if not repo:
                repo = Repository(
                    user_id=user.id,
                    github_id=int(uuid.uuid4().int >> 96),  # Generate pseudo GitHub ID
                    full_name=result.repository,
                    default_branch="main",
                    description="Auto-created repository for decision tracking",
                )
                db.add(repo)
                await db.flush()
                logger.info(f"Created Repository record for {result.repository}.")

            # 3. Save each VerifiedDecision
            persisted_ids: list[str] = []

            for d in result.decisions:
                # Map pipeline category to DB model category
                category = DecisionCategory.OTHER
                try:
                    category = DecisionCategory(d.category)
                except ValueError:
                    # Map tradeoff to other or appropriate mapping if enum mismatch
                    if d.category == "tradeoff":
                        category = DecisionCategory.TRADEOFF

                # Create the candidate
                candidate = DecisionCandidate(
                    repository_id=repo.id,
                    title=d.title,
                    description=d.summary,
                    rationale=d.rationale,
                    alternatives=d.alternatives,
                    chosen_approach=d.chosen_approach,
                    technologies=",".join(t.name for t in d.technologies),
                    has_significant_decision=True,
                    category=category,
                    status=DecisionStatus.PENDING,  # Awaiting human review
                    source_type=SourceType.PULL_REQUEST,
                    source_url=result.pr_url,
                    source_ref=str(result.pr_number),
                    confidence_score=d.confidence,
                    decided_at=datetime.now(timezone.utc),
                )
                db.add(candidate)
                await db.flush()
                persisted_ids.append(str(candidate.id))

                # Add evidence
                for e in d.evidence:
                    db_evidence = Evidence(
                        decision_id=candidate.id,
                        snippet=e.quote,
                        source_location=e.source_description or "PR Ingestion",
                        source_type=e.matched_source_key,
                        github_url=e.matched_source_url,
                        author=ctx_get_author_from_source(e.matched_source_key, result),
                        verified=e.verified,
                        match_score=e.match_score,
                    )
                    db.add(db_evidence)

                # Add governed files
                for f in d.affected_files:
                    db_file = GovernedFile(
                        decision_id=candidate.id,
                        file_path=f.file_path,
                    )
                    db.add(db_file)

            await db.commit()
            logger.info(f"Successfully persisted {len(persisted_ids)} decisions.")
            result.persisted_decision_ids = persisted_ids
            return persisted_ids

        except Exception as ex:
            await db.rollback()
            logger.error(f"Persistence error: {ex}")
            raise ex


def ctx_get_author_from_source(source_key: str | None, result: PipelineResult) -> str | None:
    """Helper to extract the author from the source context if possible."""
    if not source_key:
        return None
    if source_key == "pr_description":
        return result.pr_author
    # Fallback to the PR author or generic user
    return result.pr_author or "github_user"

from __future__ import annotations

"""
Decision models — the core domain of Decisio.

Three models here:
1. DecisionCandidate: An extracted (or manually created) architectural decision.
2. GovernedFile: A file path that a decision applies to (the VS Code link).
3. Evidence: An exact quote from a PR/commit that supports the decision.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


# ── Enums ────────────────────────────────────────────────


class DecisionStatus(str, enum.Enum):
    """Lifecycle of a decision candidate."""
    PENDING = "pending"      # Just extracted, awaiting human review
    APPROVED = "approved"    # Reviewed and confirmed — searchable
    REJECTED = "rejected"    # Reviewed and dismissed — hidden
    EDITED = "edited"        # Modified by user, then approved


class DecisionCategory(str, enum.Enum):
    """What kind of architectural decision is this?"""
    TECHNOLOGY_SELECTION = "technology_selection"  # "We chose Redis over Memcached"
    PATTERN_ADOPTION = "pattern_adoption"          # "We adopted CQRS"
    REFACTORING = "refactoring"                    # "We split the monolith"
    DEPRECATION = "deprecation"                    # "We deprecated the v1 API"
    TRADEOFF = "tradeoff"                          # "We accepted eventual consistency"
    ARCHITECTURE = "architecture"                  # "We moved to microservices"
    OTHER = "other"


class SourceType(str, enum.Enum):
    """Where the decision was extracted from."""
    PULL_REQUEST = "pull_request"
    COMMIT = "commit"
    MANUAL = "manual"  # User-created decision


# ── DecisionCandidate ────────────────────────────────────


class DecisionCandidate(Base, UUIDMixin, TimestampMixin):
    """A single architectural decision, extracted or manually created.

    Starts as 'pending', moves to 'approved' or 'rejected' after human review.
    Only 'approved' and 'edited' decisions are returned in search/query results.
    """

    __tablename__ = "decision_candidates"

    # ── Parent ───────────────────────────────────────────
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sync_job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sync_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Decision content ─────────────────────────────────
    title: Mapped[str] = mapped_column(
        String(500), nullable=False
    )  # e.g. "Migrate from REST to GraphQL for client flexibility"
    description: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Detailed explanation of the decision and rationale
    rationale: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Why the decision was made
    alternatives: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Alternatives considered/discussed
    chosen_approach: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Details on the chosen approach
    technologies: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Comma-separated list or JSON of technologies involved
    has_significant_decision: Mapped[bool] = mapped_column(
        default=True, nullable=False
    )  # False if we found no architectural decision
    category: Mapped[DecisionCategory] = mapped_column(
        SQLEnum(DecisionCategory, native_enum=False),
        default=DecisionCategory.OTHER,
        nullable=False,
    )

    # ── Status ───────────────────────────────────────────
    status: Mapped[DecisionStatus] = mapped_column(
        SQLEnum(DecisionStatus, native_enum=False),
        default=DecisionStatus.PENDING,
        nullable=False,
        index=True,
    )

    # ── Source traceability ──────────────────────────────
    source_type: Mapped[SourceType] = mapped_column(
        SQLEnum(SourceType, native_enum=False),
        nullable=False,
    )
    source_url: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )  # Link to the PR or commit on GitHub
    source_ref: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # PR number or commit SHA

    # ── LLM metadata ─────────────────────────────────────
    raw_llm_output: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Raw JSON from the LLM for debugging
    confidence_score: Mapped[float | None] = mapped_column(
        nullable=True
    )  # LLM's self-reported confidence (0.0 - 1.0)

    # ── Embedding ────────────────────────────────────────
    # Stored as a JSON-serialized list of floats for SQLite.
    # In Postgres, this would be a pgvector column.
    embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Timestamps ───────────────────────────────────────
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )  # When the original decision was made (PR merge date)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )  # When the user approved/rejected

    # ── Relationships ────────────────────────────────────
    repository: Mapped["Repository"] = relationship(
        "Repository", back_populates="decisions"
    )
    sync_job: Mapped["SyncJob | None"] = relationship(
        "SyncJob", back_populates="candidates"
    )
    governed_files: Mapped[list["GovernedFile"]] = relationship(
        "GovernedFile", back_populates="decision", cascade="all, delete-orphan"
    )
    evidence: Mapped[list["Evidence"]] = relationship(
        "Evidence", back_populates="decision", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Decision [{self.status.value}] {self.title[:50]}>"


# ── GovernedFile ─────────────────────────────────────────


class GovernedFile(Base, UUIDMixin):
    """A file path that a decision governs.

    This is the link that makes VS Code integration work.
    When a developer hovers over `src/cache/client.py`, we query this table
    to find all decisions that mention this file.

    `last_known_sha` enables staleness detection: if the file's current SHA
    differs from what we recorded, the decision may be outdated.
    """

    __tablename__ = "governed_files"

    decision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("decision_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(
        String(1000), nullable=False
    )  # e.g. "src/cache/redis_client.py"
    last_known_sha: Mapped[str | None] = mapped_column(
        String(40), nullable=True
    )  # Git blob SHA for staleness detection

    # ── Relationships ────────────────────────────────────
    decision: Mapped["DecisionCandidate"] = relationship(
        "DecisionCandidate", back_populates="governed_files"
    )

    # Index for fast lookups: "give me all decisions for this file"
    __table_args__ = (
        Index("ix_governed_files_file_path", "file_path"),
    )

    def __repr__(self) -> str:
        return f"<GovernedFile {self.file_path}>"


# ── Evidence ─────────────────────────────────────────────


class Evidence(Base, UUIDMixin):
    """An exact quote from a PR, commit, or comment that supports a decision.

    Evidence gives credibility. Instead of "we think this decision was made",
    we show "here's the exact text from PR #42 where the team discussed this."
    """

    __tablename__ = "evidence"

    decision_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("decision_candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    snippet: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # The exact quote
    source_location: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )  # e.g. "PR #42, review comment by @alice"
    source_type: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # e.g. "pr_description", "review_comment", "commit_message"
    github_url: Mapped[str | None] = mapped_column(
        String(1000), nullable=True
    )  # Direct link to this comment/commit
    author: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # GitHub login of who wrote it
    timestamp: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # ISO timestamp
    verified: Mapped[bool] = mapped_column(
        default=False, nullable=False
    )  # Whether matched against real PR text
    match_score: Mapped[float | None] = mapped_column(
        nullable=True
    )  # Score from SequenceMatcher

    # ── Relationships ────────────────────────────────────
    decision: Mapped["DecisionCandidate"] = relationship(
        "DecisionCandidate", back_populates="evidence"
    )

    def __repr__(self) -> str:
        return f"<Evidence {self.snippet[:40]}...>"

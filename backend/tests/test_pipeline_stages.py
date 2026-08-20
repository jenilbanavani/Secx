import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DecisionCandidate, Evidence, GovernedFile, Repository
from app.services.pipeline.models import (
    PRComment,
    PRCommit,
    PRContext,
    PRFile,
)
from app.services.pipeline.change_analysis import ChangeAnalyzer
from app.services.pipeline.detection import HeuristicDetector
from app.services.pipeline.verification import EvidenceVerifier, ExtractedEvidence
from app.services.pipeline.analysis import ImpactAnalyzer
from app.services.pipeline.persistence import PipelinePersistence
from app.services.pipeline.orchestrator import AnalysisPipeline


@pytest.fixture
def sample_pr_context() -> PRContext:
    """Fixture containing a mock PR Context for a real decision."""
    return PRContext(
        number=101,
        title="Adopt PostgreSQL for session caching instead of Redis",
        body=(
            "## Summary\n"
            "We decided to replace Redis with PostgreSQL for our session storage to get "
            "strict transactional consistency. We considered Redis cluster but transactional safety is paramount.\n"
            "## Impacted Files\n"
            "- backend/app/session.py\n"
            "- backend/app/config.py"
        ),
        author="lead-engineer",
        html_url="https://github.com/decisio/backend/pull/101",
        state="closed",
        merged=True,
        files=[
            PRFile(filename="backend/app/session.py", status="modified", additions=45, deletions=12),
            PRFile(filename="backend/app/config.py", status="modified", additions=5, deletions=1),
            PRFile(filename="README.md", status="modified", additions=10, deletions=0),
        ],
        review_comments=[
            PRComment(
                id=1001,
                author="reviewer-1",
                body="Using PostgreSQL solves the session synchronization consistency bug.",
                html_url="https://github.com/decisio/backend/pull/101#discussion_r1001",
                created_at="2026-08-20T10:00:00Z",
            )
        ],
        commits=[
            PRCommit(
                sha="abc12345",
                message="feat: migrate session storage backend to pg",
                author="lead-engineer",
                timestamp="2026-08-20T09:00:00Z",
            )
        ],
        total_additions=60,
        total_deletions=13,
        total_files_changed=3,
    )


@pytest.fixture
def routine_pr_context() -> PRContext:
    """Fixture containing a routine dependency update PR."""
    return PRContext(
        number=102,
        title="Bump js-yaml from 4.1.0 to 4.3.0",
        body="Bumps js-yaml from 4.1.0 to 4.3.0. No breaking changes.",
        author="dependabot",
        html_url="https://github.com/decisio/backend/pull/102",
        state="closed",
        merged=True,
        files=[
            PRFile(filename="package.json", status="modified", additions=1, deletions=1),
            PRFile(filename="package-lock.json", status="modified", additions=5, deletions=5),
        ],
        total_additions=6,
        total_deletions=6,
        total_files_changed=2,
    )


@pytest.mark.anyio
async def test_change_analyzer(sample_pr_context, routine_pr_context):
    analyzer = ChangeAnalyzer()

    # Test real decision PR
    res = analyzer.analyze(sample_pr_context)
    assert res.primary_change_type == "feature"
    assert not res.is_dependency_bump
    assert not res.is_docs_only

    # Test routine dependency PR
    res_routine = analyzer.analyze(routine_pr_context)
    assert res_routine.is_dependency_bump
    assert res_routine.primary_change_type == "dependency_bump"


@pytest.mark.anyio
async def test_heuristic_detector(sample_pr_context, routine_pr_context):
    change_analyzer = ChangeAnalyzer()
    detector = HeuristicDetector()

    # Real decision PR should be detected
    res_sample = change_analyzer.analyze(sample_pr_context)
    det_sample = detector.detect(sample_pr_context, res_sample)
    assert det_sample.decision_detected
    assert det_sample.confidence >= 50
    assert "signals" in det_sample.model_dump()

    # Routine update PR should NOT be detected
    res_routine = change_analyzer.analyze(routine_pr_context)
    det_routine = detector.detect(routine_pr_context, res_routine)
    assert not det_routine.decision_detected
    assert det_routine.confidence < 40


@pytest.mark.anyio
async def test_evidence_verification(sample_pr_context):
    verifier = EvidenceVerifier()

    # Test real evidence quote
    evidence = [
        ExtractedEvidence(
            quote="replace Redis with PostgreSQL for our session storage",
            source_description="PR description",
        ),
        # Test fabricated evidence quote (doesn't exist)
        ExtractedEvidence(
            quote="we did this because PostgreSQL is faster than Redis",
            source_description="Review comment",
        )
    ]

    from app.services.pipeline.models import ExtractedDecision
    decisions = [
        ExtractedDecision(
            title="Migrate to Postgres Cache",
            summary="Replacing Redis with Postgres",
            category="technology_selection",
            evidence=evidence,
        )
    ]

    verified = verifier.verify_decisions(decisions, sample_pr_context)
    
    # First evidence item must be verified
    assert verified[0].evidence[0].verified
    assert verified[0].evidence[0].matched_source_key == "pr_description"
    assert verified[0].evidence[0].matched_source_url == "https://github.com/decisio/backend/pull/101"
    assert verified[0].evidence[0].match_score > 0.8

    # Fabricated evidence item must NOT be verified
    assert not verified[0].evidence[1].verified
    assert verified[0].evidence[1].match_score < 0.6


@pytest.mark.anyio
async def test_impact_analyzer_filtering(sample_pr_context):
    analyzer = ImpactAnalyzer()

    # Filter to only relevant files specified by LLM
    relevant = ["backend/app/session.py"]
    affected = analyzer.analyze_affected_files(sample_pr_context, relevant_files=relevant)
    
    # Should only return backend/app/session.py, filtering out README.md and config.py
    assert len(affected) == 1
    assert affected[0].file_path == "backend/app/session.py"
    assert affected[0].relationship == "changed"

    # Verify fallback if no match or empty list
    affected_fallback = analyzer.analyze_affected_files(sample_pr_context, relevant_files=[])
    # Should fall back to sorting by additions+deletions
    assert len(affected_fallback) == 3
    assert affected_fallback[0].file_path == "backend/app/session.py"  # largest changes


@pytest.mark.anyio
async def test_pipeline_persistence(db_session: AsyncSession, sample_pr_context):
    from app.services.pipeline.models import PipelineResult, VerifiedDecision, AffectedFile, DetectedTechnology

    # Setup a PipelineResult
    dec = VerifiedDecision(
        title="Adopt PostgreSQL for caching",
        summary="Replaced Redis with Postgres for consistency",
        rationale="Need transactional consistency",
        alternatives="Redis cluster",
        chosen_approach="SQLAlchemy Session backend",
        category="technology_selection",
        evidence=[
            ExtractedEvidence(
                quote="replace Redis with PostgreSQL for our session storage",
                source_description="PR description",
                verified=True,
                matched_source_key="pr_description",
                matched_source_url="https://github.com/decisio/backend/pull/101",
                match_score=1.0,
            )
        ],
        affected_files=[
            AffectedFile(file_path="backend/app/session.py", status="modified", additions=45, deletions=12)
        ],
        technologies=[
            DetectedTechnology(name="PostgreSQL", detection_source="text_keyword", relationship="adopted")
        ],
        confidence=0.9,
    )

    result = PipelineResult(
        has_significant_decision=True,
        decisions=[dec],
        repository="decisio/backend",
        pr_number=101,
        pr_title="Adopt PostgreSQL for session caching instead of Redis",
        pr_url="https://github.com/decisio/backend/pull/101",
        pr_author="lead-engineer",
    )

    persistence = PipelinePersistence()
    persisted_ids = await persistence.persist(db_session, result)

    assert len(persisted_ids) == 1

    # Verify db records exist
    import uuid
    stmt = select(DecisionCandidate).where(DecisionCandidate.id == uuid.UUID(persisted_ids[0]))
    res = await db_session.execute(stmt)
    candidate = res.scalar_one_or_none()

    assert candidate is not None
    assert candidate.title == "Adopt PostgreSQL for caching"
    assert candidate.rationale == "Need transactional consistency"
    assert candidate.alternatives == "Redis cluster"
    assert candidate.chosen_approach == "SQLAlchemy Session backend"

    # Verify evidence was persisted
    stmt_ev = select(Evidence).where(Evidence.decision_id == candidate.id)
    res_ev = await db_session.execute(stmt_ev)
    ev_records = res_ev.scalars().all()
    assert len(ev_records) == 1
    assert ev_records[0].snippet == "replace Redis with PostgreSQL for our session storage"
    assert ev_records[0].verified
    assert ev_records[0].github_url == "https://github.com/decisio/backend/pull/101"

    # Verify governed file was persisted
    stmt_gf = select(GovernedFile).where(GovernedFile.decision_id == candidate.id)
    res_gf = await db_session.execute(stmt_gf)
    gf_records = res_gf.scalars().all()
    assert len(gf_records) == 1
    assert gf_records[0].file_path == "backend/app/session.py"

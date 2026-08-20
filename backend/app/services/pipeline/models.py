"""
Pipeline Data Models for Decisio.

Pydantic models that represent structured GitHub PR data as it flows through
the analysis pipeline. These are the lingua franca between pipeline stages —
each stage reads from and writes to these models.

Key design principle: preserve ALL useful data from GitHub. Never flatten
structured data into unstructured strings. The LLM sees a curated subset;
the verification and analysis stages see everything.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── GitHub Data Models (Stage 1-2: Ingest + Normalize) ───────


class PRFile(BaseModel):
    """A single file changed in a pull request.

    Preserves the full diff, addition/deletion counts, and blob SHA
    so verification and analysis stages can work deterministically.
    """

    filename: str
    status: str = ""  # "added", "modified", "removed", "renamed"
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    patch: str | None = None  # The actual unified diff
    sha: str | None = None  # Git blob SHA
    previous_filename: str | None = None  # For renames


class PRComment(BaseModel):
    """A review comment or issue comment on a pull request.

    Preserves the comment ID, author, timestamp, and the diff hunk context
    so we can trace evidence back to a specific GitHub URL and verify quotes.
    """

    id: int
    author: str
    body: str
    html_url: str
    created_at: str  # ISO 8601 timestamp
    comment_type: str = "review"  # "review" | "issue"
    # Review comment-specific fields (inline code comments)
    path: str | None = None  # File this comment is on
    diff_hunk: str | None = None  # Surrounding diff context
    commit_id: str | None = None


class PRCommit(BaseModel):
    """A commit in a pull request.

    Preserves SHA, message, and author so we can trace decisions
    back to specific commits and verify commit-based evidence.
    """

    sha: str
    message: str
    author: str
    timestamp: str  # ISO 8601
    html_url: str = ""


class PRLabel(BaseModel):
    """A label attached to a pull request."""

    name: str
    color: str = ""
    description: str | None = None


class PRContext(BaseModel):
    """Complete, structured representation of a GitHub PR.

    This is the central data model that flows through the pipeline.
    Every stage reads from it; ingestion writes it; nothing is discarded.

    Fields are organized by source:
    - Deterministic metadata (from GitHub API, always accurate)
    - Structured sub-objects (files, comments, commits)
    """

    # ── Deterministic PR metadata ────────────────────────────
    number: int
    title: str
    body: str = ""
    author: str = ""
    html_url: str = ""
    state: str = ""  # "open", "closed"
    merged: bool = False
    created_at: str | None = None  # ISO 8601
    merged_at: str | None = None
    updated_at: str | None = None

    # ── Repository context ───────────────────────────────────
    owner: str = ""
    repo: str = ""

    # ── Labels ───────────────────────────────────────────────
    labels: list[PRLabel] = Field(default_factory=list)

    # ── Structured sub-objects ───────────────────────────────
    files: list[PRFile] = Field(default_factory=list)
    review_comments: list[PRComment] = Field(default_factory=list)
    issue_comments: list[PRComment] = Field(default_factory=list)
    commits: list[PRCommit] = Field(default_factory=list)

    # ── Computed summaries (filled during normalization) ─────
    total_additions: int = 0
    total_deletions: int = 0
    total_files_changed: int = 0

    @property
    def all_comments(self) -> list[PRComment]:
        """All comments (review + issue) sorted by creation time."""
        return sorted(
            self.review_comments + self.issue_comments,
            key=lambda c: c.created_at,
        )

    @property
    def file_paths(self) -> list[str]:
        """Just the filenames, for quick access."""
        return [f.filename for f in self.files]

    @property
    def all_text_sources(self) -> dict[str, str]:
        """All text content keyed by source identifier, for evidence matching."""
        sources: dict[str, str] = {}
        if self.body:
            sources["pr_description"] = self.body
        for c in self.review_comments:
            sources[f"review_comment_{c.id}"] = c.body
        for c in self.issue_comments:
            sources[f"issue_comment_{c.id}"] = c.body
        for commit in self.commits:
            sources[f"commit_{commit.sha[:8]}"] = commit.message
        return sources


# ── Change Analysis Models (Stage 2: Change Analysis) ────────


class ChangeAnalysisResult(BaseModel):
    """Output of the deterministic change analysis stage."""

    change_types: list[str] = Field(default_factory=list)  # e.g., ["dependency_bump", "tests"]
    is_dependency_bump: bool = False
    is_docs_only: bool = False
    is_tests_only: bool = False
    is_config_only: bool = False
    is_bug_fix: bool = False
    is_refactor: bool = False
    is_security_related: bool = False
    is_performance_related: bool = False
    primary_change_type: str = "feature"
    summary: str = ""


# ── Detection Models (Stage 3: Candidate Detection) ─────────


class DetectionSignal(BaseModel):
    """A single signal from the heuristic detector.

    Positive signals suggest a decision exists; negative signals suggest
    the PR is routine (bump, docs, typo fix).
    """

    name: str  # e.g. "keyword_migrate", "docs_only", "dependency_bump"
    weight: float  # Positive = suggests decision, negative = suggests no decision
    reason: str  # Human-readable explanation


class DetectionResult(BaseModel):
    """Output of the detection stage."""

    decision_detected: bool = False
    confidence: int = 0  # 0 to 100
    candidate_reason: str = ""
    signals: list[str] = Field(default_factory=list)

    @property
    def has_candidate(self) -> bool:
        """Alias for backward compatibility."""
        return self.decision_detected


# ── Extraction Models (Stage 4: LLM Extraction) ─────────────


class ExtractedEvidence(BaseModel):
    """Evidence as returned by the LLM, before verification."""

    quote: str  # The exact text the LLM claims supports the decision
    source_description: str = ""  # LLM's description of where it came from
    # Filled in by verification stage:
    verified: bool = False
    matched_source_key: str | None = None  # Key from PRContext.all_text_sources
    matched_source_url: str | None = None  # GitHub URL
    match_score: float = 0.0  # Fuzzy match score (0.0-1.0)


class ExtractedDecision(BaseModel):
    """A decision as extracted by the LLM, before verification and enrichment."""

    title: str
    summary: str  # What was decided
    rationale: str = ""  # Why it was decided
    alternatives: str = ""  # Alternatives considered/discussed
    chosen_approach: str = ""  # The technical approach selected
    category: str = "other"
    evidence: list[ExtractedEvidence] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)  # Files proposed as relevant to this decision
    # LLM self-reported confidence (we calculate our own independently)
    llm_confidence: float = 0.0


# ── Analysis Models (Stage 5-6: Impact + Verification) ──────


class AffectedFile(BaseModel):
    """A file affected by a decision — from actual GitHub data, not LLM."""

    file_path: str
    status: str = ""  # "added", "modified", "removed"
    additions: int = 0
    deletions: int = 0
    relationship: str = "changed"  # "changed", "created", "deleted", "renamed"


class DetectedTechnology(BaseModel):
    """A technology detected in the PR, from deterministic analysis."""

    name: str  # e.g. "PostgreSQL", "Redis", "FastAPI"
    detection_source: str = ""  # How it was detected: "file_extension", "import", "config", "llm"
    relationship: str = "used"  # "adopted", "replaced", "deprecated", "configured", "used"


# ── Pipeline Log (Stage 7: Orchestration) ────────────────────


class PipelineStageStatus(str, enum.Enum):
    """Status of an individual pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class PipelineStage(BaseModel):
    """Log entry for a single pipeline stage execution."""

    name: str  # "ingest", "normalize", "detect", "extract", "analyze", "verify", "persist"
    status: PipelineStageStatus = PipelineStageStatus.PENDING
    duration_ms: float = 0.0
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


# ── Final Pipeline Result ────────────────────────────────────


class VerifiedDecision(BaseModel):
    """A fully processed decision — the final output of the pipeline.

    This is what gets persisted to the database and returned to the UI.
    Every field is either deterministic or verified.
    """

    title: str
    summary: str
    rationale: str = ""
    alternatives: str = ""
    chosen_approach: str = ""
    category: str = "other"

    # Evidence — each item has been checked against real PR data
    evidence: list[ExtractedEvidence] = Field(default_factory=list)

    # Affected files — from actual GitHub data
    affected_files: list[AffectedFile] = Field(default_factory=list)

    # Technologies — deterministically detected + LLM-assisted
    technologies: list[DetectedTechnology] = Field(default_factory=list)

    # Confidence — independently calculated, not LLM self-report
    confidence: float = 0.0
    llm_confidence: float = 0.0  # Preserved for comparison/debugging

    # Verification stats
    verified_evidence_count: int = 0
    total_evidence_count: int = 0


class PipelineResult(BaseModel):
    """Complete output of the analysis pipeline.

    Includes the decisions found (if any), the detection assessment,
    pipeline execution log, and the original PR context for reference.
    """

    # Core result
    has_significant_decision: bool = False
    decisions: list[VerifiedDecision] = Field(default_factory=list)

    # Change analysis (deterministic classification)
    change_analysis: ChangeAnalysisResult | None = None

    # Detection assessment
    detection: DetectionResult = Field(default_factory=DetectionResult)

    # Pipeline metadata
    pipeline_log: list[PipelineStage] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    provider_used: str = ""

    # Source context
    repository: str = ""
    pr_number: int = 0
    pr_title: str = ""
    pr_url: str = ""
    pr_author: str = ""

    # Persisted decision IDs (filled after database write)
    persisted_decision_ids: list[str] = Field(default_factory=list)

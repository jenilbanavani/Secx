from __future__ import annotations

"""
SyncJob model — tracks the progress and outcome of a repository sync.

Each time a user clicks "Sync" (or a scheduled sync runs), a SyncJob is created.
This gives us:
- Progress tracking: "5/42 PRs processed"
- Error diagnosis: if a sync fails, we know where it stopped
- Audit trail: when was each sync, how many decisions were extracted
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin

import enum


class SyncStatus(str, enum.Enum):
    """Lifecycle states of a sync job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SyncJob(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "sync_jobs"

    # ── Parent ───────────────────────────────────────────
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Status ───────────────────────────────────────────
    status: Mapped[SyncStatus] = mapped_column(
        SQLEnum(SyncStatus, native_enum=False),
        default=SyncStatus.PENDING,
        nullable=False,
    )

    # ── Progress counters ────────────────────────────────
    prs_fetched: Mapped[int] = mapped_column(Integer, default=0)
    prs_processed: Mapped[int] = mapped_column(Integer, default=0)
    candidates_extracted: Mapped[int] = mapped_column(Integer, default=0)

    # ── Timing ───────────────────────────────────────────
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Error tracking ───────────────────────────────────
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Relationships ────────────────────────────────────
    repository: Mapped["Repository"] = relationship(
        "Repository", back_populates="sync_jobs"
    )
    candidates: Mapped[list["DecisionCandidate"]] = relationship(
        "DecisionCandidate", back_populates="sync_job"
    )

    def __repr__(self) -> str:
        return f"<SyncJob {self.status.value} repo={self.repository_id} ({self.id})>"

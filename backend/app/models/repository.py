from __future__ import annotations

"""
Repository model — a GitHub repo connected to Decisio.

Tracks sync state so we can resume from the last processed PR
instead of re-fetching everything on each sync.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class Repository(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "repositories"

    # ── Owner ────────────────────────────────────────────
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── GitHub metadata ──────────────────────────────────
    github_id: Mapped[int] = mapped_column(unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )  # e.g. "acme/backend"
    default_branch: Mapped[str] = mapped_column(String(100), default="main")
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_private: Mapped[bool] = mapped_column(default=False)

    # ── Sync state ───────────────────────────────────────
    # The "cursor" for incremental sync: the number of the last PR we processed.
    # On the next sync, we only fetch PRs with number > sync_cursor.
    sync_cursor: Mapped[int] = mapped_column(Integer, default=0)
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Relationships ────────────────────────────────────
    user: Mapped["User"] = relationship("User", back_populates="repositories")
    sync_jobs: Mapped[list["SyncJob"]] = relationship(
        "SyncJob", back_populates="repository", cascade="all, delete-orphan"
    )
    decisions: Mapped[list["DecisionCandidate"]] = relationship(
        "DecisionCandidate", back_populates="repository", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Repository {self.full_name} ({self.id})>"

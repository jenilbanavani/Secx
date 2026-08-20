from __future__ import annotations

"""
User model — represents a Decisio user (authenticated via GitHub OAuth).

We store the GitHub login and an encrypted access token.
The token is needed to fetch PRs from the user's repositories.
"""

import uuid

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    # GitHub identity
    github_id: Mapped[int] = mapped_column(unique=True, nullable=False, index=True)
    github_login: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    github_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    github_avatar_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    github_email: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Encrypted GitHub access token (encrypted at rest via Fernet)
    github_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Relationships ────────────────────────────────────
    repositories: Mapped[list["Repository"]] = relationship(
        "Repository", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.github_login} ({self.id})>"

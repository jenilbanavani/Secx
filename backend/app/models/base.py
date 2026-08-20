"""
Shared base class and mixins for all SQLAlchemy models.

Every model gets:
- A UUID primary key (auto-generated, no sequential IDs to leak)
- created_at / updated_at timestamps (auto-managed)
- A sensible __repr__ for debugging
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base that all models inherit from.

    Alembic uses this to auto-detect models for migration generation.
    """
    pass


class TimestampMixin:
    """Adds created_at and updated_at to any model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class UUIDMixin:
    """Adds a UUID primary key to any model.

    UUIDs over auto-increment because:
    - No ID guessing/enumeration attacks
    - Safe for distributed systems (no coordination needed)
    - Can generate IDs client-side if needed
    """

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
    )

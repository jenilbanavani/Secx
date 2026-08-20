"""
Model registry — import all models here so Alembic can auto-detect them.

When you add a new model file, import it here. Alembic's `target_metadata`
points to `Base.metadata`, and it only knows about models that have been
imported (and thus registered on the Base).
"""

from app.models.base import Base
from app.models.user import User
from app.models.repository import Repository
from app.models.sync_job import SyncJob, SyncStatus
from app.models.decision import (
    DecisionCandidate,
    DecisionCategory,
    DecisionStatus,
    Evidence,
    GovernedFile,
    SourceType,
)

__all__ = [
    "Base",
    "User",
    "Repository",
    "SyncJob",
    "SyncStatus",
    "DecisionCandidate",
    "DecisionCategory",
    "DecisionStatus",
    "Evidence",
    "GovernedFile",
    "SourceType",
]

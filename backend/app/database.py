"""
Database engine, session factory, and dependency injection for FastAPI.

Key design decisions:
- Async everywhere: using async engine + AsyncSession for non-blocking I/O.
- SQLite for local dev: aiosqlite driver, no setup required.
- Postgres for production: asyncpg driver, pgvector for embeddings.
- Session-per-request: each API request gets its own session via `get_db()`.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import get_settings


def _build_engine():
    """Create the async engine with driver-appropriate settings."""
    settings = get_settings()
    url = settings.database_url

    if settings.is_sqlite:
        # SQLite needs special handling for async:
        # - check_same_thread=False: allow cross-thread access (uvicorn uses threadpool)
        # - StaticPool: reuse the same connection (SQLite doesn't do real connection pooling)
        return create_async_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=settings.debug,
        )
    else:
        # Postgres: real connection pool with sensible defaults
        return create_async_engine(
            url,
            pool_size=5,
            max_overflow=10,
            echo=settings.debug,
        )


engine = _build_engine()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # prevents lazy-load issues after commit
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides a database session.

    Usage in a route:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...

    The session is automatically closed when the request completes.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

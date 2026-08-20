"""
Pytest configuration and fixtures for Decisio backend using anyio.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.database import get_db
from app.main import create_app
from app.models import Base


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def test_settings() -> Settings:
    """Return test settings with an in-memory SQLite database."""
    return Settings(
        app_name="Decisio-Test",
        debug=True,
        database_url="sqlite+aiosqlite:///:memory:",
        secret_key="test-secret-key-for-testing-only",
    )


@pytest.fixture
async def test_engine(test_settings: Settings):
    """Create an isolated in-memory test database engine."""
    engine = create_async_engine(
        test_settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    """Create a database session for test cases."""
    session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session


@pytest.fixture
async def async_client(test_settings: Settings, test_engine, db_session: AsyncSession):
    """Async HTTP test client bound to FastAPI application."""
    app = create_app()

    # Override get_db dependency to use the test session
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

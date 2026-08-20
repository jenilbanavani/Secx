import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    DecisionCandidate,
    DecisionCategory,
    DecisionStatus,
    Repository,
    SourceType,
    User,
)


@pytest.mark.anyio
async def test_create_user_and_repository(db_session: AsyncSession):
    """Test creating and querying a User and Repository model."""
    user = User(
        github_id=123456,
        github_login="octocat",
        github_email="octocat@github.com",
        github_token_encrypted="encrypted-dummy-token",
    )
    db_session.add(user)
    await db_session.flush()

    repo = Repository(
        user_id=user.id,
        github_id=98765,
        full_name="octocat/hello-world",
        default_branch="main",
    )
    db_session.add(repo)
    await db_session.commit()

    # Query back
    result = await db_session.execute(select(User).where(User.github_login == "octocat"))
    queried_user = result.scalar_one_or_none()
    assert queried_user is not None
    assert queried_user.github_email == "octocat@github.com"

    result = await db_session.execute(
        select(Repository).where(Repository.full_name == "octocat/hello-world")
    )
    queried_repo = result.scalar_one_or_none()
    assert queried_repo is not None
    assert queried_repo.user_id == user.id


@pytest.mark.anyio
async def test_create_decision_candidate(db_session: AsyncSession):
    """Test creating a DecisionCandidate record."""
    user = User(
        github_id=99999,
        github_login="decisio-admin",
        github_token_encrypted="token",
    )
    db_session.add(user)
    await db_session.flush()

    repo = Repository(
        user_id=user.id,
        github_id=11111,
        full_name="decisio-org/backend-core",
    )
    db_session.add(repo)
    await db_session.flush()

    decision = DecisionCandidate(
        repository_id=repo.id,
        title="Migrate to FastAPI and SQLAlchemy async",
        description="Adopted FastAPI for high performance async operations.",
        status=DecisionStatus.APPROVED,
        confidence_score=0.95,
        category=DecisionCategory.ARCHITECTURE,
        source_type=SourceType.MANUAL,
    )
    db_session.add(decision)
    await db_session.commit()

    result = await db_session.execute(
        select(DecisionCandidate).where(DecisionCandidate.repository_id == repo.id)
    )
    queried_decision = result.scalar_one_or_none()
    assert queried_decision is not None
    assert queried_decision.title == "Migrate to FastAPI and SQLAlchemy async"
    assert queried_decision.status == DecisionStatus.APPROVED
    assert queried_decision.category == DecisionCategory.ARCHITECTURE

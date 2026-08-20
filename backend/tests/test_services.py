import pytest
from httpx import AsyncClient

from app.services.decision_extractor import DecisionExtractor
from app.services.github_service import GitHubService
from app.services.llm_client import LLMClient


@pytest.mark.anyio
async def test_llm_client_provider_status():
    client = LLMClient()
    status = client.get_provider_status()
    assert "active_provider" in status
    assert "providers" in status
    assert "mock" in status["providers"]
    assert "groq" in status["providers"]
    assert "grok" in status["providers"]
    assert "anthropic" in status["providers"]
    assert "openai" in status["providers"]


@pytest.mark.anyio
async def test_llm_client_mock_generation():
    client = LLMClient()
    result = await client.generate_json(
        system_prompt="Extract decisions",
        user_prompt="We decided to migrate database layer to SQLAlchemy async + aiosqlite",
        provider="mock",
    )
    assert "decisions" in result
    assert len(result["decisions"]) >= 1
    assert "technology_selection" in result["decisions"][0]["category"]


@pytest.mark.anyio
async def test_decision_extractor():
    extractor = DecisionExtractor()
    mock_pr = GitHubService.get_mock_pull_request()
    decisions = await extractor.extract_from_pull_request(mock_pr, provider="mock")
    assert len(decisions) >= 1
    assert "evidence" in decisions[0]
    assert "governed_files" in decisions[0]


@pytest.mark.anyio
async def test_api_llm_providers_endpoint(async_client: AsyncClient):
    response = await async_client.get("/api/v1/llm/providers")
    assert response.status_code == 200
    data = response.json()
    assert "active_provider" in data
    assert "grok" in data["providers"]


@pytest.mark.anyio
async def test_api_raw_decision_extraction(async_client: AsyncClient):
    payload = {
        "title": "Switch from REST to GraphQL for client query optimization",
        "description": "We decided to adopt GraphQL schemas to allow mobile clients to query arbitrary field subsets without over-fetching.",
        "changed_files": ["backend/schema.graphql", "backend/resolvers.py"],
        "comments": ["architect: This solves the mobile bandwidth issue."],
        "provider": "mock",
    }
    response = await async_client.post("/api/v1/decisions/extract-raw", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decisions_count"] >= 1
    assert "decisions" in data

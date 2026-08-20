"""
API Routes for Decisio.

Includes:
- Pipeline-based decision extraction from GitHub PRs
- Legacy extraction endpoints (backward compatible)
- LLM provider status
- Decision management (retrieve, approve, reject)
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.decision_extractor import DecisionExtractor
from app.services.github_service import GitHubService
from app.services.llm_client import LLMClient
from app.services.pipeline.orchestrator import AnalysisPipeline

api_router = APIRouter()
llm_client = LLMClient()
github_service = GitHubService()
decision_extractor = DecisionExtractor()
pipeline = AnalysisPipeline()


# ── Schemas ──────────────────────────────────────────────


class ExtractPreviewRequest(BaseModel):
    owner: str = Field(..., examples=["fastapi"])
    repo: str = Field(..., examples=["fastapi"])
    pull_number: int = Field(..., examples=[1234])
    provider: Literal["groq", "mock", "openai", "anthropic", "grok"] | None = Field(
        None, description="LLM provider override (default: configured in .env)"
    )
    model: str | None = Field(None, description="Specific model name (optional)")


class RawExtractRequest(BaseModel):
    title: str = Field(..., examples=["Adopt Redis for distributed session storage"])
    description: str = Field(
        ...,
        examples=[
            "We replaced in-memory dictionary cache with Redis cluster to allow multi-instance horizontal scaling."
        ],
    )
    changed_files: list[str] = Field(
        default=["backend/app/cache.py", "backend/app/main.py"]
    )
    comments: list[str] = Field(
        default=["reviewer: Ensure connection pooling is configured with retry logic."]
    )
    provider: Literal["groq", "mock", "openai", "anthropic", "grok"] | None = None
    model: str | None = None


# ── Endpoints ────────────────────────────────────────────


@api_router.get("/llm/providers", tags=["LLM & Intelligence"])
async def list_llm_providers():
    """List available LLM providers (OpenAI, Claude, Grok, Mock) and current configuration status."""
    return llm_client.get_provider_status()


@api_router.get("/github/mock-pr", tags=["GitHub Integration"])
async def get_mock_pr():
    """Get a pre-built mock PR payload for zero-token local testing."""
    return GitHubService.get_mock_pull_request()


@api_router.get("/github/pr", tags=["GitHub Integration"])
async def fetch_github_pr(
    owner: str = Query(..., examples=["fastapi"]),
    repo: str = Query(..., examples=["fastapi"]),
    pull_number: int = Query(..., examples=[1234]),
):
    """Fetch any public GitHub Pull Request without needing OAuth tokens."""
    try:
        pr_data = await github_service.get_pull_request(owner, repo, pull_number)
        return pr_data
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch PR #{pull_number} from {owner}/{repo}: {str(e)}",
        )


@api_router.post("/decisions/analyze", tags=["Decision Intelligence"])
async def analyze_github_pr(
    request: ExtractPreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Analyze a GitHub PR through the full decision intelligence pipeline.

    Pipeline stages:
    1. Ingest — fetch rich PR data (diffs, comments, commits)
    2. Detect — heuristic pre-filter for decision candidates
    3. Extract — focused LLM extraction (only if candidate detected)
    4. Verify — match evidence against real PR data
    5. Analyze — deterministic impact analysis (files, technologies)

    Returns structured decisions with verified evidence and pipeline log.
    """
    try:
        result = await pipeline.analyze(
            owner=request.owner,
            repo=request.repo,
            pull_number=request.pull_number,
            provider=request.provider,
            model=request.model,
            db=db,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline analysis failed: {str(e)}",
        )

    return result.model_dump()


@api_router.post("/decisions/preview-from-github", tags=["Decision Intelligence"])
async def extract_decisions_from_github_pr(
    request: ExtractPreviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Fetch a public GitHub PR and extract architectural decisions.

    This endpoint now uses the full pipeline internally but returns
    a backward-compatible response shape for existing UI consumers.
    """
    try:
        result = await pipeline.analyze(
            owner=request.owner,
            repo=request.repo,
            pull_number=request.pull_number,
            provider=request.provider,
            model=request.model,
            db=db,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Decision extraction failed: {str(e)}",
        )

    # Return the full pipeline result — the UI will use the new structure
    return result.model_dump()


@api_router.post("/decisions/extract-raw", tags=["Decision Intelligence"])
async def extract_from_raw_text(request: RawExtractRequest):
    """Test decision extraction directly from any PR text using OpenAI, Claude, Grok, or Mock."""
    pr_data = {
        "number": 1,
        "title": request.title,
        "body": request.description,
        "author": "developer",
        "html_url": "https://github.com/example/repo/pull/1",
        "changed_files": request.changed_files,
        "comments": request.comments,
    }

    try:
        decisions = await decision_extractor.extract_from_pull_request(
            pr_data, provider=request.provider, model=request.model
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Decision extraction failed: {str(e)}",
        )

    return {
        "provider_used": request.provider or llm_client.settings.llm_provider,
        "decisions_count": len(decisions),
        "decisions": decisions,
    }

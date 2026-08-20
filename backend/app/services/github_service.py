"""
GitHub API Service for Decisio.

Fetches Pull Requests, commit histories, changed files, and review discussions.
Supports:
1. Public repositories WITHOUT any token (unauthenticated GitHub REST API)
2. Authenticated requests with Personal Access Token or OAuth Bearer token
3. Mock PR generator for offline and unit testing
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class GitHubService:
    """Async client for interacting with the GitHub REST API."""

    def __init__(self, settings: Settings | None = None, token: str | None = None):
        self.settings = settings or get_settings()
        self.token = token or self.settings.github_token
        self.base_url = "https://api.github.com"

    def _get_headers(self) -> dict[str, str]:
        """Construct standard GitHub REST API headers."""
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Decisio-App/0.1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get_repository(self, owner: str, repo: str) -> dict[str, Any]:
        """Fetch metadata for a GitHub repository."""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=self._get_headers())
            resp.raise_for_status()
            return resp.json()

    async def get_pull_request(self, owner: str, repo: str, pull_number: int) -> dict[str, Any]:
        """Fetch details of a single pull request including changed files and comments."""
        async with httpx.AsyncClient(timeout=20.0) as client:
            # 1. PR details
            pr_url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}"
            pr_resp = await client.get(pr_url, headers=self._get_headers())
            pr_resp.raise_for_status()
            pr_data = pr_resp.json()

            # 2. Changed files
            files_url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}/files"
            files_resp = await client.get(files_url, headers=self._get_headers())
            files_data = files_resp.json() if files_resp.status_code == 200 else []

            # 3. Review comments
            comments_url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}/comments"
            comments_resp = await client.get(comments_url, headers=self._get_headers())
            comments_data = comments_resp.json() if comments_resp.status_code == 200 else []

            changed_files = [f.get("filename", "") for f in files_data if "filename" in f]
            comments_text = [
                f"{c.get('user', {}).get('login', 'User')}: {c.get('body', '')}"
                for c in comments_data
                if c.get("body")
            ]

            return {
                "number": pr_data.get("number"),
                "title": pr_data.get("title", ""),
                "body": pr_data.get("body") or "",
                "author": pr_data.get("user", {}).get("login", ""),
                "html_url": pr_data.get("html_url", ""),
                "merged": pr_data.get("merged", False),
                "state": pr_data.get("state", ""),
                "changed_files": changed_files,
                "comments": comments_text,
            }

    async def list_recent_pull_requests(
        self, owner: str, repo: str, state: str = "closed", per_page: int = 10
    ) -> list[dict[str, Any]]:
        """List recent pull requests for a repository."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {"state": state, "per_page": per_page, "sort": "updated", "direction": "desc"}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=self._get_headers(), params=params)
            resp.raise_for_status()
            prs = resp.json()
            return [
                {
                    "number": pr.get("number"),
                    "title": pr.get("title"),
                    "body": pr.get("body") or "",
                    "html_url": pr.get("html_url"),
                    "merged": pr.get("merged_at") is not None,
                    "created_at": pr.get("created_at"),
                }
                for pr in prs
            ]

    @staticmethod
    def get_mock_pull_request() -> dict[str, Any]:
        """Generate sample PR data for zero-token local testing."""
        return {
            "number": 42,
            "title": "Migrate database layer to SQLAlchemy async + aiosqlite",
            "body": (
                "## Summary of Changes\n"
                "- Replace synchronous sqlite3 connections with AsyncSession and aiosqlite.\n"
                "- We decided to adopt async everywhere to prevent event loop blocking under heavy query load.\n"
                "- Deprecated legacy raw SQL helpers in favor of declarative ORM models.\n"
                "## Trade-offs Considered\n"
                "- Async SQLite uses StaticPool which requires careful connection lifecycle management, "
                "but provides full compatibility with our upcoming PostgreSQL production migration."
            ),
            "author": "lead-architect",
            "html_url": "https://github.com/decisio/backend/pull/42",
            "merged": True,
            "state": "closed",
            "changed_files": [
                "backend/app/database.py",
                "backend/app/config.py",
                "backend/app/models/base.py",
                "backend/app/models/decision.py",
            ],
            "comments": [
                "teammate-1: The async migration makes sense. Did we verify cross-thread connection behavior?",
                "lead-architect: Yes, check_same_thread=False is enabled for SQLite StaticPool.",
            ],
        }

"""
Rich GitHub Ingestion for Decisio Pipeline.

Fetches ALL useful data from a GitHub Pull Request and returns a structured
PRContext object. Unlike the original GitHubService.get_pull_request() which
discarded diffs, SHAs, timestamps, and comment metadata, this module
preserves everything the pipeline needs for verification and analysis.

Design principles:
- Never discard data that might be useful downstream
- Structure everything — no flattening to strings
- Handle API failures gracefully per-resource (if commits fail, still return PR + files)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.services.pipeline.models import (
    PRComment,
    PRCommit,
    PRContext,
    PRFile,
    PRLabel,
)

logger = logging.getLogger(__name__)


class PRIngestion:
    """Fetches and structures complete PR data from GitHub API."""

    def __init__(self, settings: Settings | None = None, token: str | None = None):
        self.settings = settings or get_settings()
        self.token = token or self.settings.github_token
        self.base_url = "https://api.github.com"

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "Decisio-App/0.2.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def ingest(self, owner: str, repo: str, pull_number: int) -> PRContext:
        """Fetch a complete PR with all structured sub-objects.

        Makes parallel-ish requests for PR metadata, files, review comments,
        issue comments, and commits. Each sub-resource is fetched independently
        so a failure in one doesn't block the others.
        """
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=25.0) as client:
            # 1. PR metadata (required — fail if this fails)
            pr_url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}"
            pr_resp = await client.get(pr_url, headers=headers)
            pr_resp.raise_for_status()
            pr_data = pr_resp.json()

            # 2. Changed files (with diffs, SHAs, stats)
            files = await self._fetch_files(client, owner, repo, pull_number, headers)

            # 3. Review comments (inline code comments)
            review_comments = await self._fetch_review_comments(
                client, owner, repo, pull_number, headers
            )

            # 4. Issue comments (general discussion)
            issue_comments = await self._fetch_issue_comments(
                client, owner, repo, pull_number, headers
            )

            # 5. Commits
            commits = await self._fetch_commits(
                client, owner, repo, pull_number, headers
            )

            # 6. Labels
            labels = [
                PRLabel(
                    name=label.get("name", ""),
                    color=label.get("color", ""),
                    description=label.get("description"),
                )
                for label in pr_data.get("labels", [])
            ]

            # Build the complete context
            total_additions = sum(f.additions for f in files)
            total_deletions = sum(f.deletions for f in files)

            return PRContext(
                number=pr_data.get("number", pull_number),
                title=pr_data.get("title", ""),
                body=pr_data.get("body") or "",
                author=pr_data.get("user", {}).get("login", ""),
                html_url=pr_data.get("html_url", ""),
                state=pr_data.get("state", ""),
                merged=pr_data.get("merged", False),
                created_at=pr_data.get("created_at"),
                merged_at=pr_data.get("merged_at"),
                updated_at=pr_data.get("updated_at"),
                owner=owner,
                repo=repo,
                labels=labels,
                files=files,
                review_comments=review_comments,
                issue_comments=issue_comments,
                commits=commits,
                total_additions=total_additions,
                total_deletions=total_deletions,
                total_files_changed=len(files),
            )

    async def _fetch_files(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        pull_number: int,
        headers: dict,
    ) -> list[PRFile]:
        """Fetch changed files with diffs, SHAs, and stats."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}/files"
            # GitHub paginates at 30 files by default; request up to 100
            resp = await client.get(url, headers=headers, params={"per_page": 100})
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch files: {resp.status_code}")
                return []
            return [
                PRFile(
                    filename=f.get("filename", ""),
                    status=f.get("status", ""),
                    additions=f.get("additions", 0),
                    deletions=f.get("deletions", 0),
                    changes=f.get("changes", 0),
                    patch=f.get("patch"),  # The actual diff — preserved!
                    sha=f.get("sha"),
                    previous_filename=f.get("previous_filename"),
                )
                for f in resp.json()
                if isinstance(f, dict)
            ]
        except Exception as e:
            logger.warning(f"Error fetching PR files: {e}")
            return []

    async def _fetch_review_comments(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        pull_number: int,
        headers: dict,
    ) -> list[PRComment]:
        """Fetch inline review comments with full metadata."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}/comments"
            resp = await client.get(url, headers=headers, params={"per_page": 100})
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch review comments: {resp.status_code}")
                return []
            return [
                PRComment(
                    id=c.get("id", 0),
                    author=c.get("user", {}).get("login", ""),
                    body=c.get("body", ""),
                    html_url=c.get("html_url", ""),
                    created_at=c.get("created_at", ""),
                    comment_type="review",
                    path=c.get("path"),
                    diff_hunk=c.get("diff_hunk"),
                    commit_id=c.get("commit_id"),
                )
                for c in resp.json()
                if isinstance(c, dict) and c.get("body")
            ]
        except Exception as e:
            logger.warning(f"Error fetching review comments: {e}")
            return []

    async def _fetch_issue_comments(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        pull_number: int,
        headers: dict,
    ) -> list[PRComment]:
        """Fetch general issue-level comments (not inline reviews)."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues/{pull_number}/comments"
            resp = await client.get(url, headers=headers, params={"per_page": 100})
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch issue comments: {resp.status_code}")
                return []
            return [
                PRComment(
                    id=c.get("id", 0),
                    author=c.get("user", {}).get("login", ""),
                    body=c.get("body", ""),
                    html_url=c.get("html_url", ""),
                    created_at=c.get("created_at", ""),
                    comment_type="issue",
                )
                for c in resp.json()
                if isinstance(c, dict) and c.get("body")
            ]
        except Exception as e:
            logger.warning(f"Error fetching issue comments: {e}")
            return []

    async def _fetch_commits(
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo: str,
        pull_number: int,
        headers: dict,
    ) -> list[PRCommit]:
        """Fetch commits in the PR with SHA, message, author."""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pull_number}/commits"
            resp = await client.get(url, headers=headers, params={"per_page": 100})
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch commits: {resp.status_code}")
                return []
            return [
                PRCommit(
                    sha=c.get("sha", ""),
                    message=c.get("commit", {}).get("message", ""),
                    author=(
                        c.get("author", {}) or {}
                    ).get("login", "")
                    or c.get("commit", {}).get("author", {}).get("name", ""),
                    timestamp=c.get("commit", {}).get("author", {}).get("date", ""),
                    html_url=c.get("html_url", ""),
                )
                for c in resp.json()
                if isinstance(c, dict)
            ]
        except Exception as e:
            logger.warning(f"Error fetching commits: {e}")
            return []

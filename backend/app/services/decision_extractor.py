"""
Decision Extractor Service for Decisio.

Analyzes Pull Request content (title, description, discussion comments, changed files)
using LLMs (OpenAI, Claude, Grok, or Mock) to extract engineering decisions,
rationale, evidence snippets, and affected code paths.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings, get_settings
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert Principal Software Architect analyzing GitHub Pull Requests and technical discussions.
Your task is to identify and extract concrete architectural and engineering decisions.

Look for:
1. Technology selection (e.g. database choice, library adoption, protocol selection)
2. Pattern adoption (e.g. CQRS, repository pattern, event-driven architecture)
3. Refactoring decisions (e.g. splitting monolith, modularizing services)
4. Deprecations (e.g. sunsetting old endpoints or modules)
5. Technical trade-offs (e.g. consistency vs latency, caching strategies)

Return a JSON object matching this schema:
{
  "decisions": [
    {
      "title": "Short, clear title describing the decision (max 100 chars)",
      "description": "Comprehensive explanation of what was decided, why it was chosen, and what trade-offs were accepted.",
      "category": "technology_selection | pattern_adoption | refactoring | deprecation | tradeoff | architecture | other",
      "confidence_score": 0.0 to 1.0,
      "evidence": [
        {
          "snippet": "Exact quote from PR description or comments supporting this decision",
          "source_location": "PR Description / Comment by @user"
        }
      ],
      "governed_files": [
        "path/to/file1.py",
        "path/to/file2.ts"
      ]
    }
  ]
}

If no meaningful architectural decisions were made (e.g. minor typo fix or pure cosmetic changes), return:
{ "decisions": [] }
"""


class DecisionExtractor:
    """Extracts decisions from PR metadata using LLMClient."""

    def __init__(self, settings: Settings | None = None, llm_client: LLMClient | None = None):
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings)

    async def extract_from_pull_request(
        self,
        pr_data: dict[str, Any],
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze a pull request dictionary and extract architectural decision candidates."""
        user_prompt = self._build_prompt(pr_data)
        result = await self.llm_client.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
        )

        decisions = result.get("decisions", [])
        return decisions

    def _build_prompt(self, pr: dict[str, Any]) -> str:
        """Format PR metadata into a structured prompt."""
        sections = [
            f"Pull Request #{pr.get('number')}: {pr.get('title')}",
            f"Author: {pr.get('author')}",
            f"URL: {pr.get('html_url')}",
            "\n--- PR DESCRIPTION ---",
            pr.get("body") or "(No description provided)",
        ]

        changed_files = pr.get("changed_files", [])
        if changed_files:
            sections.append("\n--- CHANGED FILES ---")
            sections.extend(f"- {f}" for f in changed_files[:30])

        comments = pr.get("comments", [])
        if comments:
            sections.append("\n--- REVIEW DISCUSSIONS ---")
            sections.extend(f"> {c}" for c in comments[:10])

        return "\n".join(sections)

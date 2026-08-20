"""
Focused LLM Extraction for Decisio Pipeline.

Unlike the original monolithic prompt that asked the LLM to do everything at once,
this module uses targeted prompts for specific tasks:

1. Decision extraction: "What was decided and why?"
2. Evidence quoting: "Quote the exact text that supports this."

The LLM is NOT asked to:
- Determine affected files (we know those from GitHub)
- Calculate confidence (we compute it independently)
- Invent evidence (we verify everything against real data)

The prompt is constructed from the PRContext, including actual diffs
and comments — not just flattened text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings, get_settings
from app.services.llm_client import LLMClient
from app.services.pipeline.models import (
    ExtractedDecision,
    ExtractedEvidence,
    PRContext,
)

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """\
You are an expert software architect analyzing a GitHub Pull Request to identify \
concrete technical or architectural decisions.

Your job is to identify ONLY genuine decisions — choices that affect the project's \
architecture, technology stack, design patterns, or engineering approach.

Rules:
1. If the PR does not contain a meaningful technical decision, return {"decisions": []}
2. Do NOT invent or fabricate decisions. Only report what is explicitly stated or \
clearly implied by the PR content.
3. For evidence, quote EXACT text from the PR description, comments, or commit messages. \
Do not paraphrase or make up quotes.
4. A dependency version bump without any architectural reasoning is NOT a decision.
5. A bug fix without any design change is NOT a decision.
6. Focus on WHAT was decided and WHY — the rationale matters more than description.
7. Identify which specific files changed in the PR are relevant to the decision. Do not guess files that were not changed.

Return a JSON object:
{
  "decisions": [
    {
      "title": "Short clear title (max 100 chars)",
      "summary": "What was decided — the concrete technical choice.",
      "rationale": "Why it was decided — the reasoning, trade-offs, motivations.",
      "alternatives": "What other options or alternatives were considered or discussed.",
      "chosen_approach": "Detailed explanation of the technical approach selected.",
      "category": "technology_selection|pattern_adoption|refactoring|deprecation|tradeoff|architecture|other",
      "relevant_files": [
        "path/to/relevant_file1.py",
        "path/to/relevant_file2.ts"
      ],
      "evidence": [
        {
          "quote": "Exact text from the PR that supports this decision",
          "source_description": "PR description | Review comment by @username | Commit message SHA"
        }
      ],
      "confidence": 0.0 to 1.0
    }
  ]
}

If no meaningful decision exists, return: {"decisions": []}
"""


class DecisionExtractionService:
    """Extracts decisions from a PRContext using focused LLM prompts."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings)

    async def extract(
        self,
        ctx: PRContext,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[ExtractedDecision]:
        """Extract decisions from a PRContext using focused LLM prompts.

        Returns a list of ExtractedDecision objects with unverified evidence.
        Verification happens in the next pipeline stage.
        """
        user_prompt = self._build_focused_prompt(ctx)

        result = await self.llm_client.generate_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
        )

        return self._parse_response(result)

    def _build_focused_prompt(self, ctx: PRContext) -> str:
        """Build a structured prompt from the PRContext.

        Includes actual content but organized so the LLM can find
        decision-relevant information efficiently. Does NOT dump
        everything blindly — large diffs are truncated, and the
        focus is on description, discussion, and commit messages.
        """
        sections: list[str] = []

        # PR metadata (deterministic context)
        sections.append(f"# Pull Request #{ctx.number}: {ctx.title}")
        sections.append(f"Repository: {ctx.owner}/{ctx.repo}")
        sections.append(f"Author: {ctx.author}")
        sections.append(f"State: {ctx.state} | Merged: {ctx.merged}")
        if ctx.labels:
            sections.append(f"Labels: {', '.join(l.name for l in ctx.labels)}")

        # PR description — the most important source
        sections.append("\n## PR Description")
        if ctx.body:
            # Truncate very long descriptions but preserve the important parts
            body = ctx.body[:4000] if len(ctx.body) > 4000 else ctx.body
            sections.append(body)
        else:
            sections.append("(No description provided)")

        # Commit messages — often contain decision rationale
        if ctx.commits:
            sections.append("\n## Commit Messages")
            for commit in ctx.commits[:15]:  # Cap at 15 commits
                msg = commit.message[:300] if len(commit.message) > 300 else commit.message
                sections.append(f"- [{commit.sha[:8]}] {msg}")

        # Changed files — summary (not full diffs)
        if ctx.files:
            sections.append(f"\n## Changed Files ({ctx.total_files_changed} files, "
                            f"+{ctx.total_additions}/-{ctx.total_deletions} lines)")
            for f in ctx.files[:40]:
                sections.append(
                    f"- {f.filename} ({f.status}: +{f.additions}/-{f.deletions})"
                )

        # Review discussion — often contains the decision reasoning
        if ctx.review_comments:
            sections.append(f"\n## Review Comments ({len(ctx.review_comments)} comments)")
            for c in ctx.review_comments[:15]:
                body = c.body[:500] if len(c.body) > 500 else c.body
                file_ctx = f" [on {c.path}]" if c.path else ""
                sections.append(f"> @{c.author}{file_ctx}: {body}")

        # Issue comments — general discussion
        if ctx.issue_comments:
            sections.append(f"\n## Discussion ({len(ctx.issue_comments)} comments)")
            for c in ctx.issue_comments[:10]:
                body = c.body[:500] if len(c.body) > 500 else c.body
                sections.append(f"> @{c.author}: {body}")

        return "\n".join(sections)

    def _parse_response(self, result: dict[str, Any]) -> list[ExtractedDecision]:
        """Parse the LLM response into structured ExtractedDecision objects."""
        decisions_raw = result.get("decisions", [])

        if not isinstance(decisions_raw, list):
            logger.warning(f"LLM returned non-list 'decisions': {type(decisions_raw)}")
            return []

        decisions: list[ExtractedDecision] = []
        for d in decisions_raw:
            if not isinstance(d, dict):
                continue

            evidence = []
            for e in d.get("evidence", []):
                if isinstance(e, dict) and e.get("quote"):
                    evidence.append(
                        ExtractedEvidence(
                            quote=str(e.get("quote", "")),
                            source_description=str(e.get("source_description", "")),
                        )
                    )

            relevant_files = []
            for f in d.get("relevant_files", []):
                if isinstance(f, str):
                    relevant_files.append(f)

            decisions.append(
                ExtractedDecision(
                    title=str(d.get("title", "Untitled Decision")),
                    summary=str(d.get("summary", d.get("description", ""))),
                    rationale=str(d.get("rationale", "")),
                    alternatives=str(d.get("alternatives", "")),
                    chosen_approach=str(d.get("chosen_approach", "")),
                    category=str(d.get("category", "other")).lower(),
                    evidence=evidence,
                    relevant_files=relevant_files,
                    llm_confidence=float(d.get("confidence", d.get("confidence_score", 0.5))),
                )
            )

        return decisions

"""
Evidence Verification for Decisio Pipeline.

This is the critical stage that separates Decisio from an LLM wrapper.
For every evidence snippet the LLM claims supports a decision, we verify
whether that text actually exists in the real GitHub PR data.

Verification approach:
1. Exact substring match first (fast, high confidence)
2. Normalized match (case-insensitive, whitespace-normalized)
3. Fuzzy match using sequence matching (handles minor paraphrasing)

If the LLM fabricated a quote, it gets flagged as unverified.
If a quote matches real PR text, it gets linked to the specific GitHub source URL.

Never allow the model to fabricate evidence.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from app.services.pipeline.models import (
    ExtractedDecision,
    ExtractedEvidence,
    PRContext,
)

logger = logging.getLogger(__name__)

# Minimum fuzzy match score to consider evidence "verified"
VERIFICATION_THRESHOLD = 0.55


class EvidenceVerifier:
    """Verifies LLM-generated evidence against actual PR data."""

    def verify_decisions(
        self, decisions: list[ExtractedDecision], ctx: PRContext
    ) -> list[ExtractedDecision]:
        """Verify all evidence across all decisions against the real PR data.

        Mutates the evidence items in-place, setting:
        - verified: bool
        - matched_source_key: which source the quote matched
        - matched_source_url: GitHub URL for the matched source
        - match_score: how closely the quote matched (0.0-1.0)

        Returns the decisions with verified/flagged evidence.
        """
        # Build the source text map from the PR context
        source_texts = ctx.all_text_sources
        source_urls = self._build_source_urls(ctx)

        for decision in decisions:
            for evidence in decision.evidence:
                self._verify_single(evidence, source_texts, source_urls)

        return decisions

    def _verify_single(
        self,
        evidence: ExtractedEvidence,
        source_texts: dict[str, str],
        source_urls: dict[str, str],
    ) -> None:
        """Verify a single evidence item against all available sources."""
        quote = evidence.quote.strip()
        if not quote or len(quote) < 10:
            evidence.verified = False
            evidence.match_score = 0.0
            return

        best_score = 0.0
        best_key = None

        for source_key, source_text in source_texts.items():
            score = self._match_score(quote, source_text)
            if score > best_score:
                best_score = score
                best_key = source_key

        evidence.match_score = round(best_score, 3)

        if best_score >= VERIFICATION_THRESHOLD and best_key:
            evidence.verified = True
            evidence.matched_source_key = best_key
            evidence.matched_source_url = source_urls.get(best_key, "")
            logger.debug(
                f"Evidence verified (score={best_score:.2f}): "
                f"'{quote[:60]}...' matched '{best_key}'"
            )
        else:
            evidence.verified = False
            logger.debug(
                f"Evidence NOT verified (best_score={best_score:.2f}): "
                f"'{quote[:60]}...'"
            )

    def _match_score(self, quote: str, source_text: str) -> float:
        """Calculate how well a quote matches a source text.

        Uses a multi-strategy approach:
        1. Exact substring → 1.0
        2. Normalized substring → 0.95
        3. Best fuzzy subsequence match → 0.0-0.9
        """
        if not quote or not source_text:
            return 0.0

        # Strategy 1: Exact substring match
        if quote in source_text:
            return 1.0

        # Strategy 2: Normalized match (case-insensitive, whitespace-collapsed)
        norm_quote = self._normalize(quote)
        norm_source = self._normalize(source_text)

        if norm_quote in norm_source:
            return 0.95

        # Strategy 3: Fuzzy subsequence matching
        # For long source texts, find the best-matching window
        if len(norm_source) > len(norm_quote) * 3:
            # Sliding window approach for efficiency
            return self._best_window_match(norm_quote, norm_source)
        else:
            return SequenceMatcher(None, norm_quote, norm_source).ratio()

    def _best_window_match(self, quote: str, source: str) -> float:
        """Find the best-matching window in a long source text.

        Slides a window roughly the size of the quote across the source
        and returns the best SequenceMatcher ratio found.
        """
        quote_len = len(quote)
        # Use windows of varying sizes around the quote length
        window_sizes = [
            quote_len,
            int(quote_len * 1.3),
            int(quote_len * 1.5),
            int(quote_len * 0.7),
        ]

        best_score = 0.0
        step = max(1, quote_len // 4)  # Step size for sliding window

        for window_size in window_sizes:
            for start in range(0, len(source) - window_size + 1, step):
                window = source[start : start + window_size]
                score = SequenceMatcher(None, quote, window).ratio()
                if score > best_score:
                    best_score = score
                    if best_score > 0.85:
                        return best_score  # Early exit on strong match

        return best_score

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison: lowercase, collapse whitespace, strip punctuation edges."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _build_source_urls(self, ctx: PRContext) -> dict[str, str]:
        """Build a mapping from source keys to GitHub URLs."""
        urls: dict[str, str] = {}
        urls["pr_description"] = ctx.html_url

        for c in ctx.review_comments:
            urls[f"review_comment_{c.id}"] = c.html_url

        for c in ctx.issue_comments:
            urls[f"issue_comment_{c.id}"] = c.html_url

        for commit in ctx.commits:
            urls[f"commit_{commit.sha[:8]}"] = commit.html_url

        return urls


def calculate_independent_confidence(
    decision: ExtractedDecision,
    ctx: PRContext,
    verified_count: int,
    total_evidence: int,
) -> float:
    """Calculate a confidence score independently of the LLM's self-report.

    Factors:
    1. Evidence verification ratio (most important)
    2. Number of supporting evidence items
    3. Category clarity (some categories are easier to detect)
    4. PR scale (very small PRs rarely contain decisions)
    5. Discussion depth (comments suggest real deliberation)
    """
    score = 0.0

    # Factor 1: Evidence verification ratio (0 - 0.4)
    if total_evidence > 0:
        verification_ratio = verified_count / total_evidence
        score += verification_ratio * 0.4
    else:
        score += 0.1  # Some baseline even without evidence

    # Factor 2: Number of evidence items (0 - 0.15)
    if verified_count >= 3:
        score += 0.15
    elif verified_count >= 2:
        score += 0.10
    elif verified_count >= 1:
        score += 0.05

    # Factor 3: Category clarity (0 - 0.15)
    clear_categories = {
        "technology_selection", "pattern_adoption", "deprecation", "architecture"
    }
    if decision.category in clear_categories:
        score += 0.15
    elif decision.category in {"refactoring", "tradeoff"}:
        score += 0.10
    else:
        score += 0.05

    # Factor 4: PR scale (0 - 0.15)
    total_changes = ctx.total_additions + ctx.total_deletions
    if total_changes >= 50 and ctx.total_files_changed >= 3:
        score += 0.15
    elif total_changes >= 20:
        score += 0.10
    else:
        score += 0.05

    # Factor 5: Discussion depth (0 - 0.15)
    total_comments = len(ctx.review_comments) + len(ctx.issue_comments)
    if total_comments >= 5:
        score += 0.15
    elif total_comments >= 2:
        score += 0.10
    elif total_comments >= 1:
        score += 0.05

    return round(min(score, 1.0), 3)

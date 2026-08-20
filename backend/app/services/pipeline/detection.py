"""
Decision Candidate Detection for Decisio Pipeline.

Combines deterministic change analysis signals with keyword heuristic weights to
determine whether a Pull Request likely contains a meaningful technical decision.

Key requirements:
1. The system must be allowed to say NO.
2. Routine updates (dependency bumps, formatting, docs) are weighted negatively
   unless strong positive signals exist (e.g. ADR changes, intentional migrations).
3. The output fits the user's requested schema:
   - decision_detected (bool)
   - confidence (0 - 100)
   - candidate_reason (str)
   - signals (list of strings)
"""

from __future__ import annotations

import logging
import re

from app.services.pipeline.models import DetectionResult, PRContext
from app.services.pipeline.change_analysis import ChangeAnalysisResult

logger = logging.getLogger(__name__)

# Keywords that suggest an architectural decision
DECISION_KEYWORDS = {
    "migrate": 25,
    "migration": 25,
    "replace": 20,
    "adopt": 25,
    "deprecate": 25,
    "deprecated": 20,
    "introduce": 15,
    "architecture": 25,
    "redesign": 25,
    "refactor": 15,
    "switch to": 25,
    "switch from": 25,
    "move to": 20,
    "move from": 20,
    "chosen": 15,
    "decided": 20,
    "decision": 20,
    "trade-off": 25,
    "tradeoff": 25,
    "trade off": 20,
    "cqrs": 30,
    "event-driven": 25,
    "microservice": 25,
    "monolith": 20,
    "breaking change": 20,
    "rfc": 25,
    "adr": 30,
    "proposal": 15,
}


class HeuristicDetector:
    """Combines deterministic change analysis and heuristics to detect technical decisions."""

    def detect(self, ctx: PRContext, change_analysis: ChangeAnalysisResult) -> DetectionResult:
        """Analyze PR context & change analysis to produce a decision detection assessment."""
        signals: list[str] = []
        positive_score = 0
        negative_score = 0

        # 1. Evaluate Deterministic Change Types
        if change_analysis.is_dependency_bump:
            # Dependency bumps default to negative unless explicit decision keywords exist
            has_decision_keywords = self._has_architectural_keywords(ctx)
            if has_decision_keywords:
                positive_score += 15
                signals.append("dependency_bump_with_reasoning")
            else:
                negative_score += 40
                signals.append("routine_dependency_bump")

        if change_analysis.is_docs_only:
            # Check if updating architecture docs (ADRs)
            has_adr_focus = any(term in ctx.title.lower() or term in ctx.body.lower() for term in ["adr", "rfc", "architecture doc"])
            if has_adr_focus:
                positive_score += 30
                signals.append("architectural_docs_update")
            else:
                negative_score += 50
                signals.append("documentation_only")

        if change_analysis.is_tests_only:
            negative_score += 40
            signals.append("test_suite_only")

        if change_analysis.is_config_only:
            has_decision_keywords = self._has_architectural_keywords(ctx)
            if has_decision_keywords:
                positive_score += 15
                signals.append("config_with_reasoning")
            else:
                negative_score += 25
                signals.append("configuration_only")

        if change_analysis.is_refactor:
            positive_score += 15
            signals.append("refactoring_indicated")

        # 2. Evaluate Text Keywords (Title & Body)
        title_lower = ctx.title.lower()
        body_lower = ctx.body.lower()

        # Check title keywords (strong signal)
        for kw, weight in DECISION_KEYWORDS.items():
            if kw in title_lower:
                positive_score += weight
                signals.append(f"title_keyword_{kw.replace(' ', '_')}")
                break  # count title once

        # Check body keywords (medium signal)
        body_matches = 0
        for kw, weight in DECISION_KEYWORDS.items():
            if kw in body_lower:
                positive_score += int(weight * 0.7)
                signals.append(f"body_keyword_{kw.replace(' ', '_')}")
                body_matches += 1
                if body_matches >= 3:
                    break

        # 3. Evaluate Comment Keywords (Developer Consensus)
        all_comments_text = " ".join(c.body.lower() for c in ctx.all_comments)
        consensus_terms = ["agree", "disagree", "alternative", "instead", "propose", "should we", "trade-off", "tradeoff", "consensus"]
        matched_consensus = [term for term in consensus_terms if term in all_comments_text]
        if len(matched_consensus) >= 2:
            positive_score += 20
            signals.append("discussion_consensus_signals")

        # 4. Evaluate Commit Messages
        all_commits_text = " ".join(commit.message.lower() for commit in ctx.commits)
        commit_matches = 0
        for kw, weight in DECISION_KEYWORDS.items():
            if kw in all_commits_text:
                positive_score += 10
                signals.append(f"commit_keyword_{kw.replace(' ', '_')}")
                commit_matches += 1
                if commit_matches >= 2:
                    break

        # 5. Evaluate Scale
        total_lines = ctx.total_additions + ctx.total_deletions
        if total_lines <= 10 and len(ctx.files) <= 2:
            negative_score += 20
            signals.append("trivial_change_size")
        elif total_lines >= 150 and len(ctx.files) >= 5:
            positive_score += 10
            signals.append("substantial_change_size")

        # ── Calculate Confidence (0 - 100) ──
        total_score = positive_score + negative_score
        if total_score == 0:
            confidence = 30
        else:
            confidence = int((positive_score / total_score) * 100)

        # Decide whether to proceed
        decision_detected = confidence >= 40

        # Construct candidate reason
        if decision_detected:
            reasons = [s for s in signals if s.startswith(("title_keyword", "body_keyword", "dependency_bump_with_reasoning", "architectural_docs_update"))]
            primary_reason = reasons[0] if reasons else "presence of positive architectural signals"
            candidate_reason = f"Potential architectural decision detected via {primary_reason.replace('_', ' ')}."
        else:
            negatives = [s for s in signals if s in ["documentation_only", "routine_dependency_bump", "test_suite_only", "trivial_change_size"]]
            primary_neg = negatives[0] if negatives else "lack of significant architectural changes"
            candidate_reason = f"No significant technical decision detected due to {primary_neg.replace('_', ' ')}."

        return DetectionResult(
            decision_detected=decision_detected,
            confidence=confidence,
            candidate_reason=candidate_reason,
            signals=signals,
        )

    def _has_architectural_keywords(self, ctx: PRContext) -> bool:
        """Helper to quickly check if any core keywords are in the PR description or commits."""
        text = (ctx.title + " " + ctx.body + " " + " ".join(c.message for c in ctx.commits)).lower()
        # Clean out common negated or templated dependabot text
        text = text.replace("no breaking changes", "").replace("non-breaking", "").replace("no breaking change", "")
        check_words = [
            "decided to", "we decided", "migration", "replaced", "breaking change",
            "adr", "rfc", "architectural decision", "technology selection", "adopted"
        ]
        return any(word in text for word in check_words)

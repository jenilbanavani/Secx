"""
Pipeline Orchestrator for Decisio.

Runs all pipeline stages in sequence:
  Ingest → Change Analysis → Detect → Extract → Verify → Analyze → Persist → Result

Collects a pipeline log with timing for each stage so the user can see
exactly what Decisio did and how long each step took.

Design principles:
- Each stage is independently testable.
- A failure in one stage doesn't crash the whole pipeline.
- The pipeline log gives full transparency into the analysis process.
- "No decision" is a valid, confident result — not a failure.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.services.llm_client import LLMClient
from app.services.pipeline.analysis import ImpactAnalyzer
from app.services.pipeline.change_analysis import ChangeAnalyzer
from app.services.pipeline.detection import HeuristicDetector
from app.services.pipeline.extraction import DecisionExtractionService
from app.services.pipeline.ingestion import PRIngestion
from app.services.pipeline.models import (
    PipelineResult,
    PipelineStage,
    PipelineStageStatus,
    VerifiedDecision,
)
from app.services.pipeline.persistence import PipelinePersistence
from app.services.pipeline.verification import (
    EvidenceVerifier,
    calculate_independent_confidence,
)

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Orchestrates the complete decision extraction pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(self.settings)

        # Pipeline stages
        self.ingestion = PRIngestion(self.settings)
        self.change_analyzer = ChangeAnalyzer()
        self.detector = HeuristicDetector()
        self.extractor = DecisionExtractionService(self.settings, self.llm_client)
        self.verifier = EvidenceVerifier()
        self.analyzer = ImpactAnalyzer()
        self.persistence = PipelinePersistence()

    async def analyze(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        provider: str | None = None,
        model: str | None = None,
        db: AsyncSession | None = None,
    ) -> PipelineResult:
        """Run the complete analysis pipeline on a GitHub PR."""
        pipeline_start = time.perf_counter()
        stages: list[PipelineStage] = []
        active_provider = provider or self.settings.llm_provider

        result = PipelineResult(
            repository=f"{owner}/{repo}",
            pr_number=pull_number,
            provider_used=active_provider,
        )

        # ── Stage 1: Ingest ──────────────────────────────────
        stage = PipelineStage(name="ingest", status=PipelineStageStatus.RUNNING)
        stage_start = time.perf_counter()

        try:
            ctx = await self.ingestion.ingest(owner, repo, pull_number)
            stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
            stage.status = PipelineStageStatus.COMPLETED
            stage.message = (
                f"Fetched PR #{ctx.number}: '{ctx.title}' "
                f"({ctx.total_files_changed} files, "
                f"{len(ctx.review_comments)} review comments, "
                f"{len(ctx.issue_comments)} discussion comments, "
                f"{len(ctx.commits)} commits)"
            )
            stage.details = {
                "files_count": ctx.total_files_changed,
                "additions": ctx.total_additions,
                "deletions": ctx.total_deletions,
                "review_comments": len(ctx.review_comments),
                "issue_comments": len(ctx.issue_comments),
                "commits": len(ctx.commits),
            }
            result.pr_title = ctx.title
            result.pr_url = ctx.html_url
            result.pr_author = ctx.author
        except Exception as e:
            stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
            stage.status = PipelineStageStatus.FAILED
            stage.message = f"Failed to fetch PR: {str(e)}"
            stages.append(stage)
            result.pipeline_log = stages
            result.total_duration_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)
            return result

        stages.append(stage)

        # ── Stage 2: Change Analysis ──────────────────────────
        stage = PipelineStage(name="change_analysis", status=PipelineStageStatus.RUNNING)
        stage_start = time.perf_counter()

        try:
            change_analysis = self.change_analyzer.analyze(ctx)
            stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
            stage.status = PipelineStageStatus.COMPLETED
            stage.message = change_analysis.summary
            stage.details = change_analysis.model_dump()
            result.change_analysis = change_analysis
        except Exception as e:
            stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
            stage.status = PipelineStageStatus.FAILED
            stage.message = f"Change analysis failed: {str(e)}"
            stages.append(stage)
            result.pipeline_log = stages
            result.total_duration_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)
            return result

        stages.append(stage)

        # ── Stage 3: Decision Detection ───────────────────────
        stage = PipelineStage(name="detect", status=PipelineStageStatus.RUNNING)
        stage_start = time.perf_counter()

        try:
            detection = self.detector.detect(ctx, change_analysis)
            stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
            stage.status = PipelineStageStatus.COMPLETED

            signal_summary = ", ".join(detection.signals[:5])
            stage.message = (
                f"{'Candidate detected' if detection.decision_detected else 'No candidate'} "
                f"(confidence: {detection.confidence}%) — "
                f"signals: {signal_summary or 'none'}"
            )
            stage.details = detection.model_dump()
            result.detection = detection
        except Exception as e:
            stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
            stage.status = PipelineStageStatus.FAILED
            stage.message = f"Detection failed: {str(e)}"
            stages.append(stage)
            result.pipeline_log = stages
            result.total_duration_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)
            return result

        stages.append(stage)

        # ── Stage 4: Extract (only if candidate detected) ────
        extracted_decisions = []

        if detection.decision_detected:
            stage = PipelineStage(name="extract", status=PipelineStageStatus.RUNNING)
            stage_start = time.perf_counter()

            try:
                extracted_decisions = await self.extractor.extract(
                    ctx, provider=provider, model=model
                )
                stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
                stage.status = PipelineStageStatus.COMPLETED
                stage.message = (
                    f"Extracted {len(extracted_decisions)} decision(s) "
                    f"using {active_provider}"
                )
                stage.details = {
                    "decisions_extracted": len(extracted_decisions),
                    "provider": active_provider,
                }
            except Exception as e:
                stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
                stage.status = PipelineStageStatus.FAILED
                stage.message = f"Extraction failed: {str(e)}"
                logger.error(f"LLM extraction failed: {e}")

            stages.append(stage)
        else:
            stage = PipelineStage(
                name="extract",
                status=PipelineStageStatus.SKIPPED,
                message=f"Skipped — {detection.candidate_reason or 'no candidate detected'}",
            )
            stages.append(stage)

        # ── Stage 5: Verify Evidence ─────────────────────────
        if extracted_decisions:
            stage = PipelineStage(name="verify", status=PipelineStageStatus.RUNNING)
            stage_start = time.perf_counter()

            try:
                extracted_decisions = self.verifier.verify_decisions(
                    extracted_decisions, ctx
                )
                total_evidence = sum(len(d.evidence) for d in extracted_decisions)
                verified_count = sum(
                    sum(1 for e in d.evidence if e.verified)
                    for d in extracted_decisions
                )
                stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
                stage.status = PipelineStageStatus.COMPLETED
                stage.message = (
                    f"Verified {verified_count}/{total_evidence} evidence items "
                    f"against actual PR data"
                )
                stage.details = {
                    "total_evidence": total_evidence,
                    "verified_count": verified_count,
                }
            except Exception as e:
                stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
                stage.status = PipelineStageStatus.FAILED
                stage.message = f"Verification failed: {str(e)}"

            stages.append(stage)
        else:
            stage = PipelineStage(
                name="verify",
                status=PipelineStageStatus.SKIPPED,
                message="Skipped — no decisions to verify",
            )
            stages.append(stage)

        # ── Stage 6: Analyze Impact & Technologies ───────────
        stage = PipelineStage(name="analyze", status=PipelineStageStatus.RUNNING)
        stage_start = time.perf_counter()

        try:
            # Combine relevant files list from all decisions to analyze general impact
            all_rel_files = []
            for d in extracted_decisions:
                all_rel_files.extend(d.relevant_files)

            affected_files = self.analyzer.analyze_affected_files(ctx, all_rel_files)
            technologies = self.analyzer.detect_technologies(ctx, extracted_decisions)
            stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
            stage.status = PipelineStageStatus.COMPLETED
            stage.message = (
                f"Analyzed impact: {len(affected_files)} files, "
                f"{len(technologies)} technologies detected"
            )
            stage.details = {
                "affected_files": len(affected_files),
                "technologies": [t.name for t in technologies],
            }
        except Exception as e:
            stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
            stage.status = PipelineStageStatus.FAILED
            stage.message = f"Analysis failed: {str(e)}"
            affected_files = []
            technologies = []

        stages.append(stage)

        # Assemble Verified Decisions list
        verified_decisions: list[VerifiedDecision] = []
        for d in extracted_decisions:
            verified_count = sum(1 for e in d.evidence if e.verified)
            total_evidence = len(d.evidence)

            confidence = calculate_independent_confidence(
                d, ctx, verified_count, total_evidence
            )

            # Filter affected files to only match this decision's relevant files
            decision_files = self.analyzer.analyze_affected_files(ctx, d.relevant_files)

            verified_decisions.append(
                VerifiedDecision(
                    title=d.title,
                    summary=d.summary,
                    rationale=d.rationale,
                    alternatives=d.alternatives,
                    chosen_approach=d.chosen_approach,
                    category=d.category,
                    evidence=d.evidence,
                    affected_files=decision_files,
                    technologies=technologies,
                    confidence=confidence,
                    llm_confidence=d.llm_confidence,
                    verified_evidence_count=verified_count,
                    total_evidence_count=total_evidence,
                )
            )

        result.has_significant_decision = len(verified_decisions) > 0
        result.decisions = verified_decisions

        # ── Stage 7: Persistence ──────────────────────────────
        if db and result.has_significant_decision:
            stage = PipelineStage(name="persist", status=PipelineStageStatus.RUNNING)
            stage_start = time.perf_counter()

            try:
                persisted_ids = await self.persistence.persist(db, result)
                stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
                stage.status = PipelineStageStatus.COMPLETED
                stage.message = f"Persisted {len(persisted_ids)} decisions to DB."
                stage.details = {"persisted_ids": persisted_ids}
            except Exception as e:
                stage.duration_ms = round((time.perf_counter() - stage_start) * 1000, 1)
                stage.status = PipelineStageStatus.FAILED
                stage.message = f"Persistence failed: {str(e)}"
            
            stages.append(stage)
        else:
            stage = PipelineStage(
                name="persist",
                status=PipelineStageStatus.SKIPPED,
                message="Skipped — database session not provided or no decisions found",
            )
            stages.append(stage)

        result.pipeline_log = stages
        result.total_duration_ms = round(
            (time.perf_counter() - pipeline_start) * 1000, 1
        )

        return result

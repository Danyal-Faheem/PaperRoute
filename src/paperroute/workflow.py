from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from datetime import UTC
from pathlib import Path

from .arxiv_client import ArxivClient
from .db import RunStore
from .models import Paper, PaperAssessment, Run, RunRequest, RunStatus, TrajectoryEvent
from .openai_client import ModelClient
from .scoring import rank_assessments, score_assessment
from .verification import extract_pages, verify_evidence


class Workflow:
    def __init__(self, store: RunStore, arxiv: ArxivClient, model: ModelClient,
                 max_candidates: int = 20, shortlist_size: int = 6, concurrency: int = 3,
                 retries: int = 2) -> None:
        self.store, self.arxiv, self.model = store, arxiv, model
        self.max_candidates, self.shortlist_size, self.concurrency, self.retries = max_candidates, shortlist_size, concurrency, retries
        self._retry_count = 0

    async def execute(self, run: Run) -> Run:
        try:
            self._retry_count = 0
            reset_usage = getattr(self.model, "reset_usage", None)
            if reset_usage:
                reset_usage()
            await self._status(run, RunStatus.planning, "planner", "start")
            query = await self._retry(lambda: self.model.plan_query(run.request))
            self._event(run, "planning", "query_planner", "complete", output_summary=query)
            await self._status(run, RunStatus.searching, "search", "start")
            papers = await self._collect_papers(run.request, query)
            run.candidates = papers
            self._event(run, "searching", "arxiv_search", "complete", output_summary=f"{len(papers)} candidates", tool="arxiv")
            await self._status(run, RunStatus.screening, "abstract_screener", "start")
            ids = await self._retry(lambda: self.model.screen(run.request, papers))
            selected = [p for p in papers if p.arxiv_id in set(ids)][:self.shortlist_size]
            if not selected:
                selected = papers[:self.shortlist_size]
            self._event(run, "screening", "abstract_screener", "complete", output_summary=f"{len(selected)} shortlisted")
            await self._status(run, RunStatus.analyzing, "paper_analyst", "start")
            semaphore = asyncio.Semaphore(self.concurrency)

            async def one(paper: Paper) -> PaperAssessment:
                async with semaphore:
                    try:
                        path = await self._retry(lambda: self.arxiv.download_pdf(paper))
                        paper.pdf_path = str(path)
                        assessment = await self._retry(lambda: self._assess(run.request, paper, Path(path)))
                        assessment.paper = paper
                        assessment.evidence = verify_evidence(assessment.evidence, extract_pages(path))
                        # Unsupported quotations get one fresh analyst attempt; the scorer
                        # subsequently caps an unverified recommendation at Skim.
                        candidate = self._score(assessment, run.request)
                        verified_count = sum(e.verified for e in assessment.evidence)
                        if (assessment.evidence or candidate.total_score >= 75) and verified_count < 2:
                            rejected_items = [item for item in assessment.evidence if not item.verified]
                            rejected = len(rejected_items)
                            feedback_lines = [
                                f"- page {item.page}: {item.quotation!r} — {item.verification_note or 'Quotation failed exact page verification.'}"
                                for item in rejected_items
                            ]
                            if not feedback_lines:
                                feedback_lines.append(
                                    "- No usable quotation was returned; provide two contiguous 8-30-word spans from marked pages."
                                )
                            feedback = (
                                "Exact page verification rejected the following evidence. Do not reuse, paraphrase, "
                                "stitch, or repair these quotes; select fresh contiguous source spans.\n" +
                                "\n".join(feedback_lines)
                            )
                            self._retry_count += 1
                            self._event(run, "analyzing", "paper_analyst", "evidence_retry",
                                        input_summary=f"{paper.arxiv_id}; verifier feedback: {feedback}",
                                        output_summary=(
                                            f"rejected {rejected} unsupported quotation(s); fresh contiguous quotations requested"
                                        ),
                                        tool_response={"rejected_count": rejected, "feedback": feedback},
                                        retry=1)
                            retry_assessment = await self._retry(
                                lambda: self._assess(run.request, paper, Path(path), feedback=feedback,
                                                     temperature=0.1, seed=43)
                            )
                            retry_assessment.paper = paper
                            retry_assessment.evidence = verify_evidence(retry_assessment.evidence, extract_pages(path))
                            if sum(e.verified for e in retry_assessment.evidence) > sum(e.verified for e in assessment.evidence):
                                assessment = retry_assessment
                        return self._score(assessment, run.request)
                    except Exception as exc:
                        return PaperAssessment(paper=paper, abstract_only=True, error=str(exc), summary="Analysis failed; abstract only.")

            run.assessments = list(await asyncio.gather(*(one(p) for p in selected)))
            self._event(run, "analyzing", "paper_analyst", "complete", output_summary=f"{len(run.assessments)} assessments")
            await self._status(run, RunStatus.ranking, "ranker", "start")
            run.assessments = rank_assessments(await self._retry(lambda: self.model.rank(run.request, run.assessments)))
            run.assessments = [self._score(x, run.request) for x in run.assessments]
            model_usage = getattr(self.model, "usage", None)
            if model_usage is not None:
                run.usage = model_usage
            run.retry_count = self._retry_count
            self._event(run, "ranking", "ranker", "complete", output_summary="deterministic rubric ranking")
            await self._status(run, RunStatus.completed, "system", "complete")
        except Exception as exc:
            run.error = str(exc)
            run.retry_count = self._retry_count
            await self._status(run, RunStatus.failed, "system", "failed", output_summary=str(exc))
        return run

    async def _collect_papers(self, request: RunRequest, query: str) -> list[Paper]:
        found: dict[str, Paper] = {}
        for arxiv_id in request.pinned_arxiv_ids[:self.max_candidates]:
            try:
                paper = await self._retry(lambda arxiv_id=arxiv_id: self.arxiv.get_paper(arxiv_id))
                if paper:
                    found[paper.arxiv_id] = paper
            except Exception:
                continue
        remaining = max(0, self.max_candidates - len(found))
        if remaining:
            for paper in await self._retry(lambda: self.arxiv.search(query, remaining, request.categories)):
                found.setdefault(paper.arxiv_id, paper)
        papers = list(found.values())
        if request.date_from or request.date_to:
            def in_range(paper: Paper) -> bool:
                if paper.published is None:
                    return True
                published = paper.published if paper.published.tzinfo else paper.published.replace(tzinfo=UTC)
                start = request.date_from
                end = request.date_to
                if start and not start.tzinfo:
                    start = start.replace(tzinfo=UTC)
                if end and not end.tzinfo:
                    end = end.replace(tzinfo=UTC)
                return (not start or published >= start) and (not end or published <= end)
            papers = [p for p in papers if in_range(p)]
        return papers[:self.max_candidates]

    async def _retry(self, operation: Callable[[], Awaitable]):
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return await operation()
            except Exception as exc:
                last = exc
                if attempt < self.retries:
                    self._retry_count += 1
                    await asyncio.sleep(0.15 * (2 ** attempt))
        raise last or RuntimeError("operation failed")

    async def _assess(self, request: RunRequest, paper: Paper, path: Path, *,
                      feedback: str | None = None, temperature: float | None = None,
                      seed: int | None = None) -> PaperAssessment:
        """Call assess with optional retry controls when the adapter supports them.

        Signature inspection keeps older injected test doubles working without
        catching arbitrary ``TypeError`` exceptions raised inside their method.
        """
        method = self.model.assess
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        parameter_list = list(parameters)
        accepts_any_keyword = any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                                  for parameter in parameter_list)
        kwargs: dict[str, object] = {}
        for name, value in (("feedback", feedback), ("temperature", temperature), ("seed", seed)):
            if value is not None and (accepts_any_keyword or any(parameter.name == name
                                                                  for parameter in parameter_list)):
                kwargs[name] = value
        return await method(request, paper, path, **kwargs)

    def _score(self, assessment: PaperAssessment, request: RunRequest) -> PaperAssessment:
        hay = f"{assessment.paper.title} {assessment.paper.abstract}".casefold()
        excluded = any(term.casefold() in hay for term in request.exclusion_criteria)
        return score_assessment(assessment, excluded)

    async def _status(self, run: Run, status: RunStatus, role: str, event: str, output_summary: str = "") -> None:
        run.status = status
        self._event(run, status.value, role, event, output_summary=output_summary)
        self.store.update(run)

    def _event(self, run: Run, stage: str, role: str, event: str, **kwargs) -> None:
        run.trajectories.append(TrajectoryEvent(stage=stage, role=role, event=event, **kwargs))
        self.store.update(run)

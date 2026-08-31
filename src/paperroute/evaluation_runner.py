"""Offline-safe live benchmark orchestration.

This module deliberately keeps evaluation concerns separate from the product
workflow.  The normal CLI validates the manifest; ``--live`` is the only path
that can contact arXiv or OpenAI, and ``--demo-live`` uses deterministic local
adapters for a complete end-to-end smoke run.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .arxiv_client import ArxivClient
from .benchmark import ndcg_at_k
from .db import RunStore
from .demo import DemoModelClient
from .models import Decision, Paper, PaperAssessment, Run, RunRequest
from .openai_client import EVIDENCE_INSTRUCTIONS, BatchAssessmentOutput, OpenAIResponsesClient
from .scoring import rank_assessments, score_assessment
from .verification import extract_pages, verify_evidence
from .workflow import Workflow


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot satisfy its reproducibility contract."""


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _usage_snapshot(model: Any) -> dict[str, int]:
    usage = getattr(model, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {key: int(getattr(usage, key, 0) or 0) for key in ("input_tokens", "output_tokens", "total_tokens")}


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: max(0, after[key] - before[key]) for key in before}


async def _with_retries(operation: Callable[[], Awaitable[Any]], retries: int) -> tuple[Any, int]:
    """Run an operation, returning its value and number of retries consumed."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await operation(), attempt
        except Exception as exc:  # errors are reported by the caller with context
            last = exc
            if attempt < retries:
                await asyncio.sleep(0.05 * (2**attempt))
    raise last or EvaluationError("operation failed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HydratedArxiv:
    """Arxiv adapter that reuses metadata/PDFs fetched during hydration."""

    def __init__(self, papers: list[Paper]):
        self.papers = {paper.arxiv_id: paper for paper in papers}

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        return self.papers.get(arxiv_id)

    async def search(self, query: str, max_results: int = 20, categories: list[str] | None = None) -> list[Paper]:
        return []

    async def download_pdf(self, paper: Paper) -> Path:
        if not paper.pdf_path:
            raise FileNotFoundError(f"no cached PDF for {paper.arxiv_id}")
        return Path(paper.pdf_path)


class DemoArxiv:
    """Deterministic six-paper fixture adapter used by ``--demo-live``."""

    def __init__(self, papers: list[Paper], cache_dir: Path):
        self.papers = {paper.arxiv_id: paper for paper in papers}
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        return self.papers.get(arxiv_id)

    async def search(self, query: str, max_results: int = 20, categories: list[str] | None = None) -> list[Paper]:
        return []

    async def download_pdf(self, paper: Paper) -> Path:
        path = self.cache_dir / (hashlib.sha256(paper.arxiv_id.encode()).hexdigest()[:20] + ".pdf")
        if not path.exists():
            from pypdf import PdfWriter

            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            with path.open("wb") as stream:
                writer.write(stream)
        return path


def _labels_from_payload(payload: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, dict[str, float]] | None:
    candidates = payload.get("labels")
    if isinstance(candidates, dict):
        result = candidates
    elif all(isinstance(case.get("labels"), dict) for case in cases):
        result = {case["id"]: case["labels"] for case in cases}
    elif all(isinstance(case.get("relevance"), dict) for case in cases):
        result = {case["id"]: case["relevance"] for case in cases}
    else:
        return None
    checked: dict[str, dict[str, float]] = {}
    for case in cases:
        values = result.get(case["id"])
        if not isinstance(values, dict) or set(values) != set(case["arxiv_ids"]):
            raise EvaluationError(f"labels for {case['id']} must cover all six pinned papers")
        checked[case["id"]] = {}
        for paper_id, score in values.items():
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 3:
                raise EvaluationError(f"label for {case['id']}/{paper_id} must be between 0 and 3")
            checked[case["id"]][paper_id] = float(score)
    return checked


async def _call_judge(model: Any, request: RunRequest, paper: Paper, *, temperature: float, seed: int) -> int:
    """Call old injected judges as well as the seed-aware model adapter."""
    method = model.judge_relevance
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    supports_keywords = any(parameter.kind == inspect.Parameter.VAR_KEYWORD or
                             parameter.name in {"temperature", "seed"} for parameter in parameters)
    if supports_keywords:
        return await method(request, paper, temperature=temperature, seed=seed)
    return await method(request, paper)


async def _judge_all(cases: list[dict[str, Any]], model: Any, demo: bool,
                     paper_map: dict[str, dict[str, Paper]] | None = None,
                     diagnostics: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    local_controls = bool(getattr(model, "custom_compatible", False))
    if diagnostics is not None:
        diagnostics.update({"available": not demo, "paired_judgments": 0,
                             "agreements": 0, "disagreements": 0, "tie_breaks": 0,
                             "agreement_rate": None, "passes": 2,
                             "seeds": [42, 43, 44] if local_controls else [],
                             "temperature": 0.1 if local_controls else 0.0,
                             "seed_controls_applied": local_controls,
                             "temperature_controls_applied": local_controls,
                             "model": getattr(model, "judge_model", "demo"),
                             "source": "demo-fixed-proxy" if demo else "generated"})
    labels: dict[str, dict[str, float]] = {}
    for case in cases:
        request = RunRequest(research_question=case["question"], inclusion_criteria=case.get("include", []),
                             exclusion_criteria=case.get("exclude", []), categories=case.get("categories", []),
                             pinned_arxiv_ids=case["arxiv_ids"])
        case_labels: dict[str, float] = {}
        for index, paper_id in enumerate(case["arxiv_ids"]):
            paper = (paper_map or {}).get(case["id"], {}).get(paper_id) or Paper(
                arxiv_id=paper_id, title=case["question"], abstract=case["question"])
            if demo:
                # Demo labels are fixed graded proxies, never presented as expert ground truth.
                case_labels[paper_id] = float(max(0, 3 - index // 2))
                continue
            judge_temperature = 0.1 if local_controls else 0.0
            first = await _call_judge(model, request, paper, temperature=judge_temperature, seed=42)
            second = await _call_judge(model, request, paper, temperature=judge_temperature, seed=43)
            if diagnostics is not None:
                diagnostics["paired_judgments"] += 1
            if first == second:
                if diagnostics is not None:
                    diagnostics["agreements"] += 1
                case_labels[paper_id] = float(first)
            else:
                if diagnostics is not None:
                    diagnostics["disagreements"] += 1
                    diagnostics["tie_breaks"] += 1
                case_labels[paper_id] = float(await _call_judge(model, request, paper, temperature=judge_temperature, seed=44))
        labels[case["id"]] = case_labels
    if diagnostics is not None and diagnostics["paired_judgments"]:
        diagnostics["agreement_rate"] = round(diagnostics["agreements"] / diagnostics["paired_judgments"], 4)
    return labels


async def _direct_assess(model: Any, request: RunRequest, paper: Paper, pdf_path: Path | None) -> PaperAssessment:
    if isinstance(model, DemoModelClient):
        return await model.assess(request, paper, pdf_path)
    if not isinstance(model, OpenAIResponsesClient) or not model._client:
        raise EvaluationError("live baseline requires an OpenAI client")
    if model.custom_compatible:
        # The compatible server has no /v1/files endpoint. OpenAIResponsesClient
        # extracts pages locally and sends bounded, marked text to chat.
        assessment = await model.assess(request, paper, pdf_path)
        assessment.evidence = verify_evidence(assessment.evidence, extract_pages(pdf_path)) if pdf_path else assessment.evidence
        return assessment
    content: list[dict[str, Any]] = [{"type": "input_text", "text": json.dumps({
        "request": request.model_dump(), "paper": paper.model_dump(),
        "task": "Directly rank this paper for the research question using the four rubric scores."
    }, default=str)}]
    uploaded_id: str | None = None
    file_handle = None
    try:
        if pdf_path and pdf_path.exists():
            file_handle = pdf_path.open("rb")
            uploaded = await model._client.files.create(file=file_handle, purpose="user_data")
            uploaded_id = getattr(uploaded, "id", None)
            if uploaded_id:
                content.append({"type": "input_file", "file_id": uploaded_id})
        data = await model._json("Return one JSON object with four integer scores from 0 to 100, a summary, and limitations.", content)
        data["paper"] = paper.model_dump()
        assessment = PaperAssessment.model_validate(data)
        assessment.evidence = verify_evidence(assessment.evidence, extract_pages(pdf_path)) if pdf_path else assessment.evidence
        return score_assessment(assessment)
    finally:
        if file_handle:
            file_handle.close()
        if uploaded_id:
            try:
                await model._client.files.delete(uploaded_id)
            except Exception:
                pass


async def _direct_batch_assess(model: Any, request: RunRequest, papers: list[Paper]) -> list[PaperAssessment]:
    """Run the fair baseline as one direct prompt over all six PDF inputs."""
    if isinstance(model, DemoModelClient):
        return [await _direct_assess(model, request, paper, Path(paper.pdf_path) if paper.pdf_path else None)
                for paper in papers]
    if not isinstance(model, OpenAIResponsesClient) or not model._client:
        raise EvaluationError("live baseline requires an OpenAI client")
    if model.custom_compatible:
        # Keep one direct baseline request while avoiding file uploads. Every
        # paper receives the same configured per-paper budget as staged assess.
        content: list[dict[str, Any]] = [{"type": "input_text", "text": json.dumps({
            "request": request.model_dump(), "papers": [paper.model_dump() for paper in papers],
            "task": "Assess all papers directly. Return one assessment per arxiv_id."
        }, default=str)}]
        per_paper_budget = max(1, model.pdf_text_limit)
        for paper in papers:
            pages = extract_pages(paper.pdf_path) if paper.pdf_path else []
            text = "\n\n".join(f"--- PAGE {index} ---\n{page}" for index, page in enumerate(pages, 1))
            if len(text) > per_paper_budget:
                text = text[:per_paper_budget] + "\n--- PDF TEXT TRUNCATED ---"
            if text:
                content.append({"type": "input_text", "text": f"--- PAPER {paper.arxiv_id} ---\n{text}"})
        data = await model._json(
            "Return strict structured assessments for every paper. For each arxiv_id, return "
            "topic_relevance, methodological_fit, evidence_usefulness, and constraint_fit as "
            "integer scores from 0 to 100, plus the summary, limitations, and evidence fields. "
            + EVIDENCE_INSTRUCTIONS,
            content, schema=BatchAssessmentOutput)
        by_id = {paper.arxiv_id: paper for paper in papers}
        assessments = []
        for item in data.get("assessments", []):
            paper = by_id.get(item.get("arxiv_id"))
            if paper is None:
                raise EvaluationError("direct baseline returned an unknown paper")
            item = {key: value for key, value in item.items() if key != "arxiv_id"}
            item["paper"] = paper.model_dump()
            assessment = PaperAssessment.model_validate(item)
            assessment.evidence = verify_evidence(assessment.evidence, extract_pages(paper.pdf_path)) if paper.pdf_path else assessment.evidence
            assessments.append(score_assessment(assessment))
        if {assessment.paper.arxiv_id for assessment in assessments} != set(by_id):
            raise EvaluationError("direct baseline must return exactly one assessment per paper")
        return assessments
    content: list[dict[str, Any]] = [{"type": "input_text", "text": json.dumps({
        "request": request.model_dump(), "papers": [paper.model_dump() for paper in papers],
        "task": "Assess all papers directly in one response using the four rubric scores. Return one assessment per paper."
    }, default=str)}]
    uploaded_ids: list[str] = []
    file_handles = []
    try:
        for paper in papers:
            if paper.pdf_path and Path(paper.pdf_path).exists():
                handle = Path(paper.pdf_path).open("rb")
                file_handles.append(handle)
                uploaded = await model._client.files.create(file=handle, purpose="user_data")
                uploaded_id = getattr(uploaded, "id", None)
                if uploaded_id:
                    uploaded_ids.append(uploaded_id)
                    content.append({"type": "input_file", "file_id": uploaded_id})
        data = await model._json(
            "Return JSON with an assessments array. Each item must contain arxiv_id, four integer scores from 0 to 100, summary, no more than three limitations, and evidence fields. "
            + EVIDENCE_INSTRUCTIONS,
            content,
        )
        raw_assessments = data.get("assessments") if isinstance(data, dict) else None
        if not isinstance(raw_assessments, list):
            raise EvaluationError("direct baseline returned no assessments array")
        by_id = {paper.arxiv_id: paper for paper in papers}
        assessments = []
        for raw in raw_assessments:
            if not isinstance(raw, dict) or raw.get("arxiv_id") not in by_id:
                raise EvaluationError("direct baseline returned an unknown paper")
            paper = by_id[raw.pop("arxiv_id")]
            raw["paper"] = paper.model_dump()
            assessment = PaperAssessment.model_validate(raw)
            assessment.evidence = (verify_evidence(assessment.evidence, extract_pages(paper.pdf_path))
                                   if paper.pdf_path else assessment.evidence)
            assessments.append(score_assessment(assessment))
        if {assessment.paper.arxiv_id for assessment in assessments} != set(by_id):
            raise EvaluationError("direct baseline must return exactly one assessment per paper")
        return assessments
    finally:
        for handle in file_handles:
            handle.close()
        for uploaded_id in uploaded_ids:
            try:
                await model._client.files.delete(uploaded_id)
            except Exception:
                pass


async def _hydrate(case: dict[str, Any], arxiv: Any, retries: int) -> tuple[list[Paper], list[dict[str, str]], int]:
    papers: list[Paper] = []
    failures: list[dict[str, str]] = []
    retry_count = 0
    metadata: dict[str, Paper] = {}
    if hasattr(arxiv, "get_papers"):
        try:
            metadata, used = await _with_retries(lambda: arxiv.get_papers(case["arxiv_ids"]), retries)
            retry_count += used
        except Exception as exc:
            # Keep per-ID diagnostics even if a batched metadata request fails.
            failures.extend({"paper_id": paper_id, "stage": "hydrate", "error": f"batch metadata: {exc}"}
                             for paper_id in case["arxiv_ids"])
            return papers, failures, retry_count
    for paper_id in case["arxiv_ids"]:
        try:
            if metadata:
                paper = metadata.get(paper_id)
            else:
                paper, used = await _with_retries(lambda paper_id=paper_id: arxiv.get_paper(paper_id), retries)
                retry_count += used
            if not paper:
                raise EvaluationError("metadata not found")
            path, used = await _with_retries(lambda paper=paper: arxiv.download_pdf(paper), retries)
            retry_count += used
            paper.pdf_path = str(path)
            paper_dict = paper.model_copy(update={"pdf_path": str(path)})
            papers.append(paper_dict)
        except Exception as exc:
            failures.append({"paper_id": paper_id, "stage": "hydrate", "error": str(exc)})
    return papers, failures, retry_count


def _assessment_output(assessments: list[PaperAssessment]) -> list[dict[str, Any]]:
    return [assessment.model_dump(mode="json") for assessment in assessments]


async def _run_case(case: dict[str, Any], papers: list[Paper], labels: dict[str, float], model: Any,
                    store: RunStore, retries: int, cost_rates: tuple[float, float] | None,
                    invocation_id: str | None = None) -> dict[str, Any]:
    request = RunRequest(research_question=case["question"], inclusion_criteria=case.get("include", []),
                         exclusion_criteria=case.get("exclude", []), categories=case.get("categories", []),
                         pinned_arxiv_ids=[paper.arxiv_id for paper in papers])
    case_result: dict[str, Any] = {"case_id": case["id"], "papers": [
        {"arxiv_id": paper.arxiv_id, "title": paper.title, "sha256": _sha256(Path(paper.pdf_path))}
        for paper in papers if paper.pdf_path
    ]}
    before = _usage_snapshot(model)
    baseline_errors: list[dict[str, str]] = []
    baseline_assessments: list[PaperAssessment] = []
    baseline_retries = 0
    baseline_start = time.perf_counter()
    try:
        baseline_assessments, used = await _with_retries(
            lambda: _direct_batch_assess(model, request, papers), retries)
        baseline_retries += used
    except Exception as exc:
        baseline_errors.append({"paper_id": "<batch>", "error": str(exc)})
    baseline_assessments = rank_assessments([score_assessment(item) for item in baseline_assessments])
    baseline_usage = _usage_delta(before, _usage_snapshot(model))
    base_ids = [item.paper.arxiv_id for item in baseline_assessments]
    baseline_read_now = [item for item in baseline_assessments if item.decision == Decision.read_now]
    baseline_verified = sum(1 for item in baseline_read_now for evidence in item.evidence if evidence.verified)
    baseline_evidence_total = sum(len(item.evidence) for item in baseline_read_now)
    baseline_result = {"model": getattr(model, "solver_model", "demo"), "ranked_papers": base_ids,
                      "assessments": _assessment_output(baseline_assessments), "errors": baseline_errors,
                      "latency_seconds": round(time.perf_counter() - baseline_start, 4),
                      "verified_evidence_rate": round(baseline_verified / baseline_evidence_total, 4) if baseline_evidence_total else 0.0,
                      "read_now_evidence_verified": baseline_verified,
                      "read_now_evidence_total": baseline_evidence_total,
                      "retry_count": baseline_retries, **baseline_usage}
    if cost_rates:
        baseline_result["estimated_cost_usd"] = round(baseline_usage["input_tokens"] * cost_rates[0] / 1000 + baseline_usage["output_tokens"] * cost_rates[1] / 1000, 6)
    baseline_result["ndcg_at_5"] = round(ndcg_at_k(base_ids, labels), 4)

    staged_start = time.perf_counter()
    reset_usage = getattr(model, "reset_usage", None)
    if reset_usage:
        reset_usage()
    staged_before = _usage_snapshot(model)
    staged_run = Run(id=f"evaluation-{invocation_id or uuid.uuid4().hex}-{case['id']}", request=request)
    store.create(staged_run)
    staged_model = model
    workflow = Workflow(store, HydratedArxiv(papers), staged_model, max_candidates=6, shortlist_size=6,
                        concurrency=3, retries=retries)
    staged_run = await workflow.execute(staged_run)
    staged_usage = _usage_delta(staged_before, _usage_snapshot(model))
    staged_assessments = staged_run.assessments
    staged_ids = [item.paper.arxiv_id for item in staged_assessments]
    read_now = [item for item in staged_assessments if item.decision == Decision.read_now]
    verified = sum(1 for item in read_now for evidence in item.evidence if evidence.verified)
    evidence_total = sum(len(item.evidence) for item in read_now)
    solver_result = {"model": getattr(model, "solver_model", "demo"), "ranked_papers": staged_ids,
                     "assessments": _assessment_output(staged_assessments), "errors": [
                         {"paper_id": item.paper.arxiv_id, "error": item.error}
                         for item in staged_assessments if item.error
                     ], "latency_seconds": round(time.perf_counter() - staged_start, 4), **staged_usage,
                     "verified_evidence_rate": round(verified / evidence_total, 4) if evidence_total else 0.0,
                     "read_now_evidence_verified": verified, "read_now_evidence_total": evidence_total,
                     "retry_count": staged_run.retry_count,
                     "ndcg_at_5": round(ndcg_at_k(staged_ids, labels), 4)}
    if cost_rates:
        solver_result["estimated_cost_usd"] = round(staged_usage["input_tokens"] * cost_rates[0] / 1000 + staged_usage["output_tokens"] * cost_rates[1] / 1000, 6)
    case_result["baseline"] = baseline_result
    case_result["solver"] = solver_result
    case_result["failures"] = baseline_errors + solver_result["errors"]
    return case_result


def _demo_papers(case: dict[str, Any], cache_dir: Path) -> list[Paper]:
    return [Paper(arxiv_id=paper_id, title=f"{case['question']} — paper {index}", abstract=case["question"])
            for index, paper_id in enumerate(case["arxiv_ids"], 1)]


async def run_evaluation(manifest: dict[str, Any], output_dir: Path, *, demo: bool = False,
                         api_key: str | None = None, base_url: str | None = None, retries: int = 2,
                         input_cost_per_1k: float | None = None,
                         output_cost_per_1k: float | None = None) -> tuple[dict[str, Any], Path, Path]:
    """Run and persist one benchmark comparison.

    Returns ``(full_result, versioned_path, latest_summary_path)``. Real mode
    requires a hosted API key or compatible base URL before any request; demo
    mode never reads a key.
    """
    api_key = api_key or os.getenv("PAPERROUTE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("PAPERROUTE_OPENAI_BASE_URL") or None
    invocation_id = uuid.uuid4().hex
    if not demo and not api_key and not base_url:
        raise EvaluationError("--live requires OPENAI_API_KEY/PAPERROUTE_API_KEY or PAPERROUTE_OPENAI_BASE_URL; use --demo-live for an offline run")
    cases = manifest["cases"]
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_dir = output_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    cost_rates = (input_cost_per_1k, output_cost_per_1k) if input_cost_per_1k is not None and output_cost_per_1k is not None else None
    if demo:
        model: Any = DemoModelClient()
        arxiv: Any = None
    else:
        model = OpenAIResponsesClient(
            api_key,
            solver_model=os.getenv("PAPERROUTE_SOLVER_MODEL", "gpt-5.6-terra"),
            judge_model=os.getenv("PAPERROUTE_JUDGE_MODEL", "gpt-5.6"),
            base_url=base_url,
            timeout=float(os.getenv("PAPERROUTE_TIMEOUT", "45")),
            pdf_text_limit=int(os.getenv("PAPERROUTE_PDF_TEXT_LIMIT", "60000")),
        )
        arxiv = ArxivClient(cache_dir=papers_dir)
    hydrated: dict[str, tuple[list[Paper], list[dict[str, str]], int]] = {}
    for case in cases:
        if demo:
            case_papers = _demo_papers(case, papers_dir)
            arxiv_case = DemoArxiv(case_papers, papers_dir)
        else:
            arxiv_case = arxiv
        hydrated[case["id"]] = await _hydrate(case, arxiv_case, retries)
    labels = _labels_from_payload(manifest, cases)
    labels_source = "manifest"
    judge_diagnostics: dict[str, Any]
    if labels is None:
        paper_map = {case_id: {paper.arxiv_id: paper for paper in values[0]}
                     for case_id, values in hydrated.items()}
        judge_diagnostics = {}
        labels = await _judge_all(cases, model, demo, paper_map, judge_diagnostics)
        labels_source = "demo-lexical" if demo else "two-pass-plus-tie-break"
    else:
        source_judge = manifest.get("judge") if isinstance(manifest.get("judge"), dict) else {}
        prior = source_judge.get("diagnostics") or manifest.get("judge_diagnostics")
        if isinstance(prior, dict):
            judge_diagnostics = {**prior, "source": "labeled-manifest"}
        else:
            judge_diagnostics = {"available": False, "source": "labeled-manifest-unavailable",
                                 "reason": "No judge diagnostics were present in the reused labels artifact."}
    all_case_results: list[dict[str, Any]] = []
    store = RunStore(output_dir / "evaluation.sqlite3")
    total_hydration_failures = 0
    total_retries = 0
    for case in cases:
        papers, failures, used_retries = hydrated[case["id"]]
        total_hydration_failures += len(failures)
        result = await _run_case(case, papers, labels[case["id"]], model, store, retries, cost_rates,
                                 invocation_id)
        result["failures"] = failures + result["failures"]
        result["retry_count"] = (used_retries + result["baseline"].get("retry_count", 0) +
                                  result["solver"].get("retry_count", 0))
        total_retries += result["retry_count"]
        all_case_results.append(result)
    solver_cases = [{"case_id": result["case_id"], **result["solver"]} for result in all_case_results]
    baseline_cases = [{"case_id": result["case_id"], **result["baseline"]} for result in all_case_results]
    solver_ndcg = [case["ndcg_at_5"] for case in solver_cases]
    baseline_ndcg = [case["ndcg_at_5"] for case in baseline_cases]
    solver_latency = [case["latency_seconds"] for case in solver_cases]
    baseline_latency = [case["latency_seconds"] for case in baseline_cases]
    solver_verified_total = sum(case.get("read_now_evidence_verified", 0) for case in solver_cases)
    solver_evidence_total = sum(case.get("read_now_evidence_total", 0) for case in solver_cases)
    baseline_verified_total = sum(case.get("read_now_evidence_verified", 0) for case in baseline_cases)
    baseline_evidence_total = sum(case.get("read_now_evidence_total", 0) for case in baseline_cases)
    failure_count = sum(len(case["failures"]) for case in all_case_results)
    generated_at = datetime.now(UTC).isoformat()
    result = {"evaluation_version": "1.0", "generated_at": generated_at,
              "manifest_version": manifest["version"], "labels_source": labels_source,
              "labels": labels, "solver_model": getattr(model, "solver_model", "demo"),
              "judge_model": getattr(model, "judge_model", "demo"),
              "judge_diagnostics": judge_diagnostics,
              "prompt_hashes": {"baseline": prompt_hash("direct-single-prompt-v1"), "solver": prompt_hash("paperroute-staged-v1")},
              "retries": retries, "cases": all_case_results,
              "baseline": {"model": getattr(model, "solver_model", "demo"), "cases": baseline_cases},
              "solver": {"model": getattr(model, "solver_model", "demo"), "cases": solver_cases},
              "metrics": {"baseline_mean_ndcg_at_5": round(sum(baseline_ndcg) / len(baseline_ndcg), 4),
                          "solver_mean_ndcg_at_5": round(sum(solver_ndcg) / len(solver_ndcg), 4),
                          "delta_ndcg_at_5": round(sum(solver_ndcg) / len(solver_ndcg) - sum(baseline_ndcg) / len(baseline_ndcg), 4),
                          "solver_mean_latency_seconds": round(sum(solver_latency) / len(solver_latency), 4),
                          "baseline_mean_latency_seconds": round(sum(baseline_latency) / len(baseline_latency), 4),
                          "solver_mean_verified_evidence_rate": round(solver_verified_total / solver_evidence_total, 4) if solver_evidence_total else 0.0,
                          "solver_read_now_evidence_verified": solver_verified_total,
                          "solver_read_now_evidence_total": solver_evidence_total,
                          "baseline_read_now_evidence_verified": baseline_verified_total,
                          "baseline_read_now_evidence_total": baseline_evidence_total,
                          "baseline_verified_evidence_rate": round(baseline_verified_total / baseline_evidence_total, 4) if baseline_evidence_total else 0.0,
                          "solver_read_now_verified_evidence_rate": round(solver_verified_total / solver_evidence_total, 4) if solver_evidence_total else 0.0,
                          "solver_input_tokens": sum(case.get("input_tokens", 0) for case in solver_cases),
                          "solver_output_tokens": sum(case.get("output_tokens", 0) for case in solver_cases),
                          "baseline_input_tokens": sum(case.get("input_tokens", 0) for case in baseline_cases),
                          "baseline_output_tokens": sum(case.get("output_tokens", 0) for case in baseline_cases),
                          "hydration_failures": total_hydration_failures, "failure_count": failure_count,
                          "partial_case_count": sum(bool(case["failures"]) for case in all_case_results),
                          "retry_count": total_retries, "case_count": len(all_case_results)}}
    if cost_rates:
        result["metrics"]["baseline_estimated_cost_usd"] = round(sum(case.get("estimated_cost_usd", 0) for case in baseline_cases), 6)
        result["metrics"]["solver_estimated_cost_usd"] = round(sum(case.get("estimated_cost_usd", 0) for case in solver_cases), 6)
        result["cost_rates"] = {"input_per_1k": input_cost_per_1k, "output_per_1k": output_cost_per_1k}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    versioned_path = output_dir / f"evaluation-{stamp}-{invocation_id}.json"
    versioned_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    latest = {"evaluation_version": result["evaluation_version"], "generated_at": generated_at,
              "manifest_version": result["manifest_version"], "metrics": result["metrics"],
              "solver_model": result["solver_model"], "baseline_model": result["baseline"]["model"],
              "case_count": len(all_case_results), "result_file": versioned_path.name,
              "judge_diagnostics": judge_diagnostics}
    latest_path = output_dir / "latest-evaluation.json"
    latest_path.write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
    if not demo and labels_source != "manifest":
        labeled = copy.deepcopy(manifest)
        labeled["labels"] = labels
        labeled.setdefault("judge", {})
        labeled["judge"]["labels_frozen"] = True
        labeled["judge"]["label_status"] = "Frozen by evaluation runner"
        labeled["judge"]["diagnostics"] = judge_diagnostics
        labeled["judge"]["label_generation"] = {"generated_at": generated_at,
                                                    "source": labels_source,
                                                    "model": judge_diagnostics.get("model", "")}
        (output_dir / f"benchmark-labeled-{manifest['version']}.json").write_text(json.dumps(labeled, indent=2) + "\n", encoding="utf-8")
    return result, versioned_path, latest_path

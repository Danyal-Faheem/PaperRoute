from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

from .arxiv_client import ArxivClient, OfflineArxivClient
from .db import RunStore
from .demo import DemoModelClient
from .export import report_json, report_markdown
from .models import Run, RunRequest
from .openai_client import OpenAIResponsesClient
from .settings import Settings, get_settings
from .workflow import Workflow

try:
    from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
    from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:  # Allows importing core modules without web dependencies installed.
    FastAPI = None  # type: ignore


def create_app(settings: Settings | None = None, store: RunStore | None = None,
               arxiv: ArxivClient | None = None, model=None):
    if FastAPI is None:
        raise RuntimeError("fastapi is required to create the web app")
    settings = settings or get_settings()
    settings.prepare_dirs()
    injected_store = store is not None
    store = store or RunStore(settings.database_path)
    # A caller-provided store denotes an isolated app/test workspace. Avoid
    # accidentally rendering the repository's shared runtime evaluation while
    # keeping the production default at data/runtime/latest-evaluation.json.
    if injected_store and settings.evaluation_path == Path("data/runtime/latest-evaluation.json"):
        settings = replace(settings, evaluation_path=settings.database_path.parent / "latest-evaluation.json")
    arxiv = arxiv or (OfflineArxivClient(settings.cache_dir) if settings.demo_mode else
                      ArxivClient(settings.arxiv_base_url, settings.cache_dir, settings.request_timeout))
    model = model or (DemoModelClient() if settings.demo_mode else OpenAIResponsesClient(
        (settings.openai_api_key if settings.openai_api_key and settings.openai_api_key.strip().lower() not in {"replace-me", "changeme"} else None),
        settings.solver_model, settings.judge_model,
        base_url=settings.openai_base_url, timeout=settings.request_timeout, pdf_text_limit=settings.pdf_text_limit))
    workflow = Workflow(store, arxiv, model, settings.max_candidates, settings.shortlist_size,
                        settings.concurrency, settings.retry_count)
    app = FastAPI(title="PaperRoute", version="0.1.0")
    app.state.store, app.state.workflow = store, workflow
    template_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(template_dir))
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    async def run_work(run: Run) -> None:
        await workflow.execute(run)

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> Response:
        return templates.TemplateResponse(request=request, name="home.html", context=_base_context({"form_action": "/runs"}))

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/benchmark", response_class=HTMLResponse)
    async def benchmark_page(request: Request) -> Response:
        return templates.TemplateResponse(request=request, name="benchmark.html", context=_benchmark_context(_base_context({})))

    @app.get("/new", response_class=HTMLResponse)
    async def new_run_page(request: Request) -> Response:
        return templates.TemplateResponse(request=request, name="new_run.html", context=_base_context({"form_action": "/runs"}))

    @app.post("/runs")
    async def create_html_run(request: Request, background_tasks: BackgroundTasks,
                              research_question: str = Form(...), inclusion_criteria: str = Form(""),
                              exclusion_criteria: str = Form(""), categories: list[str] = Form(default_factory=list),  # noqa: B008
                              date_from: str = Form(""), date_to: str = Form(""),
                              pinned_arxiv_ids: str = Form("")) -> Response:
        def lines(value: str | list[str]) -> list[str]:
            if isinstance(value, list):
                return [x.strip() for x in value if x.strip()]
            return [x.strip() for x in value.replace(",", "\n").splitlines() if x.strip()]
        def date(value: str, end: bool = False):
            from datetime import UTC, datetime, time
            from datetime import date as date_type
            if not value:
                return None
            parsed = date_type.fromisoformat(value)
            # HTML date fields represent whole days. Make both boundaries
            # aware, and include the complete selected end day.
            boundary = time.max if end else time.min
            return datetime.combine(parsed, boundary, tzinfo=UTC)
        form = {"research_question": research_question, "inclusion_criteria": inclusion_criteria,
                "exclusion_criteria": exclusion_criteria, "categories": categories,
                "date_from": date_from, "date_to": date_to, "pinned_arxiv_ids": pinned_arxiv_ids}
        try:
            payload = RunRequest(research_question=research_question, inclusion_criteria=lines(inclusion_criteria),
                exclusion_criteria=lines(exclusion_criteria), categories=lines(categories), date_from=date(date_from),
                date_to=date(date_to, end=True), pinned_arxiv_ids=lines(pinned_arxiv_ids))
        except Exception as exc:
            return templates.TemplateResponse(request=request, name="home.html", status_code=422,
                                              context=_base_context({"error": str(exc), "form": form, "form_action": "/runs"}))
        run = Run(id=uuid.uuid4().hex, request=payload)
        store.create(run)
        background_tasks.add_task(run_work, run)
        return RedirectResponse(url=f"/runs/{run.id}", status_code=303)

    @app.get("/runs/{run_id}/status", response_class=HTMLResponse)
    async def html_status(request: Request, run_id: str) -> Response:
        run = store.get(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        context = _run_context(run)
        status_html = templates.get_template("_run_status.html").render(**context)
        terminal = run.status.value in {"completed", "failed"}
        results_html = templates.get_template("_results.html").render(**context) if terminal else ""
        polling = "" if terminal else f' hx-get="/runs/{run.id}/status" hx-trigger="load, every 2s" hx-target="#run-status" hx-swap="outerHTML"'
        return HTMLResponse(f'<div id="run-status"{polling}>{status_html}{results_html}</div>')

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(request: Request, run_id: str) -> Response:
        run = store.get(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        return templates.TemplateResponse(request=request, name="run.html", context=_run_context(run))

    @app.post("/api/runs", response_model=Run, status_code=202)
    async def create_run(payload: RunRequest, background_tasks: BackgroundTasks) -> Run:
        run = Run(id=uuid.uuid4().hex, request=payload)
        store.create(run)
        background_tasks.add_task(run_work, run)
        return run

    @app.get("/api/runs/{run_id}", response_model=Run)
    async def get_run(run_id: str) -> Run:
        run = store.get(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        return run

    def _run_context(run: Run) -> dict:
        papers = []
        evidence_total = 0
        evidence_verified = 0
        errors = []
        for assessment in run.assessments:
            row = assessment.model_dump(mode="json")
            row["paper"] = assessment.paper.model_dump(mode="json")
            row["scores"] = {"topic_relevance": assessment.topic_relevance,
                              "methodological_fit": assessment.methodological_fit,
                              "evidence_usefulness": assessment.evidence_usefulness,
                              "constraint_fit": assessment.constraint_fit}
            papers.append(row)
            evidence_total += len(assessment.evidence)
            evidence_verified += sum(item.verified for item in assessment.evidence)
            if assessment.error:
                errors.append({"paper_title": assessment.paper.title, "message": assessment.error})
        progress_by_status = {"queued": 2, "planning": 15, "searching": 30, "screening": 45,
                              "analyzing": 70, "ranking": 88, "completed": 100, "failed": 100}
        stages = [("query_planner", "Planning search"), ("abstract_screener", "Shortlisting candidates"),
                  ("paper_analyst", "Reading papers"), ("ranker", "Comparing evidence")]
        current_index = {"planning": 0, "searching": 0, "screening": 1, "analyzing": 2, "ranking": 3}.get(run.status.value, 4)
        agents = [{"name": label, "role": role,
                   "status": "complete" if i < current_index or run.status.value == "completed" else
                             ("running" if i == current_index and run.status.value != "failed" else "pending"),
                   "message": "Complete" if i < current_index else ("In progress" if i == current_index else "Waiting")}
                  for i, (role, label) in enumerate(stages)]
        trajectories = [{"timestamp": event.timestamp, "agent": event.role,
                         "summary": " — ".join(x for x in (event.event, event.input_summary, event.output_summary) if x)}
                        for event in run.trajectories]
        evidence_rate = round(100 * evidence_verified / evidence_total, 1) if evidence_total else None
        return {"run": run, "papers": papers, "research_question": run.request.research_question,
                "progress": progress_by_status.get(run.status.value, 0), "agents": agents,
                "errors": errors, "trajectories": trajectories, "verified_evidence_rate": evidence_rate,
                "poll_url": f"/runs/{run.id}/status", "export_md_url": f"/api/runs/{run.id}/report.md",
                "export_json_url": f"/api/runs/{run.id}/report.json"}

    def _base_context(context: dict) -> dict:
        warning = settings.configuration_warning
        if settings.demo_mode:
            warning = {"code": "demo_mode", "mode": "demo",
                       "message": "Demo mode is active; no network or model provider calls will be made."}
        elif warning:
            warning = {"code": "missing_openai_api_key", "message": warning}
        context.update({"configured": settings.configured, "demo_mode": settings.demo_mode,
                        "configuration_warning": warning,
                        "benchmark_url": "/benchmark"})
        return context

    def _benchmark_context(context: dict) -> dict:
        import json
        payload = None
        try:
            if settings.evaluation_path.exists():
                candidate = json.loads(settings.evaluation_path.read_text(encoding="utf-8"))
                if isinstance(candidate, dict) and isinstance(candidate.get("metrics"), dict):
                    payload = candidate
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if payload is None:
            context.update({"cases": 0, "case_count": 0, "metrics": {}, "evaluation": {}, "ndcg": None})
            return context
        metrics = payload["metrics"]
        solver = payload.get("solver") if isinstance(payload.get("solver"), dict) else {}
        cases = solver.get("cases", []) if isinstance(solver.get("cases", []), list) else []
        # Summary artifacts intentionally omit the full per-case solver payload;
        # prefer their explicit case count before falling back to full artifacts.
        case_count = payload.get("case_count")
        if not isinstance(case_count, int):
            case_count = metrics.get("case_count")
        if not isinstance(case_count, int):
            case_count = len(cases)
        baseline_latency = metrics.get("baseline_mean_latency_seconds")
        solver_latency = metrics.get("solver_mean_latency_seconds")
        latency_faster_pct = None
        if isinstance(baseline_latency, (int, float)) and baseline_latency > 0 and isinstance(solver_latency, (int, float)):
            latency_faster_pct = round((baseline_latency - solver_latency) / baseline_latency * 100, 1)
        context.update({"evaluation": payload, "latest_evaluation": payload, "cases": case_count,
                        "case_count": case_count, "metrics": metrics,
                        "baseline_ndcg": metrics.get("baseline_mean_ndcg_at_5"),
                        "solver_ndcg": metrics.get("solver_mean_ndcg_at_5", metrics.get("mean_ndcg_at_5")),
                        "ndcg_delta": metrics.get("delta_ndcg_at_5"),
                        "baseline_latency": baseline_latency, "solver_latency": solver_latency,
                        "latency_faster_pct": latency_faster_pct,
                        "generated_at": payload.get("generated_at"),
                        "solver_model": solver.get("model", payload.get("solver_model", "")),
                        "baseline_model": payload.get("baseline_model", "")})
        return context

    @app.get("/api/runs/{run_id}/report.md", response_class=PlainTextResponse)
    async def markdown_report(run_id: str) -> str:
        run = store.get(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        return report_markdown(run)

    @app.get("/api/runs/{run_id}/report.json")
    async def json_report(run_id: str):
        run = store.get(run_id)
        if not run:
            raise HTTPException(404, "run not found")
        return Response(report_json(run), media_type="application/json")

    return app


try:
    app = create_app()
except RuntimeError:
    app = None

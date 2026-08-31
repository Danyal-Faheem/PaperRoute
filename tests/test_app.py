from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from paperroute.app import create_app
from paperroute.db import RunStore
from paperroute.models import Paper, PaperAssessment
from paperroute.openai_client import OpenAIResponsesClient
from paperroute.settings import Settings


class AppArxivStub:
    def __init__(self, tmp_path: Path):
        self.paper = Paper(arxiv_id="2401.12345", title="Attention paper", abstract="attention")
        self.path = tmp_path / "paper.pdf"
        self.path.write_bytes(b"not a pdf")

    async def get_paper(self, arxiv_id):
        return self.paper if arxiv_id == self.paper.arxiv_id else None

    async def search(self, query, max_results=20, categories=None):
        return [self.paper]

    async def download_pdf(self, paper):
        return self.path


class AppModelStub:
    async def plan_query(self, request):
        return "attention"

    async def screen(self, request, papers):
        return [papers[0].arxiv_id]

    async def assess(self, request, paper, pdf_path):
        return PaperAssessment(paper=paper, topic_relevance=40, methodological_fit=40,
                               evidence_usefulness=0, constraint_fit=40, summary="stub")

    async def rank(self, request, assessments):
        return assessments


class NeverNetworkArxiv:
    async def get_paper(self, arxiv_id):
        raise AssertionError("network access must not occur when the API key is absent")

    async def search(self, query, max_results=20, categories=None):
        raise AssertionError("network access must not occur when the API key is absent")

    async def download_pdf(self, paper):
        raise AssertionError("network access must not occur when the API key is absent")


def test_create_run_and_get_markdown_and_json_exports(tmp_path):
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers", demo_mode=True)
    store = RunStore(settings.database_path)
    app = create_app(settings, store=store, arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "PaperRoute" in home.text
        health = client.get("/healthz")
        assert health.status_code == 200

        response = client.post("/api/runs", json={"research_question": "attention in translation"})
        assert response.status_code == 202
        run_id = response.json()["id"]

        fetched = client.get(f"/api/runs/{run_id}")
        assert fetched.status_code == 200
        assert fetched.json()["status"] == "completed"
        page = client.get(f"/runs/{run_id}")
        assert page.status_code == 200
        assert "Attention paper" in page.text
        status = client.get(f"/runs/{run_id}/status")
        assert status.status_code == 200
        assert "A ranked reading list" in status.text

        markdown = client.get(f"/api/runs/{run_id}/report.md")
        assert markdown.status_code == 200
        assert "# PaperRoute report" in markdown.text
        assert "Attention paper" in markdown.text

        exported = client.get(f"/api/runs/{run_id}/report.json")
        assert exported.status_code == 200
        assert exported.headers["content-type"].startswith("application/json")
        assert exported.json()["id"] == run_id


def test_missing_run_returns_json_404(tmp_path):
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers")
    app = create_app(settings, store=RunStore(settings.database_path), arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        response = client.get("/api/runs/does-not-exist")
        assert response.status_code == 404
        assert client.get("/runs/does-not-exist").status_code == 404
        assert client.get("/runs/does-not-exist/status").status_code == 404
        assert client.get("/api/runs/does-not-exist/report.md").status_code == 404
        assert client.get("/api/runs/does-not-exist/report.json").status_code == 404


def test_html_benchmark_new_form_status_and_validation(tmp_path):
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers")
    app = create_app(settings, store=RunStore(settings.database_path), arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        assert client.get("/benchmark").status_code == 200
        assert "benchmark" in client.get("/benchmark").text.casefold()
        assert client.get("/new").status_code == 200
        invalid = client.post("/runs", data={"research_question": "x"})
        assert invalid.status_code == 422
        valid = client.post("/runs", data={
            "research_question": "attention in translation",
            "inclusion_criteria": "attention, translation",
            "exclusion_criteria": "biology\nweather",
            "categories": "cs.AI",
            "pinned_arxiv_ids": "2401.12345",
        }, follow_redirects=False)
        assert valid.status_code == 303
        assert valid.headers["location"].startswith("/runs/")


def test_html_category_dropdown_is_curated_and_reaches_run_payload(tmp_path):
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers")
    app = create_app(settings, store=RunStore(settings.database_path), arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        page = client.get("/")
        assert '<select id="categories" name="categories" multiple' in page.text
        assert '<input id="categories"' not in page.text
        assert '<optgroup label="Computer Science">' in page.text
        assert '<optgroup label="Mathematics">' in page.text
        assert "Command (Mac)" in page.text
        assert "cs.AR — Hardware architecture" in page.text
        assert "cs.OS — Operating systems" in page.text
        assert "cs.PF — Performance" in page.text
        assert "cs.PL — Programming languages" in page.text
        values = re.findall(r'<option value="([^"]+)"', page.text)
        assert values and all(value.startswith(("cs.", "math.")) for value in values)
        assert "cs.AR" in values
        assert re.search(r'<option value="cs\.AI" selected>', page.text)
        assert re.search(r'<option value="cs\.CL" selected>', page.text)

        invalid = client.post("/runs", data={"research_question": "x",
                                              "categories": ["cs.AR", "cs.PL"]})
        assert invalid.status_code == 422
        assert re.search(r'<option value="cs\.AR" selected>', invalid.text)
        assert re.search(r'<option value="cs\.PL" selected>', invalid.text)

        submitted = client.post("/runs", data={
            "research_question": "NVIDIA and AMD hardware architecture research",
            "categories": ["cs.AR", "cs.PF", "cs.PL"],
        }, follow_redirects=False)
        assert submitted.status_code == 303
        run_id = submitted.headers["location"].rsplit("/", 1)[-1]
        payload = client.get(f"/api/runs/{run_id}").json()
        assert payload["request"]["categories"] == ["cs.AR", "cs.PF", "cs.PL"]


def test_explicit_demo_mode_web_run_ranks_six_papers(tmp_path):
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers", demo_mode=True)
    app = create_app(settings, store=RunStore(settings.database_path))
    with TestClient(app) as client:
        response = client.post("/api/runs", json={
            "research_question": "How do retrieval-augmented language models reduce factual errors in biomedical question answering?",
            "inclusion_criteria": ["retrieval", "factuality"],
            "categories": ["cs.AI", "cs.CL"],
        })
        assert response.status_code == 202
        run_id = response.json()["id"]
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["status"] == "completed"
        assert len(result["assessments"]) == 6
        assert len({item["paper"]["arxiv_id"] for item in result["assessments"]}) == 6
        assert all(evidence["verified"] for item in result["assessments"] for evidence in item["evidence"])


def test_non_demo_no_key_web_run_fails_before_network(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PAPERROUTE_API_KEY", raising=False)
    monkeypatch.delenv("PAPERROUTE_OPENAI_BASE_URL", raising=False)
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers", demo_mode=False)
    app = create_app(settings, store=RunStore(settings.database_path), arxiv=NeverNetworkArxiv(),
                     model=OpenAIResponsesClient(None))
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"research_question": "no key run"})
        assert response.status_code == 202
        run_id = response.json()["id"]
        result = client.get(f"/api/runs/{run_id}").json()
        assert result["status"] == "failed"
        assert "OPENAI_API_KEY" in result["error"]


def test_benchmark_page_renders_placeholder_until_latest_summary_exists(tmp_path):
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers")
    app = create_app(settings, store=RunStore(settings.database_path), arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        page = client.get("/benchmark")
        assert page.status_code == 200
        assert "No evaluation run yet" in page.text


def test_benchmark_page_loads_latest_evaluation_summary(tmp_path):
    summary = tmp_path / "latest-evaluation.json"
    summary.write_text(json.dumps({
        "generated_at": "2026-08-28T16:00:00Z",
        "solver_model": "demo",
        "baseline_model": "demo",
        "solver": {"cases": [{"case_id": "case-01"}] * 10},
        "metrics": {"baseline_mean_ndcg_at_5": 0.4, "solver_mean_ndcg_at_5": 0.8,
                     "delta_ndcg_at_5": 0.4, "case_count": 10},
    }), encoding="utf-8")
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers",
                        evaluation_path=summary)
    app = create_app(settings, store=RunStore(settings.database_path), arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        page = client.get("/benchmark")
        assert page.status_code == 200
        assert "NDCG@5" in page.text
        assert "0.8" in page.text
        assert "0.4" in page.text
        assert "Demo / offline result" in page.text


def test_benchmark_page_renders_latest_local_evaluation_metrics(tmp_path):
    summary = tmp_path / "latest-evaluation.json"
    summary.write_text(json.dumps({
        "generated_at": "2026-08-28T19:40:43.493229+00:00",
        "solver_model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
        "baseline_model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
        "case_count": 10,
        "metrics": {
            "baseline_mean_latency_seconds": 187.082,
            "solver_mean_latency_seconds": 106.586,
            "baseline_mean_ndcg_at_5": 0.997,
            "solver_mean_ndcg_at_5": 0.9957,
            "delta_ndcg_at_5": -0.0013,
            "baseline_verified_evidence_rate": 1.0,
            "solver_mean_verified_evidence_rate": 1.0,
            "baseline_read_now_evidence_verified": 72,
            "baseline_read_now_evidence_total": 72,
            "solver_read_now_evidence_verified": 80,
            "solver_read_now_evidence_total": 80,
            "failure_count": 0,
            "partial_case_count": 0,
        },
    }), encoding="utf-8")
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers",
                        evaluation_path=summary)
    app = create_app(settings, store=RunStore(settings.database_path), arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        page = client.get("/benchmark")
        assert page.status_code == 200
        assert "Same 10 fixed research questions" in page.text
        assert "Faster first passes" in page.text
        assert "187.082s" in page.text
        assert "106.586s" in page.text
        assert "43.0% faster" in page.text
        assert "-0.0013 · Pass" in page.text
        assert "100.0% (72/72)" in page.text
        assert "100.0% (80/80)" in page.text
        assert "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL" in page.text
        assert "Demo / offline result" not in page.text


def test_nonterminal_status_fragment_includes_htmx_polling(tmp_path):
    from paperroute.models import Run, RunRequest

    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers")
    store = RunStore(settings.database_path)
    pending = Run(id="pending", request=RunRequest(research_question="pending research question"))
    store.create(pending)
    app = create_app(settings, store=store, arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        status = client.get("/runs/pending/status")
        assert status.status_code == 200
        assert 'hx-get="/runs/pending/status"' in status.text


def test_malformed_latest_summary_is_treated_as_missing(tmp_path):
    summary = tmp_path / "latest.json"
    summary.write_text("{not json", encoding="utf-8")
    settings = Settings(database_path=tmp_path / "run.sqlite3", cache_dir=tmp_path / "papers",
                        evaluation_path=summary)
    app = create_app(settings, store=RunStore(settings.database_path), arxiv=AppArxivStub(tmp_path), model=AppModelStub())
    with TestClient(app) as client:
        page = client.get("/benchmark")
        assert page.status_code == 200
        assert "No evaluation run yet" in page.text


def test_console_entrypoint_delegates_to_uvicorn(monkeypatch):
    import paperroute.main as main_module

    calls = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: calls.append((args, kwargs)))
    main_module.run()
    assert calls[0][0] == ("paperroute.app:app",)

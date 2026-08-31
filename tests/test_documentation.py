from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_reproduction_guide_covers_offline_live_and_docker_paths():
    guide = (ROOT / "docs/reproduction.md").read_text(encoding="utf-8")
    for required in (
        "uv sync --locked --extra dev",
        "make test",
        "make lint",
        "make eval",
        "--demo-live",
        "--live",
        "docker compose up --build",
        "PAPERROUTE_OPENAI_BASE_URL",
        "PAPERROUTE_JUDGE_MODEL",
        "PAPERROUTE_TIMEOUT=300",
        "49 minutes",
        "Case 02",
        "2303.08774",
    ):
        assert required in guide


def test_readme_is_project_facing_and_links_core_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "graduate students", "## Features", "## Architecture", "## Local setup and run",
        "## Demo and evaluation", "## Docker, configuration, and security",
        "docs/reproduction.md", "docs/evaluation.md", "data/benchmark.json",
    ):
        assert required in readme


def test_published_labels_and_results_have_the_measured_contract():
    labels = json.loads((ROOT / "data/benchmark-labeled-1.0.json").read_text(encoding="utf-8"))
    results = json.loads((ROOT / "docs/evaluation-results.json").read_text(encoding="utf-8"))
    assert labels["judge"]["labels_frozen"] is True
    assert len(labels["cases"]) == 10
    assert labels["judge"]["diagnostics"]["agreements"] == 60
    assert results["contract"]["result"].startswith("PASS")
    assert len(results["per_case"]) == 10
    assert results["final"]["delta_ndcg_at_5"] == -0.0013
    assert results["final"]["latency_reduction"] == 0.4303
    assert results["final"]["solver_read_now_evidence"] == {"verified": 80, "total": 80, "rate": 1.0}
    assert results["artifacts"]["frozen_labels"]["published"] is True


def test_project_docs_do_not_claim_live_results_are_pending():
    paths = [ROOT / "README.md", ROOT / "docs/reproduction.md", ROOT / "docs/evaluation.md"]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "pending" not in combined.casefold()
    assert "not run—api key required" not in combined.casefold()

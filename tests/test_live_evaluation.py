from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.evaluate import load_manifest, main

from paperroute.arxiv_client import _demo_pdf_bytes
from paperroute.evaluation_runner import EvaluationError, _direct_batch_assess, run_evaluation
from paperroute.models import Paper, RunRequest
from paperroute.openai_client import OpenAIResponsesClient

MANIFEST = Path(__file__).parents[1] / "data" / "benchmark.json"


class _Responses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": json.dumps(self.payload)})()


class _Files:
    def __init__(self):
        self.created = []
        self.deleted = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return type("File", (), {"id": f"file-{len(self.created)}"})()

    async def delete(self, file_id):
        self.deleted.append(file_id)


class _OpenAI:
    def __init__(self, payload):
        self.responses = _Responses(payload)
        self.files = _Files()


def test_demo_live_completes_all_cases_and_writes_reproducible_artifacts(tmp_path):
    manifest = load_manifest(MANIFEST)
    result, versioned, latest = asyncio.run(run_evaluation(manifest, tmp_path, demo=True))
    assert result["evaluation_version"] == "1.0"
    assert result["labels_source"] == "demo-lexical"
    assert result["metrics"]["case_count"] == 10
    assert versioned.exists()
    assert latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8"))["result_file"] == versioned.name
    assert all(len(case["papers"]) == 6 for case in result["cases"])
    assert all(len(case["baseline"]["ranked_papers"]) == 6 for case in result["cases"])
    assert all(len(case["solver"]["ranked_papers"]) == 6 for case in result["cases"])
    assert all(len(item["sha256"]) == 64 for case in result["cases"] for item in case["papers"])


def test_repeated_evaluations_share_output_dir_without_run_collisions(tmp_path):
    manifest = load_manifest(MANIFEST)

    first, first_path, _ = asyncio.run(run_evaluation(manifest, tmp_path, demo=True))
    second, second_path, _ = asyncio.run(run_evaluation(manifest, tmp_path, demo=True))

    assert first["metrics"]["case_count"] == second["metrics"]["case_count"] == 10
    assert first_path != second_path
    assert first_path.exists() and second_path.exists()
    from paperroute.db import RunStore

    runs = RunStore(tmp_path / "evaluation.sqlite3").list_recent(limit=25)
    assert len(runs) == 20
    assert len({run.id for run in runs}) == 20


def test_live_baseline_uses_one_batch_model_request_for_six_papers(tmp_path):
    papers = []
    assessments = []
    for index in range(6):
        pdf = tmp_path / f"paper-{index}.pdf"
        pdf.write_bytes(b"pdf")
        paper = Paper(arxiv_id=f"2401.0000{index + 1}", title=f"Paper {index}", pdf_path=str(pdf))
        papers.append(paper)
        assessments.append({"arxiv_id": paper.arxiv_id, "topic_relevance": 60,
                            "methodological_fit": 50, "evidence_usefulness": 40,
                            "constraint_fit": 30, "summary": "direct"})
    fake = _OpenAI({"assessments": assessments})
    model = OpenAIResponsesClient("key", client=fake)
    result = asyncio.run(_direct_batch_assess(model, RunRequest(research_question="test question"), papers))
    assert len(result) == 6
    assert len(fake.responses.calls) == 1
    assert len(fake.files.created) == 6
    assert fake.files.deleted == [f"file-{index}" for index in range(1, 7)]


def test_hosted_batch_baseline_verifies_evidence_with_local_pdf_matcher(tmp_path):
    paper = Paper(arxiv_id="2401.00001", title="Paper", abstract="verbatim text")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_demo_pdf_bytes(paper))
    paper.pdf_path = str(pdf)
    payload = {"assessments": [{
        "arxiv_id": paper.arxiv_id,
        "topic_relevance": 80,
        "methodological_fit": 70,
        "evidence_usefulness": 60,
        "constraint_fit": 50,
        "evidence": [{"claim": "source", "quotation": "Evidence one: verbatim text.", "page": 1}],
    }]}
    fake = _OpenAI(payload)
    model = OpenAIResponsesClient("key", client=fake)

    result = asyncio.run(_direct_batch_assess(model, RunRequest(research_question="test question"), [paper]))

    assert result[0].evidence[0].verified is True


def test_live_baseline_rejects_missing_assessment_from_batch_response(tmp_path):
    papers = []
    for index in range(6):
        pdf = tmp_path / f"paper-{index}.pdf"
        pdf.write_bytes(b"pdf")
        papers.append(Paper(arxiv_id=f"2401.0000{index + 1}", title=f"Paper {index}", pdf_path=str(pdf)))
    fake = _OpenAI({"assessments": [{"arxiv_id": papers[0].arxiv_id, "topic_relevance": 1,
                                      "methodological_fit": 1, "evidence_usefulness": 1,
                                      "constraint_fit": 1}]})
    model = OpenAIResponsesClient("key", client=fake)
    with pytest.raises(EvaluationError, match="exactly one assessment"):
        asyncio.run(_direct_batch_assess(model, RunRequest(research_question="test question"), papers))
    assert len(fake.responses.calls) == 1
    assert len(fake.files.deleted) == 6


def test_live_without_key_fails_before_network_or_output(tmp_path):
    manifest = load_manifest(MANIFEST)
    with pytest.raises(EvaluationError, match="OPENAI_API_KEY"):
        asyncio.run(run_evaluation(manifest, tmp_path, demo=False, api_key=None))
    assert not list(tmp_path.iterdir())


def test_labeled_manifest_is_reused_without_judging(tmp_path):
    manifest = load_manifest(MANIFEST)
    manifest["labels"] = {case["id"]: {paper_id: 2 for paper_id in case["arxiv_ids"]}
                          for case in manifest["cases"]}
    result, _, _ = asyncio.run(run_evaluation(manifest, tmp_path, demo=True))
    assert result["labels_source"] == "manifest"
    assert set(result["labels"]["case-01"].values()) == {2.0}


def test_cost_is_reported_only_when_both_rates_are_explicit(tmp_path):
    manifest = load_manifest(MANIFEST)
    result, _, _ = asyncio.run(run_evaluation(manifest, tmp_path / "priced", demo=True,
                                               input_cost_per_1k=1.0, output_cost_per_1k=2.0))
    assert result["cost_rates"] == {"input_per_1k": 1.0, "output_per_1k": 2.0}
    assert "estimated_cost_usd" in result["cases"][0]["baseline"]
    free_result, _, _ = asyncio.run(run_evaluation(manifest, tmp_path / "unpriced", demo=True,
                                                   input_cost_per_1k=1.0))
    assert "cost_rates" not in free_result


def test_cli_live_without_key_returns_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PAPERROUTE_API_KEY", raising=False)
    monkeypatch.delenv("PAPERROUTE_OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", str(MANIFEST), "--live",
                                       "--output-dir", str(tmp_path)])
    assert main() == 3
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_evaluation_help_mentions_compatible_base_url():
    completed = subprocess.run([sys.executable, "scripts/evaluate.py", "--help"],
                               capture_output=True, text=True, check=True)

    assert "requires a hosted API" in completed.stdout
    assert "configured compatible base URL" in completed.stdout


def test_cli_demo_live_reports_paths_and_supports_offline_output(tmp_path, monkeypatch, capsys):
    for name in ("OPENAI_API_KEY", "PAPERROUTE_API_KEY", "PAPERROUTE_OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", str(MANIFEST), "--demo-live",
                                       "--output-dir", str(tmp_path)])
    assert main() == 0
    output = capsys.readouterr().out
    assert "evaluation complete:" in output
    assert (tmp_path / "latest-evaluation.json").exists()

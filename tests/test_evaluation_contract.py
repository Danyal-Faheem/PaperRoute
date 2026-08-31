from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate import _merge_labels, load_manifest

from paperroute import evaluation_runner
from paperroute.arxiv_client import ArxivClient
from paperroute.evaluation_runner import _hydrate, _judge_all, _run_case
from paperroute.models import Paper, RunRequest
from paperroute.openai_client import OpenAIResponsesClient


def test_judge_controls_and_agreement_diagnostics() -> None:
    class Judge:
        custom_compatible = True
        judge_model = "local-judge"

        def __init__(self):
            self.calls = []
            self.answers = iter([2, 2, 1, 0, 2])

        async def judge_relevance(self, request, paper, *, temperature, seed):
            self.calls.append((temperature, seed))
            return next(self.answers)

    model = Judge()
    case = {"id": "case", "question": "attention models", "arxiv_ids": ["a", "b"]}
    papers = {"case": {key: Paper(arxiv_id=key, title=key, abstract="") for key in ("a", "b")}}
    diagnostics = {}
    labels = asyncio.run(_judge_all([case], model, False, papers, diagnostics))
    assert labels == {"case": {"a": 2.0, "b": 2.0}}
    assert model.calls == [(0.1, 42), (0.1, 43), (0.1, 42), (0.1, 43), (0.1, 44)]
    assert diagnostics["paired_judgments"] == 2
    assert diagnostics["agreements"] == 1
    assert diagnostics["disagreements"] == diagnostics["tie_breaks"] == 1
    assert diagnostics["agreement_rate"] == 0.5


def test_frozen_label_provenance_survives_merge(tmp_path: Path) -> None:
    manifest_path = Path("data/benchmark.json")
    manifest = load_manifest(manifest_path)
    labeled = json.loads(manifest_path.read_text())
    labeled["labels"] = {case["id"]: {paper_id: 2 for paper_id in case["arxiv_ids"]}
                          for case in labeled["cases"]}
    labeled["judge"]["diagnostics"] = {"available": True, "agreement_rate": 0.9}
    path = tmp_path / "labeled.json"
    path.write_text(json.dumps(labeled))
    merged = _merge_labels(manifest, path)
    assert merged["judge"]["diagnostics"]["agreement_rate"] == 0.9


def test_get_papers_batches_and_maps_latest_version() -> None:
    atom = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>' \
           'http://arxiv.org/abs/2401.00001v2</id><title>X</title><summary>Y</summary></entry></feed>'

    class Transport:
        def __init__(self):
            self.urls = []

        async def get(self, url, **kwargs):
            self.urls.append(url)
            return SimpleNamespace(text=atom, raise_for_status=lambda: None)

    async def run():
        transport = Transport()
        result = await ArxivClient(transport=transport).get_papers(["2401.00001"])
        return result, transport.urls

    result, urls = asyncio.run(run())
    assert list(result) == ["2401.00001"]
    assert result["2401.00001"].arxiv_id == "2401.00001"
    assert len(urls) == 1


def test_hydrate_uses_one_batched_metadata_query() -> None:
    paper = Paper(arxiv_id="a", title="A", abstract="A")

    class Batched:
        def __init__(self, root: Path):
            self.root, self.calls = root, 0

        async def get_papers(self, ids):
            self.calls += 1
            return {"a": paper}

        async def download_pdf(self, value):
            path = self.root / "a.pdf"
            path.write_bytes(b"pdf")
            return path

    async def run(root: Path):
        client = Batched(root)
        hydrated = await _hydrate({"arxiv_ids": ["a"]}, client, 0)
        return client.calls, hydrated

    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        calls, (papers, failures, _) = asyncio.run(run(Path(directory)))
    assert calls == 1
    assert len(papers) == 1 and not failures


def test_direct_baseline_allocates_full_budget_per_paper(tmp_path: Path, monkeypatch) -> None:
    papers = [Paper(arxiv_id=f"p{i}", title=f"P{i}", abstract="", pdf_path=str(tmp_path / f"{i}.pdf")) for i in range(6)]
    for paper in papers:
        Path(paper.pdf_path).write_bytes(b"pdf")
    payload = {"assessments": [{"arxiv_id": paper.arxiv_id} for paper in papers]}

    class Completions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    completions = Completions()
    client = OpenAIResponsesClient("local", client=SimpleNamespace(
        chat=SimpleNamespace(completions=completions), responses=SimpleNamespace()),
        base_url="http://local/v1", pdf_text_limit=24)
    monkeypatch.setattr(evaluation_runner, "extract_pages", lambda path: ["x" * 100])
    result = asyncio.run(evaluation_runner._direct_batch_assess(client, RunRequest(research_question="question"), papers))
    assert len(result) == 6
    content = completions.calls[0]["messages"][1]["content"]
    for paper in papers:
        block = content.split(f"--- PAPER {paper.arxiv_id} ---\n", 1)[1]
        body = block.split("\n--- PDF TEXT TRUNCATED ---", 1)[0]
        assert body == ("--- PAGE 1 ---\n" + "x" * 100)[:24]
        assert "--- PDF TEXT TRUNCATED ---" in block


def test_run_case_isolates_nonzero_baseline_and_solver_usage(tmp_path: Path) -> None:
    papers = [Paper(arxiv_id=f"p{i}", title=f"P{i}", abstract="", pdf_path=str(tmp_path / f"{i}.pdf")) for i in range(6)]
    for paper in papers:
        Path(paper.pdf_path).write_bytes(b"pdf")
    payloads = {
        "BatchAssessmentOutput": {"assessments": [{"arxiv_id": paper.arxiv_id} for paper in papers]},
        "QueryOutput": {"query": "question"},
        "ScreenOutput": {"arxiv_ids": [paper.arxiv_id for paper in papers]},
        "AssessmentOutput": {"topic_relevance": 10, "methodological_fit": 10,
                              "evidence_usefulness": 10, "constraint_fit": 10},
    }

    class Completions:
        async def create(self, **kwargs):
            schema_name = kwargs["response_format"]["json_schema"]["name"]
            usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(payloads[schema_name])))], usage=usage)

    async def run():
        model = OpenAIResponsesClient("local", client=SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()), responses=SimpleNamespace()),
            base_url="http://local/v1")
        store = evaluation_runner.RunStore(tmp_path / "runs.db")
        case = {"id": "case", "question": "question", "arxiv_ids": [paper.arxiv_id for paper in papers]}
        return await _run_case(case, papers, {paper.arxiv_id: 1 for paper in papers}, model, store, 0, None)

    result = asyncio.run(run())
    assert result["baseline"]["input_tokens"] == 11
    assert result["baseline"]["output_tokens"] == 7
    assert result["solver"]["input_tokens"] == 8 * 11
    assert result["solver"]["output_tokens"] == 8 * 7

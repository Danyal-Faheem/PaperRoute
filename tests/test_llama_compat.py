from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import evaluate as evaluate_script

from paperroute import evaluation_runner
from paperroute.arxiv_client import _demo_pdf_bytes
from paperroute.models import Paper, RunRequest
from paperroute.openai_client import BatchAssessmentOutput, OpenAIResponsesClient
from paperroute.settings import Settings

MANIFEST = Path(__file__).parents[1] / "data" / "benchmark.json"


class _ChatCompletions:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.payload))
        usage = SimpleNamespace(prompt_tokens=17, completion_tokens=5, total_tokens=22)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _ChatClient:
    def __init__(self, payload: dict):
        self.chat = SimpleNamespace(completions=_ChatCompletions(payload))
        self.responses = SimpleNamespace()
        self.files = SimpleNamespace()


def _assessment_payload(papers: list[Paper]) -> dict:
    return {
        "assessments": [
            {
                "arxiv_id": paper.arxiv_id,
                "topic_relevance": 80,
                "methodological_fit": 70,
                "evidence_usefulness": 60,
                "constraint_fit": 50,
                "summary": "structured result",
            }
            for paper in papers
        ]
    }


@pytest.mark.asyncio
async def test_compatible_chat_uses_strict_schema_and_chat_usage_without_files():
    fake = _ChatClient({"query": "attention translation"})
    client = OpenAIResponsesClient("local", client=fake, base_url="http://llama/v1")

    result = await client.plan_query(RunRequest(research_question="attention translation"))

    assert result == "attention translation"
    call = fake.chat.completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == "QueryOutput"
    assert client.usage.input_tokens == 17
    assert client.usage.output_tokens == 5
    assert client.usage.total_tokens == 22
    assert not hasattr(fake.files, "create")
    assert not hasattr(fake.responses, "create")


@pytest.mark.asyncio
async def test_compatible_schema_requires_assessment_and_nested_evidence_fields():
    fake = _ChatClient({"query": "attention translation"})
    client = OpenAIResponsesClient("local", client=fake, base_url="http://llama/v1")

    await client._json("schema", "payload", schema=BatchAssessmentOutput)

    schema = fake.chat.completions.calls[0]["response_format"]["json_schema"]["schema"]
    assert set(schema["required"]) == {"assessments"}
    batch = schema["$defs"]["BatchAssessmentItem"]
    assert set(batch["required"]) >= {
        "arxiv_id", "topic_relevance", "methodological_fit", "evidence_usefulness",
        "constraint_fit", "summary", "limitations", "evidence",
    }
    evidence = schema["$defs"]["Evidence"]
    assert set(evidence["required"]) == {
        "claim", "quotation", "page", "verified", "verification_note",
    }
    assert schema["additionalProperties"] is False
    assert batch["additionalProperties"] is False
    assert evidence["additionalProperties"] is False


@pytest.mark.asyncio
async def test_compatible_schema_strips_max_length_but_keeps_item_bounds():
    fake = _ChatClient({"query": "attention translation"})
    client = OpenAIResponsesClient("local", client=fake, base_url="http://llama/v1")

    await client._json("schema", "payload", schema=BatchAssessmentOutput)

    schema = fake.chat.completions.calls[0]["response_format"]["json_schema"]["schema"]

    def walk(value):
        if isinstance(value, dict):
            assert "maxLength" not in value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    list(walk(schema))
    assessment = schema["$defs"]["BatchAssessmentItem"]["properties"]
    assert assessment["limitations"]["maxItems"] == 3
    assert assessment["evidence"]["maxItems"] == 2


@pytest.mark.asyncio
async def test_compatible_assess_extracts_page_markers_and_truncates_locally(tmp_path):
    paper = Paper(arxiv_id="2401.12345", title="Attention", abstract="A long abstract")
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(_demo_pdf_bytes(paper.model_copy(update={"abstract": "evidence " * 80})))
    fake = _ChatClient({
        "topic_relevance": 80,
        "methodological_fit": 70,
        "evidence_usefulness": 60,
        "constraint_fit": 50,
        "summary": "structured result",
    })
    client = OpenAIResponsesClient("local", client=fake, base_url="http://llama/v1", pdf_text_limit=24)

    result = await client.assess(RunRequest(research_question="attention translation"), paper, pdf)

    assert result.paper.arxiv_id == paper.arxiv_id
    content = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert "topic_relevance" in content
    assert "methodological_fit" in content
    assert "evidence_usefulness" in content
    assert "constraint_fit" in content
    assert "integer scores from 0 to 100" in content
    assert "--- PAGE 1 ---" in content
    assert "--- PDF TEXT TRUNCATED ---" in content
    assert not hasattr(fake.files, "create")


@pytest.mark.asyncio
async def test_compatible_assess_retry_feedback_reaches_chat_with_distinct_controls():
    paper = Paper(arxiv_id="2401.12345", title="Attention", abstract="A long abstract")
    fake = _ChatClient({
        "topic_relevance": 80,
        "methodological_fit": 70,
        "evidence_usefulness": 60,
        "constraint_fit": 50,
        "summary": "structured result",
    })
    client = OpenAIResponsesClient("local", client=fake, base_url="http://llama/v1")

    await client.assess(
        RunRequest(research_question="attention translation"), paper, None,
        feedback="page 7: 'stitched ... quote' — Quotation not found in extracted page text.",
        temperature=0.1, seed=43,
    )

    call = fake.chat.completions.calls[0]
    assert "page 7" in call["messages"][1]["content"]
    assert "stitched ... quote" in call["messages"][1]["content"]
    assert call["temperature"] == 0.1
    assert call["seed"] == 43


@pytest.mark.asyncio
async def test_compatible_chat_rejects_schema_invalid_payload():
    fake = _ChatClient({"score": 9})
    client = OpenAIResponsesClient("local", client=fake, base_url="http://llama/v1")

    with pytest.raises(ValueError, match="invalid structured JSON"):
        await client.judge_relevance(RunRequest(research_question="attention translation"),
                                     Paper(arxiv_id="2401.12345", title="Attention"))


@pytest.mark.asyncio
async def test_compatible_direct_batch_gives_each_pdf_an_equal_text_slice(tmp_path):
    papers = []
    for index in range(6):
        paper = Paper(arxiv_id=f"2401.0000{index + 1}", title=f"Paper {index}", abstract="evidence " * 80)
        path = tmp_path / f"paper-{index}.pdf"
        path.write_bytes(_demo_pdf_bytes(paper))
        papers.append(paper.model_copy(update={"pdf_path": str(path)}))
    fake = _ChatClient(_assessment_payload(papers))
    client = OpenAIResponsesClient("local", client=fake, base_url="http://llama/v1", pdf_text_limit=60)

    result = await evaluation_runner._direct_batch_assess(
        client, RunRequest(research_question="attention translation"), papers
    )

    assert len(result) == 6
    content = fake.chat.completions.calls[0]["messages"][1]["content"]
    assert "integer scores from 0 to 100" in content
    for field in ("topic_relevance", "methodological_fit", "evidence_usefulness", "constraint_fit"):
        assert field in content
    for paper in papers:
        marker = f"--- PAPER {paper.arxiv_id} ---"
        assert marker in content
    assert content.count("--- PDF TEXT TRUNCATED ---") == 6


def test_settings_accept_compatible_base_url_without_api_key(monkeypatch):
    for name in ("PAPERROUTE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PAPERROUTE_OPENAI_BASE_URL", "http://llama/v1")

    settings = Settings()

    assert settings.openai_api_key is None
    assert settings.openai_base_url == "http://llama/v1"
    assert settings.configured is True


def test_evaluation_accepts_compatible_base_url_before_hydration(monkeypatch, tmp_path):
    manifest = {"version": "1", "cases": [{"id": "case-01", "question": "question", "arxiv_ids": []}]}
    seen = {}

    def fake_client(api_key, **kwargs):
        seen.update(api_key=api_key, **kwargs)
        return object()

    async def stop_before_network(*args, **kwargs):
        raise RuntimeError("hydration stopped")

    monkeypatch.setattr(evaluation_runner, "OpenAIResponsesClient", fake_client)
    monkeypatch.setattr(evaluation_runner, "_hydrate", stop_before_network)
    monkeypatch.delenv("PAPERROUTE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAPERROUTE_OPENAI_BASE_URL", "http://llama/v1")

    with pytest.raises(RuntimeError, match="hydration stopped"):
        asyncio.run(evaluation_runner.run_evaluation(manifest, tmp_path, base_url=None, api_key=None))
    assert seen["api_key"] is None
    assert seen["base_url"] == "http://llama/v1"


def test_evaluation_cli_forwards_local_endpoint_and_all_model_env(monkeypatch, tmp_path):
    for name in (
        "PAPERROUTE_API_KEY",
        "OPENAI_API_KEY",
        "PAPERROUTE_OPENAI_BASE_URL",
        "PAPERROUTE_SOLVER_MODEL",
        "PAPERROUTE_JUDGE_MODEL",
        "PAPERROUTE_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PAPERROUTE_API_KEY", "local")
    monkeypatch.setenv("PAPERROUTE_OPENAI_BASE_URL", "http://llama/v1")
    monkeypatch.setenv("PAPERROUTE_SOLVER_MODEL", "local-solver")
    monkeypatch.setenv("PAPERROUTE_JUDGE_MODEL", "local-judge")
    monkeypatch.setenv("PAPERROUTE_TIMEOUT", "300")
    captured = {}

    async def fake_run(manifest, output_dir, **kwargs):
        captured.update(kwargs)
        return {"metrics": {}}, output_dir / "evaluation.json", output_dir / "latest-evaluation.json"

    monkeypatch.setattr(evaluate_script, "run_evaluation", fake_run)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", str(MANIFEST), "--live",
                                       "--output-dir", str(tmp_path)])

    assert evaluate_script.main() == 0
    assert captured["api_key"] == "local"
    assert captured["base_url"] == "http://llama/v1"

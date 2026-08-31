from __future__ import annotations

import json

import pytest

from paperroute.models import Paper, RunRequest
from paperroute.openai_client import OpenAIResponsesClient


class UsageStub:
    input_tokens = 11
    output_tokens = 7


class ResponseStub:
    def __init__(self, payload: object, with_usage: bool = True, output_text: bool = True):
        self.output_text = json.dumps(payload) if output_text else ""
        self.output = []
        if with_usage:
            self.usage = UsageStub()


class ResponsesStub:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FilesStub:
    def __init__(self, file_id="file-test", delete_error=False):
        self.file_id = file_id
        self.delete_error = delete_error
        self.created = []
        self.deleted = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return type("File", (), {"id": self.file_id})()

    async def delete(self, file_id):
        self.deleted.append(file_id)
        if self.delete_error:
            raise RuntimeError("cleanup failed")


class ClientStub:
    def __init__(self, responses, files=None):
        self.responses = ResponsesStub(responses)
        self.files = files or FilesStub()


@pytest.fixture
def run_request():
    return RunRequest(research_question="attention translation")


@pytest.fixture
def paper():
    return Paper(arxiv_id="2401.00001", title="Attention", abstract="Translation")


@pytest.mark.asyncio
async def test_plan_screen_and_judge_filter_outputs_track_usage_and_score(run_request, paper):
    fake = ClientStub([
        ResponseStub({"query": "attention translation"}),
        ResponseStub({"arxiv_ids": [paper.arxiv_id, "not-allowed"]}),
        ResponseStub({"score": 3}),
    ])
    client = OpenAIResponsesClient("key", client=fake)
    assert await client.plan_query(run_request) == "attention translation"
    assert await client.screen(run_request, [paper]) == [paper.arxiv_id]
    assert await client.judge_relevance(run_request, paper) == 3
    assert client.usage.input_tokens == 33
    assert client.usage.output_tokens == 21
    assert fake.responses.calls[-1]["model"] == "gpt-5.6"


@pytest.mark.asyncio
async def test_plan_falls_back_to_question_and_rank_is_passthrough(run_request, paper):
    fake = ClientStub([ResponseStub({"query": ""}, with_usage=False)])
    client = OpenAIResponsesClient("key", client=fake)
    assert await client.plan_query(run_request) == run_request.research_question
    assert await client.rank(run_request, []) == []


@pytest.mark.asyncio
async def test_assess_uploads_pdf_closes_handle_and_deletes_remote_file(run_request, paper, tmp_path):
    files = FilesStub()
    fake = ClientStub([ResponseStub({"topic_relevance": 80, "methodological_fit": 70,
                                     "evidence_usefulness": 60, "constraint_fit": 50})], files)
    client = OpenAIResponsesClient("key", client=fake)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    result = await client.assess(run_request, paper, pdf)
    assert result.paper.arxiv_id == paper.arxiv_id
    assert files.deleted == ["file-test"]
    assert files.created[0]["file"].closed is True
    assert fake.responses.calls[0]["input"][1]["content"][2]["type"] == "input_file"


@pytest.mark.asyncio
async def test_assess_without_pdf_and_cleanup_failure_still_returns_model_result(run_request, paper):
    files = FilesStub(delete_error=True)
    fake = ClientStub([ResponseStub({"topic_relevance": 1})], files)
    client = OpenAIResponsesClient("key", client=fake)
    result = await client.assess(run_request, paper, None)
    assert result.topic_relevance == 1
    assert all(item["type"] == "input_text" for item in fake.responses.calls[0]["input"][1]["content"])


@pytest.mark.asyncio
async def test_invalid_json_and_missing_key_are_explicit_errors(run_request):
    invalid = ClientStub([ResponseStub({"ignored": True})])
    invalid.responses.responses = iter([type("Bad", (), {"output_text": "no json", "output": []})()])
    with pytest.raises(ValueError, match="invalid structured JSON"):
        await OpenAIResponsesClient("key", client=invalid).plan_query(run_request)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await OpenAIResponsesClient(None).plan_query(run_request)

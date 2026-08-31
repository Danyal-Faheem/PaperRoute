import pytest

from paperroute.db import RunStore
from paperroute.models import Evidence, Paper, PaperAssessment, RunStatus
from paperroute.workflow import Workflow


class ModelStub:
    async def plan_query(self, request):
        return "attention"

    async def screen(self, request, papers):
        return [paper.arxiv_id for paper in papers]

    async def assess(self, request, paper, pdf_path):
        return PaperAssessment(paper=paper, topic_relevance=90, methodological_fit=90,
                               evidence_usefulness=90, constraint_fit=90, evidence=[
                                   Evidence(claim="a", quotation="verified", page=1, verified=False),
                                   Evidence(claim="b", quotation="also verified", page=1, verified=False),
                               ])

    async def rank(self, request, assessments):
        return assessments


class ArxivStub:
    def __init__(self, tmp_path):
        self.paper = Paper(arxiv_id="2401.12345", title="Attention paper", abstract="attention")
        self.path = tmp_path / "paper.pdf"
        self.path.write_bytes(b"not a real pdf")

    async def get_paper(self, arxiv_id):
        return self.paper if arxiv_id == self.paper.arxiv_id else None

    async def search(self, query, max_results=20, categories=None):
        return [self.paper]

    async def download_pdf(self, paper):
        return self.path


@pytest.mark.asyncio
async def test_workflow_completes_and_keeps_unverified_quotes_as_skim(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite3")
    request = __import__("paperroute.models", fromlist=["RunRequest"]).RunRequest(
        research_question="attention in translation", pinned_arxiv_ids=["2401.12345"]
    )
    run = __import__("paperroute.models", fromlist=["Run"]).Run(id="run-1", request=request)
    store.create(run)
    result = await Workflow(store, ArxivStub(tmp_path), ModelStub()).execute(run)
    assert result.status == RunStatus.completed
    assert len(result.assessments) == 1
    assert result.assessments[0].decision.value == "Skim"
    assert result.retry_count == 1
    assert any(event.event == "evidence_retry" and event.retry == 1 and "2401.12345" in event.input_summary
               for event in result.trajectories)
    assert any(event.role == "query_planner" for event in result.trajectories)


@pytest.mark.asyncio
async def test_workflow_retry_persists_actionable_feedback_and_supports_legacy_controls(tmp_path):
    class FeedbackModel(ModelStub):
        def __init__(self):
            self.assess_calls = []

        async def assess(self, request, paper, pdf_path, *, feedback=None, temperature=None, seed=None):
            self.assess_calls.append({"feedback": feedback, "temperature": temperature, "seed": seed})
            return await super().assess(request, paper, pdf_path)

    model = FeedbackModel()
    store = RunStore(tmp_path / "runs.sqlite3")
    from paperroute.models import Run, RunRequest

    run = Run(id="run-feedback", request=RunRequest(research_question="attention in translation"))
    store.create(run)
    result = await Workflow(store, ArxivStub(tmp_path), model).execute(run)

    assert result.status == RunStatus.completed
    assert model.assess_calls[0] == {"feedback": None, "temperature": None, "seed": None}
    assert model.assess_calls[1]["temperature"] == 0.1
    assert model.assess_calls[1]["seed"] == 43
    feedback = model.assess_calls[1]["feedback"]
    assert feedback and "page 1" in feedback and "Quotation not found" in feedback
    event = next(event for event in result.trajectories if event.event == "evidence_retry")
    assert event.tool_response["rejected_count"] == 2
    assert event.tool_response["feedback"] == feedback
    assert "fresh contiguous" in event.input_summary


@pytest.mark.asyncio
async def test_workflow_retries_transient_planner_failure(tmp_path):
    class Flaky(ModelStub):
        calls = 0

        async def plan_query(self, request):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("transient")
            return "attention"

    store = RunStore(tmp_path / "runs.sqlite3")
    from paperroute.models import Run, RunRequest

    run = Run(id="run-2", request=RunRequest(research_question="attention in translation"))
    store.create(run)
    result = await Workflow(store, ArxivStub(tmp_path), Flaky()).execute(run)
    assert result.status == RunStatus.completed


@pytest.mark.asyncio
async def test_workflow_keeps_partial_assessment_when_one_pdf_analysis_fails(tmp_path):
    class PartialModel(ModelStub):
        async def assess(self, request, paper, pdf_path):
            raise RuntimeError("model unavailable")

    store = RunStore(tmp_path / "runs.sqlite3")
    from paperroute.models import Run, RunRequest

    run = Run(id="run-partial", request=RunRequest(research_question="attention in translation"))
    store.create(run)
    result = await Workflow(store, ArxivStub(tmp_path), PartialModel(), retries=1).execute(run)
    assert result.status == RunStatus.completed
    assert result.assessments[0].abstract_only is True
    assert result.assessments[0].error == "model unavailable"

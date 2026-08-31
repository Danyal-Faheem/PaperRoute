import json

import pytest

from paperroute.demo import DemoModelClient
from paperroute.export import report_json, report_markdown
from paperroute.models import Evidence, Paper, PaperAssessment, Run, RunRequest


@pytest.mark.asyncio
async def test_demo_client_is_offline_and_deterministic(tmp_path):
    request = RunRequest(research_question="attention translation")
    relevant = Paper(arxiv_id="2401.00001", title="Attention", abstract="translation")
    unrelated = Paper(arxiv_id="2401.00002", title="Biology", abstract="cells")
    model = DemoModelClient()
    assert await model.plan_query(request) == request.research_question
    assert await model.screen(request, [unrelated, relevant]) == [relevant.arxiv_id, unrelated.arxiv_id]
    no_pdf = await model.assess(request, relevant, None)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"pdf")
    with_pdf = await model.assess(request, relevant, pdf)
    assert no_pdf.model_dump() == (await model.assess(request, relevant, None)).model_dump()
    assert no_pdf.evidence_usefulness == 0
    assert with_pdf.evidence_usefulness == 80
    assert await model.rank(request, [no_pdf]) == [no_pdf]


def test_exports_include_empty_and_populated_report_fields(paper):
    empty = Run(id="empty", request=RunRequest(research_question="test question"))
    assert "No assessments" in report_markdown(empty)
    assert json.loads(report_json(empty))["id"] == "empty"

    populated = Run(id="full", request=empty.request, assessments=[PaperAssessment(
        paper=paper, total_score=80, summary="Useful paper", limitations=["small sample"],
        evidence=[Evidence(claim="result", quotation="A result", page=2, verified=True)],
    )])
    markdown = report_markdown(populated)
    assert "Useful paper" in markdown
    assert "small sample" in markdown
    assert "p. 2 (verified)" in markdown
    assert json.loads(report_json(populated))["assessments"][0]["paper"]["arxiv_id"] == paper.arxiv_id

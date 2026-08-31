import pytest

from paperroute.baseline import baseline_assess
from paperroute.models import Paper, RunRequest


@pytest.mark.parametrize("papers", [
    [Paper(arxiv_id="2401.00001", title="Attention methods", abstract="translation")],
    [Paper(arxiv_id="2401.00001", title="Unrelated topic", abstract="biology")],
])
def test_baseline_is_deterministic_and_returns_sorted_assessments(papers):
    request = RunRequest(research_question="attention translation")
    first = baseline_assess(request, papers)
    second = baseline_assess(request, papers)
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert len(first) == len(papers)

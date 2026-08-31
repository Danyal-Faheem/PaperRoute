import pytest

from paperroute.benchmark import (
    BenchmarkCase,
    evaluate_baseline,
    evaluate_ranking,
    judge_labels,
    ndcg_at_k,
)
from paperroute.models import Paper, RunRequest


def make_case() -> BenchmarkCase:
    papers = [Paper(arxiv_id=f"2401.0000{i}", title=title, abstract=abstract)
              for i, (title, abstract) in enumerate([
                  ("Attention translation", "attention neural translation"),
                  ("Graph methods", "graph learning"),
                  ("Unrelated biology", "cell biology"),
              ], 1)]
    return BenchmarkCase("small", RunRequest(research_question="attention translation"), papers,
                         {paper.arxiv_id: score for paper, score in zip(papers, [3, 1, 0], strict=True)})


def test_ndcg_and_evaluate_handle_cutoff_and_empty_relevance():
    relevance = {"a": 3, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], relevance, 2) == pytest.approx(1.0)
    assert ndcg_at_k([], {}, 5) == 0.0
    assert evaluate_ranking(["a", "b"], relevance, 2) == {"ndcg@2": 1.0}


def test_baseline_evaluation_returns_metric():
    result = evaluate_baseline(make_case())
    assert "ndcg@5" in result
    assert 0 <= result["ndcg@5"] <= 1


@pytest.mark.asyncio
async def test_judge_labels_uses_tie_breaker_only_on_disagreement():
    case = make_case()
    answers = iter([3, 3, 1, 2, 2, 0, 0])
    calls = 0

    async def judge(request, paper):
        nonlocal calls
        calls += 1
        return next(answers)

    labels = await judge_labels(case, judge)
    assert labels == {"2401.00001": 3.0, "2401.00002": 2.0, "2401.00003": 0.0}
    assert calls == 7

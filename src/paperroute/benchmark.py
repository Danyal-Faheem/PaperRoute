from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .baseline import baseline_assess
from .models import Paper, RunRequest


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    request: RunRequest
    papers: list[Paper]
    relevance: dict[str, float]


def ndcg_at_k(ranking: list[str], relevance: dict[str, float], k: int = 5) -> float:
    selected = ranking[:k]
    dcg = sum((2 ** relevance.get(pid, 0) - 1) / math.log2(i + 2) for i, pid in enumerate(selected))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** score - 1) / math.log2(i + 2) for i, score in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(ranking: list[str], relevance: dict[str, float], k: int = 5) -> dict[str, float]:
    return {f"ndcg@{k}": round(ndcg_at_k(ranking, relevance, k), 4)}


def evaluate_baseline(case: BenchmarkCase) -> dict[str, float]:
    return evaluate_ranking([a.paper.arxiv_id for a in baseline_assess(case.request, case.papers)], case.relevance)


async def judge_labels(case: BenchmarkCase, judge: Callable[[RunRequest, Paper], Awaitable[int]]) -> dict[str, float]:
    """Create frozen-style graded labels with two passes and a tie-breaker.

    The caller persists the returned labels and must not regenerate them while
    comparing solver and baseline rankings.
    """
    labels: dict[str, float] = {}
    for paper in case.papers:
        first, second = await judge(case.request, paper), await judge(case.request, paper)
        labels[paper.arxiv_id] = float(first if first == second else await judge(case.request, paper))
    return labels

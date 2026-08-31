from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from .models import Paper, PaperAssessment, RunRequest
from .openai_client import ModelClient
from .scoring import rank_assessments, score_assessment


def baseline_assess(request: RunRequest, papers: list[Paper]) -> list[PaperAssessment]:
    """Transparent direct-prompt proxy used as a stable benchmark baseline.

    It uses title/abstract lexical overlap and has no PDF evidence, intentionally
    representing a reasonable low-complexity approach.
    """
    terms = {word.casefold() for word in request.research_question.split() if len(word) > 3}
    result = []
    for paper in papers:
        haystack = f"{paper.title} {paper.abstract}".casefold()
        overlap = len({t for t in terms if t in haystack}) / max(len(terms), 1)
        score = int(round(overlap * 100))
        result.append(score_assessment(PaperAssessment(paper=paper, topic_relevance=score,
            methodological_fit=score, evidence_usefulness=0, constraint_fit=score,
            summary="Title/abstract baseline assessment.")))
    return rank_assessments(result)


async def baseline_model_assess(request: RunRequest, papers: list[Paper], model: ModelClient,
                                downloader: Callable[[Paper], Awaitable[Path]] | None = None) -> list[PaperAssessment]:
    """Run the same analyst schema directly over every candidate, with no stages.

    This is the evaluation baseline when a live model is available; the sync
    lexical function above remains useful for offline CI.
    """
    assessments: list[PaperAssessment] = []
    for paper in papers:
        path = await downloader(paper) if downloader else (Path(paper.pdf_path) if paper.pdf_path else None)
        assessment = await model.assess(request, paper, path)
        assessments.append(score_assessment(assessment))
    return rank_assessments(assessments)

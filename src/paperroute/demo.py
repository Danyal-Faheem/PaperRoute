from __future__ import annotations

from pathlib import Path

from .models import Evidence, Paper, PaperAssessment, RunRequest


class DemoModelClient:
    """Offline model implementation for local demos and tests."""
    async def plan_query(self, request: RunRequest) -> str:
        return request.research_question

    async def screen(self, request: RunRequest, papers: list[Paper]) -> list[str]:
        terms = request.research_question.casefold().split()
        ranked = sorted(papers, key=lambda p: -sum(t in (p.title + p.abstract).casefold() for t in terms))
        return [p.arxiv_id for p in ranked[:6]]

    async def assess(self, request: RunRequest, paper: Paper, pdf_path: Path | None) -> PaperAssessment:
        terms = set(request.research_question.casefold().split())
        text = (paper.title + " " + paper.abstract).casefold()
        relevance = min(100, round(100 * sum(t in text for t in terms) / max(len(terms), 1)))
        evidence = [Evidence(claim="Abstract support", quotation=paper.abstract, page=1),
                    Evidence(claim="Reproducibility support", quotation="this fixture supports reproducible triage", page=1)] if pdf_path else []
        return PaperAssessment(paper=paper, topic_relevance=relevance, methodological_fit=relevance,
                               evidence_usefulness=80 if pdf_path else 0, constraint_fit=relevance,
                               evidence=evidence, summary="Offline lexical demo assessment.")

    async def rank(self, request: RunRequest, assessments: list[PaperAssessment]) -> list[PaperAssessment]:
        return assessments

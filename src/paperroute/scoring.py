from __future__ import annotations

from .models import Decision, PaperAssessment

WEIGHTS = {"topic_relevance": 0.40, "methodological_fit": 0.25, "evidence_usefulness": 0.20, "constraint_fit": 0.15}


def score_assessment(assessment: PaperAssessment, excluded: bool = False) -> PaperAssessment:
    assessment.total_score = round(sum(getattr(assessment, key) * weight for key, weight in WEIGHTS.items()), 2)
    if excluded:
        assessment.total_score, assessment.decision = 0, Decision.skip
    elif assessment.total_score >= 75 and sum(e.verified for e in assessment.evidence) >= 2:
        assessment.decision = Decision.read_now
    elif assessment.total_score >= 45:
        assessment.decision = Decision.skim
    else:
        assessment.decision = Decision.skip
    if assessment.decision == Decision.read_now and len([e for e in assessment.evidence if e.verified]) < 2:
        assessment.decision = Decision.skim
    return assessment


def rank_assessments(assessments: list[PaperAssessment]) -> list[PaperAssessment]:
    return sorted(assessments, key=lambda a: (-a.total_score, a.paper.arxiv_id))


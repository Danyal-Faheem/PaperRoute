from paperroute.models import Decision, Evidence, PaperAssessment
from paperroute.scoring import rank_assessments, score_assessment


def test_weighted_score_and_read_now_require_two_verified_quotes(assessment):
    score_assessment(assessment)
    assert assessment.total_score == 86.0
    assert assessment.decision == Decision.skim

    assessment.evidence = [
        Evidence(claim="method", quotation="first", page=1, verified=True),
        Evidence(claim="result", quotation="second", page=2, verified=True),
    ]
    score_assessment(assessment)
    assert assessment.decision == Decision.read_now


def test_exclusion_forces_skip_even_when_score_is_high(assessment):
    assessment.evidence = [
        Evidence(claim="a", quotation="a", page=1, verified=True),
        Evidence(claim="b", quotation="b", page=2, verified=True),
    ]
    score_assessment(assessment, excluded=True)
    assert assessment.total_score == 0
    assert assessment.decision == Decision.skip


def test_rank_is_descending_and_tie_breaks_by_arxiv_id(paper):
    low = PaperAssessment(paper=paper, total_score=40)
    high = PaperAssessment(paper=paper.model_copy(update={"arxiv_id": "2401.00001"}), total_score=80)
    assert [item.total_score for item in rank_assessments([low, high])] == [80, 40]

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from paperroute.models import Paper, PaperAssessment, Run, RunRequest


@pytest.fixture
def paper() -> Paper:
    return Paper(arxiv_id="2401.12345", title="Attention for graduate research", abstract="A useful abstract.")


@pytest.fixture
def run_request() -> RunRequest:
    return RunRequest(research_question="attention for graduate research", inclusion_criteria=["attention"])


@pytest.fixture
def run(run_request: RunRequest) -> Run:
    return Run(id="test-run", request=run_request, created_at=datetime.now(UTC))


@pytest.fixture
def assessment(paper: Paper) -> PaperAssessment:
    return PaperAssessment(paper=paper, topic_relevance=90, methodological_fit=80,
                           evidence_usefulness=90, constraint_fit=80)

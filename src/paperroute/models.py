from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def now_utc() -> datetime:
    return datetime.now(UTC)


class RunStatus(StrEnum):
    queued = "queued"
    planning = "planning"
    searching = "searching"
    screening = "screening"
    analyzing = "analyzing"
    ranking = "ranking"
    completed = "completed"
    failed = "failed"


class Decision(StrEnum):
    read_now = "Read now"
    skim = "Skim"
    skip = "Skip"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claim: str = Field(max_length=300)
    quotation: str = Field(min_length=1, max_length=1200)
    page: int = Field(ge=1)
    verified: bool = False
    verification_note: str | None = None


class Paper(BaseModel):
    model_config = ConfigDict(extra="ignore")
    arxiv_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: datetime | None = None
    updated: datetime | None = None
    pdf_url: str = ""
    abs_url: str = ""
    pdf_path: str | None = None


class PaperAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    paper: Paper
    topic_relevance: int = Field(default=0, ge=0, le=100)
    methodological_fit: int = Field(default=0, ge=0, le=100)
    evidence_usefulness: int = Field(default=0, ge=0, le=100)
    constraint_fit: int = Field(default=0, ge=0, le=100)
    total_score: float = Field(default=0, ge=0, le=100)
    decision: Decision = Decision.skip
    summary: str = Field(default="", max_length=2000)
    # Persisted runs may predate the current output contract, so historical
    # collections remain readable. Generation caps live in AssessmentOutput.
    limitations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    abstract_only: bool = False
    error: str | None = None


class RunRequest(BaseModel):
    research_question: str = Field(min_length=5, max_length=2000)
    inclusion_criteria: list[str] = Field(default_factory=list, max_length=20)
    exclusion_criteria: list[str] = Field(default_factory=list, max_length=20)
    categories: list[str] = Field(default_factory=list, max_length=20)
    date_from: datetime | None = None
    date_to: datetime | None = None
    pinned_arxiv_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("research_question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("research_question cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_date_range(self) -> RunRequest:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be earlier than or equal to date_to")
        return self


class TrajectoryEvent(BaseModel):
    timestamp: datetime = Field(default_factory=now_utc)
    stage: str
    role: str
    event: str
    input_summary: str = ""
    output_summary: str = ""
    tool: str | None = None
    tool_response: Any = None
    retry: int = 0
    human_checkpoint: bool = False


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None
    latency_ms: int = 0


class Run(BaseModel):
    id: str
    request: RunRequest
    status: RunStatus = RunStatus.queued
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)
    candidates: list[Paper] = Field(default_factory=list)
    assessments: list[PaperAssessment] = Field(default_factory=list)
    trajectories: list[TrajectoryEvent] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    retry_count: int = Field(default=0, ge=0)
    error: str | None = None


class AgentOutput(BaseModel):
    """Strict-ish envelope used when decoding structured model responses."""
    model_config = ConfigDict(extra="ignore")
    payload: dict[str, Any] = Field(default_factory=dict)

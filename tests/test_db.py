import json
import sqlite3

from paperroute.db import RunStore
from paperroute.models import (
    Evidence,
    Paper,
    PaperAssessment,
    Run,
    RunRequest,
    RunStatus,
    TrajectoryEvent,
)


def test_run_store_crud_status_trajectory_and_recent(tmp_path):
    store = RunStore(tmp_path / "nested" / "runs.sqlite3")
    missing = "missing"
    assert store.get(missing) is None
    assert store.set_status(missing, RunStatus.failed, "no run") is None
    assert store.append_trajectory(missing, TrajectoryEvent(stage="x", role="y", event="z")) is None

    first = store.create(Run(id="first", request=RunRequest(research_question="first question")))
    second = store.create(Run(id="second", request=RunRequest(research_question="second question")))
    assert store.get(first.id).request.research_question == "first question"
    assert store.set_status(first.id, RunStatus.completed).status == RunStatus.completed
    updated = store.append_trajectory(first.id, TrajectoryEvent(stage="ranking", role="ranker", event="done"))
    assert updated.trajectories[0].event == "done"
    assert store.list_recent(1)[0].id == second.id
    assert len(store.list_recent(10)) == 2


def test_run_store_loads_legacy_assessment_collections(tmp_path):
    store = RunStore(tmp_path / "legacy.sqlite3")
    paper = Paper(arxiv_id="2401.12345", title="Legacy paper")
    assessment = PaperAssessment(
        paper=paper,
        evidence=[Evidence(claim="claim", quotation="quote", page=1)],
        limitations=["one"],
    )
    run = Run(id="legacy", request=RunRequest(research_question="legacy research"),
              assessments=[assessment])
    payload = run.model_dump(mode="json")
    payload["assessments"][0]["evidence"] = [
        {"claim": f"claim {index}", "quotation": f"quote {index}", "page": 1}
        for index in range(3)
    ]
    payload["assessments"][0]["limitations"] = [f"limitation {index}" for index in range(4)]
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            (run.id, run.status.value, run.created_at.isoformat(), run.updated_at.isoformat(), json.dumps(payload)),
        )
        connection.commit()

    loaded = store.get("legacy")

    assert loaded is not None
    assert len(loaded.assessments[0].evidence) == 3
    assert len(loaded.assessments[0].limitations) == 4

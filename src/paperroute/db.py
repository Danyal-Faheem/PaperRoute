from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import Run, RunStatus, TrajectoryEvent, now_utc


class RunStore:
    """Small SQLite repository. A whole Run is stored as JSON for schema agility."""

    def __init__(self, path: str | Path = "data/paperroute.sqlite3") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, payload TEXT NOT NULL)""")
            conn.commit()

    def create(self, run: Run) -> Run:
        with self._connect() as conn:
            conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                         (run.id, run.status.value, run.created_at.isoformat(),
                          run.updated_at.isoformat(), run.model_dump_json()))
            conn.commit()
        return run

    def get(self, run_id: str) -> Run | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
        return Run.model_validate_json(row["payload"]) if row else None

    def update(self, run: Run) -> Run:
        run.updated_at = now_utc()
        with self._connect() as conn:
            conn.execute("UPDATE runs SET status=?, updated_at=?, payload=? WHERE id=?",
                         (run.status.value, run.updated_at.isoformat(), run.model_dump_json(), run.id))
            conn.commit()
        return run

    def set_status(self, run_id: str, status: RunStatus, error: str | None = None) -> Run | None:
        run = self.get(run_id)
        if not run:
            return None
        run.status = status
        run.error = error
        return self.update(run)

    def append_trajectory(self, run_id: str, event: TrajectoryEvent) -> Run | None:
        run = self.get(run_id)
        if not run:
            return None
        run.trajectories.append(event)
        return self.update(run)

    def list_recent(self, limit: int = 50) -> list[Run]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [Run.model_validate_json(row["payload"]) for row in rows]

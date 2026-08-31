from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings; environment variables are read only at construction."""

    database_path: Path = field(default_factory=lambda: Path(os.getenv("PAPERROUTE_DATABASE_URL", os.getenv("PAPERROUTE_DATABASE", "data/paperroute.sqlite3"))))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("PAPERROUTE_CACHE_DIR", "data/papers")))
    solver_model: str = field(default_factory=lambda: os.getenv("PAPERROUTE_SOLVER_MODEL", "gpt-5.6-terra"))
    judge_model: str = field(default_factory=lambda: os.getenv("PAPERROUTE_JUDGE_MODEL", "gpt-5.6"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("PAPERROUTE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    openai_base_url: str | None = field(default_factory=lambda: os.getenv("PAPERROUTE_OPENAI_BASE_URL") or None)
    arxiv_base_url: str = field(default_factory=lambda: os.getenv("PAPERROUTE_ARXIV_URL", "https://export.arxiv.org/api/query"))
    max_candidates: int = field(default_factory=lambda: _int("PAPERROUTE_MAX_CANDIDATES", 20))
    shortlist_size: int = field(default_factory=lambda: _int("PAPERROUTE_SHORTLIST_SIZE", 6))
    concurrency: int = field(default_factory=lambda: _int("PAPERROUTE_CONCURRENCY", 3))
    retry_count: int = field(default_factory=lambda: _int("PAPERROUTE_RETRIES", 2))
    request_timeout: float = field(default_factory=lambda: float(os.getenv("PAPERROUTE_TIMEOUT", "45")))
    demo_mode: bool = field(default_factory=lambda: os.getenv("PAPERROUTE_DEMO_MODE", "0").lower() in {"1", "true", "yes"})
    evaluation_path: Path = field(default_factory=lambda: Path(os.getenv("PAPERROUTE_EVALUATION_PATH", "data/runtime/latest-evaluation.json")))
    pdf_text_limit: int = field(default_factory=lambda: _int("PAPERROUTE_PDF_TEXT_LIMIT", 60000))

    @property
    def configured(self) -> bool:
        key_ok = bool(self.openai_api_key and self.openai_api_key.strip().lower() not in {"replace-me", "changeme"})
        return bool(self.openai_base_url and self.openai_base_url.strip()) or key_ok

    @property
    def configuration_warning(self) -> str | None:
        if self.demo_mode or self.configured:
            return None
        return "Live runs require OPENAI_API_KEY/PAPERROUTE_API_KEY or PAPERROUTE_OPENAI_BASE_URL. Set one in .env or enable PAPERROUTE_DEMO_MODE=true."

    def prepare_dirs(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()

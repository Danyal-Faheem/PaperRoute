from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _compose_config(**overrides: str) -> dict:
    if not shutil.which("docker"):
        pytest.skip("Docker is not installed")
    environment = os.environ.copy()
    for name in ("PAPERROUTE_DEMO_MODE", "PAPERROUTE_EVALUATION_PATH", "PAPERROUTE_OPENAI_BASE_URL",
                 "PAPERROUTE_API_KEY", "PAPERROUTE_SOLVER_MODEL", "PAPERROUTE_JUDGE_MODEL",
                 "PAPERROUTE_TIMEOUT", "PAPERROUTE_PDF_TEXT_LIMIT"):
        environment.pop(name, None)
    environment.update(overrides)
    completed = subprocess.run(["docker", "compose", "config", "--format", "json"], cwd=ROOT,
                               env=environment, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def test_compose_defaults_to_live_mode_and_persistent_evaluation_path():
    service = _compose_config()["services"]["paperroute"]
    assert service["environment"]["PAPERROUTE_DEMO_MODE"] == "false"
    assert service["environment"]["PAPERROUTE_EVALUATION_PATH"] == "data/runtime/latest-evaluation.json"
    assert service["read_only"] is True
    assert {volume["target"] for volume in service["volumes"]} == {"/app/data/runtime", "/app/data/papers"}


def test_compose_forwards_demo_mode_and_custom_evaluation_path():
    service = _compose_config(PAPERROUTE_DEMO_MODE="true", PAPERROUTE_EVALUATION_PATH="data/runtime/demo.json")["services"]["paperroute"]
    assert service["environment"]["PAPERROUTE_DEMO_MODE"] == "true"
    assert service["environment"]["PAPERROUTE_EVALUATION_PATH"] == "data/runtime/demo.json"


def test_compose_forwards_compatible_endpoint_models_and_timeout():
    environment = _compose_config(PAPERROUTE_OPENAI_BASE_URL="http://llama/v1", PAPERROUTE_API_KEY="local",
                                  PAPERROUTE_SOLVER_MODEL="local-solver", PAPERROUTE_JUDGE_MODEL="local-judge",
                                  PAPERROUTE_TIMEOUT="300")["services"]["paperroute"]["environment"]
    assert environment["PAPERROUTE_OPENAI_BASE_URL"] == "http://llama/v1"
    assert environment["PAPERROUTE_API_KEY"] == "local"
    assert environment["PAPERROUTE_SOLVER_MODEL"] == "local-solver"
    assert environment["PAPERROUTE_JUDGE_MODEL"] == "local-judge"
    assert environment["PAPERROUTE_TIMEOUT"] == "300"


def test_dockerfile_uses_pinned_uv_and_locked_runtime_install():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "ghcr.io/astral-sh/uv:0.12.3" in dockerfile
    assert "COPY pyproject.toml uv.lock README.md ./" in dockerfile
    assert "uv sync --locked --no-dev" in dockerfile


def test_dockerignore_excludes_secrets_and_generated_runtime_data():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert ".env.*" in dockerignore
    assert "!.env.example" in dockerignore
    for generated in ("data/runtime", "data/papers", "graphify-out", "tmp"):
        assert generated in dockerignore

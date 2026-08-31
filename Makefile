.DEFAULT_GOAL := help

PYTHON ?= python3.12
UV ?= uv

.PHONY: help setup dev test lint eval docker-build docker-run

help:
	@printf '%s\n' 'setup        Install application and development dependencies' 'dev          Start the local FastAPI server' 'test         Run deterministic tests with coverage' 'lint         Run Ruff checks' 'eval         Run the offline benchmark evaluator' 'docker-build Build the reproducible image' 'docker-run   Start the container and persistent database'

setup:
	$(UV) sync --extra dev

dev:
	$(UV) run uvicorn paperroute.main:app --reload --host 127.0.0.1 --port 8000 --env-file .env

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .

eval:
	$(UV) run python scripts/evaluate.py --manifest data/benchmark.json

docker-build:
	docker compose build

docker-run:
	docker compose up

# PaperRoute

PaperRoute is a literature-triage assistant for graduate students. It turns a
research question and inclusion/exclusion criteria into a short, ranked
reading list so researchers can spend limited reading time on the most useful
papers first. Every recommendation includes page-verified evidence,
limitations, and the underlying score breakdown; the researcher remains the
final decision-maker.

## Features

- Searches public arXiv metadata and supports pinned paper IDs, categories,
  and date ranges.
- Uses a staged planner → screener → concurrent PDF analysts → deterministic
  ranker workflow.
- Labels papers **Read now**, **Skim**, or **Skip**, with evidence-health,
  errors, retries, timing, token usage, and Markdown/JSON exports.
- Runs offline with deterministic fixtures, or against a hosted OpenAI API or
  a reachable llama.cpp-compatible `/v1/chat/completions` server.
- Provides a benchmark page comparing the staged workflow with a direct
  six-paper baseline.

## Architecture

The FastAPI application serves Jinja templates and local static assets. A
SQLite store persists runs and execution history. arXiv access, model calls, PDF
page extraction, evidence verification, scoring, and evaluation are separated
into small modules under [`src/paperroute`](src/paperroute/). The web entry
point is [`app.py`](src/paperroute/app.py); the workflow is in
[`workflow.py`](src/paperroute/workflow.py).

## Local setup and run

Requires Python 3.12 and uv 0.12.3 (or a compatible recent uv):

```sh
uv sync --locked --extra dev
cp .env.example .env
make test
make lint
make dev                         # http://127.0.0.1:8000
```

For a hosted provider, set `PAPERROUTE_API_KEY` in `.env`. For a local
OpenAI-compatible or llama.cpp server, set:

```sh
PAPERROUTE_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
PAPERROUTE_API_KEY=local
PAPERROUTE_SOLVER_MODEL=your-solver-model
PAPERROUTE_JUDGE_MODEL=your-judge-model
PAPERROUTE_TIMEOUT=300
```

The compatible path uses local page-marked PDF extraction and does not require
the Files API or tool calls. See the [reproduction guide](docs/reproduction.md)
for provider setup, expected output, and Docker.

## Demo and evaluation

The deterministic demo makes no network or model calls:

```sh
make eval
uv run python scripts/evaluate.py --manifest data/benchmark.json --demo-live \
  --output-dir data/runtime
PAPERROUTE_EVALUATION_PATH=data/runtime/latest-evaluation.json make dev
```

This writes the summary used by `/benchmark`. A live run evaluates both the
staged solution and the direct baseline; its full report contains baseline and
solver sections. See the [evaluation method](docs/evaluation.md),
[publication summary](docs/evaluation-results.json), frozen
[benchmark](data/benchmark.json), and [labels](data/benchmark-labeled-1.0.json).
The complete test suite is under [`tests`](tests/).

## Docker, configuration, and security

`PAPERROUTE_DEMO_MODE=true docker compose up --build` builds the locked,
non-root container and serves it on port 8000. Runtime settings include the
database, cache, evaluation path, provider endpoint/key, solver and judge
models, timeout, and PDF text limit; see [`.env.example`](.env.example).

Never commit `.env`, API keys, private PDFs, personal data, or unsanitized
runtime output. Paper text and model output are untrusted input. PaperRoute
currently supports arXiv and is a triage aid, not systematic-review ground
truth or an autonomous consequential system.

## License

PaperRoute is released under the [MIT License](LICENSE). Papers, model weights,
and provider services remain subject to their own terms.

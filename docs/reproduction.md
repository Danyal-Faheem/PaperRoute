# Clean-environment reproduction guide

PaperRoute is tested with Python 3.12 (the recorded environment used Python
3.12.13) and `uv 0.12.3` in the locked Docker image. The lockfile requires
`==3.12.*` and pins the main runtime packages to aiosqlite 0.22.1, FastAPI
0.141.1, httpx 0.28.1, Jinja2 3.1.6, OpenAI 1.109.1, Pydantic 2.13.5, pypdf
5.9.0, and Uvicorn 0.52.4. Dev tools are pytest 8.4.2, pytest-asyncio 0.26.0,
pytest-cov 5.0.0, and Ruff 0.16.5.

## Setup and application run

From a clean checkout, install `uv 0.12.3`, then run:

```sh
uv sync --locked --extra dev
cp .env.example .env
make test
make lint
make dev                 # http://127.0.0.1:8000
```

The required public fixture is `data/benchmark.json`: ten questions with six
pinned public arXiv IDs each. No private data or credentials are needed for
the offline path. The test suite took about 5 seconds in the recorded
environment; the deterministic ten-case demo evaluator took about 1 second
after dependencies were installed.

## Solution

The web application is the staged solution (planner → screener → analysts →
ranker). Start it with `make dev` after setup and open
`http://127.0.0.1:8000`. Expected output is a Uvicorn server listening on
`127.0.0.1:8000` and a form at `/`.

```sh
# Start the local solution (after setup above).
make dev
```

## Baseline / fair comparison

The baseline is one direct structured-output request over the same six PDFs per
case. The combined live evaluator below runs that baseline and the staged
solution with the same manifest and labels; the full report contains the
baseline results under its `baseline` object.

```sh
# Live solution plus one-request baseline comparison (requires a provider).
uv run python scripts/evaluate.py --manifest data/benchmark.json --live \
  --labels data/benchmark-labeled-1.0.json --output-dir data/runtime
```

Expected output is `evaluation complete: ...evaluation-*.json`, a
`latest summary: ...latest-evaluation.json` line, and aggregate baseline/solver
metrics. The command requires a hosted key or configured compatible base URL.

## Evaluation

The frozen manifest validator is offline; the deterministic evaluator exercises
both solution and baseline with local stubs.

```sh
# Validate the frozen manifest only (offline default).
make eval

# Complete deterministic solution/baseline walkthrough (no network or API key).
uv run python scripts/evaluate.py --manifest data/benchmark.json --demo-live \
  --output-dir /tmp/paperroute-evaluation

# Start the Docker demo instead of the local server.
PAPERROUTE_DEMO_MODE=true docker compose up --build
```

`make eval` should print `valid benchmark v1.0: 10 cases`. The Docker command
builds the image with `uv 0.12.3`, installs from `uv.lock`, and serves the demo
on `http://127.0.0.1:8000`.

For the clean-checkout benchmark page, use the generated summary rather than
the absent ignored runtime file:

```sh
uv run python scripts/evaluate.py --manifest data/benchmark.json --demo-live \
  --output-dir data/runtime
PAPERROUTE_EVALUATION_PATH=data/runtime/latest-evaluation.json make dev
```

The demo writes a timestamped `evaluation-*.json` and
`latest-evaluation.json`, with ten cases, six papers per case, and checksums.
Expected deterministic output is a completed 10-case report, a zero exit
status, and both `baseline` and `solver` sections in the versioned report;
live output additionally includes evidence counts, failures, retries, and
model usage. Demo labels and PDF fixtures are synthetic; they are not
model-quality or live evidence claims. The measured
local comparison is summarized in the published [evaluation-results.json
summary](evaluation-results.json), not the runtime artifact validated by
`data/evaluation.schema.json`; that artifact is retained outside the clean
checkout and identified there by SHA-256.

## Local llama.cpp-compatible server

Start a reachable server that implements the OpenAI-compatible
`/v1/chat/completions` interface, then set these values in `.env` (the endpoint
may be on another reachable host):

```sh
PAPERROUTE_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
PAPERROUTE_API_KEY=local
PAPERROUTE_SOLVER_MODEL=your-solver-model
PAPERROUTE_JUDGE_MODEL=your-judge-model
PAPERROUTE_TIMEOUT=300
```

Verify the endpoint with `curl http://127.0.0.1:8000/v1/models` (or the
corresponding configured host), then run the live command above. The compatible
path does not require `/v1/files` or tool calls.

The compatible path uses `chat.completions`, strict JSON Schema, and local
page-marked PDF extraction.
Local inference has no API charge, but electricity/host hardware cost is not
priced. The measured ten-case run took approximately 49 minutes on the local
server. Optional cost fields are reported only when both
input and output rates are supplied to the evaluator.

Docker forwards the endpoint, key, models, timeout, demo mode, and evaluation
path. Its named volumes persist the SQLite database and paper cache. The image
uses the pinned `ghcr.io/astral-sh/uv:0.12.3` builder and `uv sync --locked
--no-dev` against `uv.lock`, so the container and local `uv sync --locked`
paths use the same dependency lock:

```sh
PAPERROUTE_DEMO_MODE=true docker compose up --build
PAPERROUTE_DEMO_MODE=false docker compose up --build
```

Case 02 is the designated challenging case: it asks about factuality and
hallucination in language models while excluding image-only generation. Its
six frozen IDs span factuality, retrieval, and language-model evidence, so
the workflow must distinguish directly useful grounding methods from merely
adjacent LLM papers without changing the pinned inputs. The frozen IDs are
`2303.08774`, `2305.14251`, `2310.06825`, `2309.11495`, `2005.11401`, and
`2202.08906`.

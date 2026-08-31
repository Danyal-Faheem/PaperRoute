# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python/FastAPI literature-triage application. Keep the root directory limited to project-wide configuration and documentation. The current layout is:

- `src/paperroute/` for application code, Jinja templates, and static assets.
- `tests/` for deterministic tests that mirror the boundaries under `src/`.
- `data/` for the frozen benchmark, published labels, and local runtime/cache output.
- `docs/` for evaluation and clean-environment reproduction notes.
- `scripts/` for command-line utilities; `Dockerfile` and `compose.yaml` define the container path.

Group code by feature or domain where practical. Avoid placing generated output, dependencies, editor settings, or secrets under version control; add them to `.gitignore`.

## Build, Test, and Development Commands

Use the repository commands below from the root; they are suitable for CI and do not require interactive prompts:

- `uv sync --locked --extra dev` or `make setup` installs the locked development environment.
- `make dev` starts the local FastAPI server at `http://127.0.0.1:8000`.
- `make test` runs the deterministic pytest suite.
- `make lint` runs Ruff.
- `make eval` validates the ten-case offline benchmark; `uv run python scripts/evaluate.py --manifest data/benchmark.json --demo-live` runs the deterministic evaluator.
- `PAPERROUTE_DEMO_MODE=true docker compose up --build` builds and starts the local demo container.

Commands should run from the repository root and be suitable for CI without interactive prompts.


## Architecture and Engineering

Before building anything, verify if any components can be reused from the existing codebase.
Each architecture and engineering principle should be as simple as possible, we do not want to overengineer for performance. We just need something working for now.
Act as a Principal Software Architect to make sure you use existing components and libraries that already do most of the work instead of reimplementing everything.

## Coding Style & Naming Conventions

Adopt the formatter and linter standard for the chosen language, commit their configuration, and run both before submitting changes. Use spaces rather than tabs unless the language ecosystem strongly dictates otherwise. Choose descriptive names: `snake_case` for Python files and functions, `camelCase` for JavaScript or TypeScript variables, and `PascalCase` for classes and components. Keep modules focused and avoid unrelated refactors in feature changes.

## Testing Guidelines

Add tests with every behavior change or bug fix. Name tests after observable behavior, for example `test_rejects_expired_token.py` or `auth.test.ts`. Keep unit tests fast and deterministic; isolate network and filesystem dependencies with fixtures or mocks. Once tooling is selected, define coverage expectations in the main README and enforce the test command in CI.

## Commit & Pull Request Guidelines

Git history uses short, imperative Conventional Commit subjects (for example, `feat: add request validation` or `fix: handle missing config`). Pull requests should explain the problem and solution, list verification steps, link relevant issues, and include screenshots or logs for user-visible changes. Keep each pull request narrowly scoped and ensure all automated checks pass.

## Security & Configuration

Never commit credentials or local environment files. Provide sanitized examples such as `.env.example`, document every required variable, and validate configuration at startup.

# Coding and Testing

For any coding tasks for implementation, modification or anything such, you (GPT-5.6-Sol) are to act as the orchestrator and the reviewer and the manager while you make GPT-5.6-Luna subagents do all the work for you.

For implementation, try to create and use the local softwares already installed (npm, python) first and then move it to docker containers at the end for reproducibility.

# Reusability

After the implementation is complete, put everything in a docker container such taht it can be easily reproduced by anyone else without fear of security issues.

# Saving intermittent work

After every milestone, you should commit your work to git history. For commit messages, use conventional commits. Such as feat: floobarze the blarginator. Do not do feat(something): floobarize...

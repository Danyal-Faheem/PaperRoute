# Evaluation and Reproduction Contract

## Frozen input

`data/benchmark.json` contains ten representative CS/AI graduate-student questions, six pinned arXiv identifiers per case, inclusion/exclusion criteria, and judge metadata. The evaluator must not replace papers after the manifest is frozen. Store downloaded-PDF SHA-256 checksums and the manifest version with every result.

## Baseline

The baseline is one direct structured-output prompt per case to the solver model, with all six PDF inputs in that single request. It receives the same question, criteria, papers, rubric, and output schema as PaperRoute, but has no planner, staged screening, parallel analyst roles, or evidence-verification retry. The staged solver reuses the same hydrated PDFs and solver model.

## Labels and metrics

For each case, two independent `gpt-5.6` judge passes assign relevance from 0 (not useful) to 3 (directly useful), using only the question, criteria, abstract, and paper. A third pass breaks disagreements. Freeze labels before comparing systems and report judge agreement.

Evaluation artifacts include `judge_diagnostics` with paired-judgment,
agreement, disagreement, tie-break, agreement-rate, model, temperature, and
seed fields. Local-compatible judging uses reproducible low-temperature seeds
`42`, `43`, and `44` for the two passes and tie-break. If a prior labeled
manifest has no diagnostics, the artifact records them as unavailable rather
than inferring agreement.

An earlier NDCG@5 target was not met; it is not the current primary criterion.
The current primary criterion is at least 20% lower mean wall-clock latency than the fair direct
baseline, with solver NDCG@5 no worse than baseline by 0.02, aggregate
`Read now` evidence verification of at least 95%, zero hydration failures, and
zero partial cases, with at least 20 aggregate `Read now` evidence items so
the rate is non-vacuous. Also report verified-evidence rate (numerator/denominator
over quoted evidence in `Read now` items only; zero when no `Read now` evidence
exists), the corresponding counts, coverage, wall-clock seconds, input/output
tokens, estimated cost, failed-paper rate, and partial-run count. Do not omit
failed cases from denominators.

The baseline is verified with the same local page-text matcher as the staged
workflow; baseline and solver results each include read-now verified/total
counts and rates. Retry counts include transient retries and the one proactive
evidence-verification retry per affected paper, with those events retained in
the run history.

```sh
make eval
# deterministic end-to-end evaluator, no credentials or network:
uv run python scripts/evaluate.py --manifest data/benchmark.json --demo-live
# live evaluation with the published frozen labels:
uv run python scripts/evaluate.py --manifest data/benchmark.json --live --labels data/benchmark-labeled-1.0.json
```

For the verified local llama.cpp server, configure the endpoint and model
before running the same command:

```sh
export PAPERROUTE_OPENAI_BASE_URL=http://10.127.78.117:8000/v1
export PAPERROUTE_API_KEY=local
export PAPERROUTE_SOLVER_MODEL='unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL'
export PAPERROUTE_JUDGE_MODEL='unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL'
export PAPERROUTE_TIMEOUT=300
uv run python scripts/evaluate.py --manifest data/benchmark.json --live
```

This compatible path uses strict JSON Schema with `chat.completions`, sends
page-marked PDF text extracted locally, and does not claim full OpenAI API
compatibility: `/v1/files` and tool/function calls are not used.

Live runs write a timestamped `evaluation-*.json` and a compact
`latest-evaluation.json` under ignored `data/runtime/`. The frozen labels are
published at `data/benchmark-labeled-1.0.json` so clean reproduction does not
depend on ignored runtime files. Unresolved pinned papers are retained as
hydration failures/partial-case diagnostics; they are not silently replaced.
Malformed manifests still fail validation. Optional pricing is
explicit: add `--input-cost-per-1k` and `--output-cost-per-1k`; without both,
cost is omitted rather than inferred.

The offline command validates manifest shape and computes no fabricated model
score. Live results belong under ignored `data/runtime/`; publish a sanitized
copy with model IDs, timestamps, checksums, and raw per-case metrics.

## Measured final result

The final local run used the same Qwen model for solver and judge, so judge
agreement is diagnostic rather than independent ground truth. Baseline mean
NDCG@5 was **0.9970** and solver mean NDCG@5 was **0.9957** (delta **-0.0013**).
Mean latency fell from **187.082s** to **106.586s**, a **43.03% reduction**;
tokens fell from **987,578** to **830,575** (**15.90% fewer**). Baseline
Read now evidence was **72/72** verified and solver evidence **80/80**;
hydration failures and partial cases were zero, with three solver evidence
retries. The run took approximately 49 minutes on local hardware, incurred no
API fee, and leaves electricity/hardware cost unpriced. Case 02 is the
challenging factuality/hallucination case: baseline NDCG was 1.0000 versus
solver 0.9829, while both verified 8/8 evidence items.

| Case | Baseline NDCG@5 | Solver NDCG@5 | Baseline latency (s) | Solver latency (s) | Evidence (baseline / solver) |
|---|---:|---:|---:|---:|---:|
| 01 | 1.0000 | 0.9789 | 104.0183 | 93.9125 | 4/4 · 8/8 |
| 02 ★ | 1.0000 | 0.9829 | 196.0928 | 102.0042 | 8/8 · 8/8 |
| 03 | 0.9697 | 0.9947 | 213.2704 | 141.3230 | 10/10 · 10/10 |
| 04 | 1.0000 | 1.0000 | 230.1476 | 149.4053 | 10/10 · 12/12 |
| 05 | 1.0000 | 1.0000 | 182.9754 | 84.8920 | 8/8 · 6/6 |
| 06 | 1.0000 | 1.0000 | 252.9101 | 138.5667 | 10/10 · 12/12 |
| 07 | 1.0000 | 1.0000 | 129.1593 | 45.3111 | 4/4 · 4/4 |
| 08 | 1.0000 | 1.0000 | 223.2838 | 176.4027 | 8/8 · 12/12 |
| 09 | 1.0000 | 1.0000 | 161.0396 | 42.7642 | 4/4 · 4/4 |
| 10 | 1.0000 | 1.0000 | 177.9228 | 91.2786 | 6/6 · 4/4 |

## Current success contract

The revised primary success criterion is at least 20% lower mean wall-clock
latency than the fair direct baseline. Guardrails are solver NDCG@5 no worse
than baseline by 0.02, aggregate read-now evidence verification of at least
95% over at least 20 evidence items, zero hydration failures, and zero partial
cases. The original +0.10 NDCG@5 improvement is recorded as a failed initial
hypothesis; it is not silently rewritten. The final pass/fail arithmetic and
per-case table are in [evaluation-results.json](evaluation-results.json).

## Report contents

Reports should include the full ten-case table, baseline and solver outputs,
aggregate metrics, model/version and protocol/version identifiers,
token/cost totals, retries, and errors. Redact credentials, private paper text,
and personal data from any shared runtime output.

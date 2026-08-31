"""Validate or execute the PaperRoute benchmark.

The default command is deterministic and offline. ``--live`` is explicit and
requires a hosted API key or configured compatible base URL; ``--demo-live``
exercises the complete evaluator with local deterministic adapters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from paperroute.evaluation_runner import EvaluationError, run_evaluation

ARXIV_ID = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("manifest must contain exactly ten cases")
    for case in cases:
        ids = case.get("arxiv_ids")
        if not isinstance(ids, list) or len(ids) != 6:
            raise ValueError(f"{case.get('id', '<unknown>')} must contain six arXiv IDs")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{case.get('id', '<unknown>')} contains duplicate paper IDs")
        for paper_id in ids:
            if not isinstance(paper_id, str) or not ARXIV_ID.fullmatch(paper_id):
                raise ValueError(f"invalid arXiv identifier: {paper_id!r}")
    if payload.get("judge", {}).get("independent_passes") != 2:
        raise ValueError("judge metadata must define two independent passes")
    return payload


def dcg(relevances: list[int], cutoff: int = 5) -> float:
    return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(relevances[:cutoff]))


def ndcg(relevances: list[int], cutoff: int = 5) -> float:
    ideal = dcg(sorted(relevances, reverse=True), cutoff)
    return dcg(relevances, cutoff) / ideal if ideal else 0.0


def _merge_labels(manifest: dict[str, Any], labels_path: Path | None) -> dict[str, Any]:
    if labels_path is None:
        return manifest
    labeled = load_manifest(labels_path)
    if [case["id"] for case in labeled["cases"]] != [case["id"] for case in manifest["cases"]]:
        raise ValueError("labeled manifest cases do not match benchmark manifest")
    labels = labeled.get("labels")
    if labels is None:
        labels = {case["id"]: case.get("labels", case.get("relevance")) for case in labeled["cases"]}
    merged = dict(manifest)
    merged["labels"] = labels
    # Preserve provenance from the frozen artifact so a reuse run can report
    # how labels were generated instead of silently discarding diagnostics.
    if isinstance(labeled.get("judge"), dict):
        merged["judge"] = {**(manifest.get("judge") or {}), **labeled["judge"]}
    if isinstance(labeled.get("judge_diagnostics"), dict):
        merged["judge_diagnostics"] = labeled["judge_diagnostics"]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, help="Optional JSON evaluation result to summarize")
    parser.add_argument("--labels", type=Path, help="Optional frozen labeled manifest to reuse")
    parser.add_argument("--output-dir", type=Path, default=Path("data/runtime"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Run against arXiv and OpenAI")
    mode.add_argument("--demo-live", action="store_true", help="Run full evaluation with deterministic local stubs")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--input-cost-per-1k", type=float)
    parser.add_argument("--output-cost-per-1k", type=float)
    args = parser.parse_args()
    try:
        manifest = _merge_labels(load_manifest(args.manifest), args.labels)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2
    if args.retries < 0:
        print("--retries must be non-negative", file=sys.stderr)
        return 2
    if not args.live and not args.demo_live:
        if args.results:
            try:
                result = json.loads(args.results.read_text(encoding="utf-8"))
                for name in ("baseline", "solver"):
                    values = [case["ndcg_at_5"] for case in result[name]["cases"] if "ndcg_at_5" in case]
                    mean = sum(values) / len(values) if values else 0.0
                    print(f"{name}: mean NDCG@5={mean:.3f} ({len(values)} cases)")
            except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
                print(f"results error: {exc}", file=sys.stderr)
                return 2
        else:
            print(f"valid benchmark v{manifest['version']}: {len(manifest['cases'])} cases")
        return 0
    try:
        result, versioned_path, latest_path = asyncio.run(run_evaluation(
            manifest, args.output_dir, demo=args.demo_live,
            api_key=os.getenv("PAPERROUTE_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("PAPERROUTE_OPENAI_BASE_URL") or None,
            retries=args.retries, input_cost_per_1k=args.input_cost_per_1k,
            output_cost_per_1k=args.output_cost_per_1k))
    except (EvaluationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 3
    print(f"evaluation complete: {versioned_path}")
    print(f"latest summary: {latest_path}")
    print(json.dumps(result["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

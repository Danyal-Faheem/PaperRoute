import json
from pathlib import Path

import pytest
from scripts.evaluate import load_manifest, ndcg

MANIFEST = Path(__file__).parents[1] / "data" / "benchmark.json"


def test_frozen_manifest_has_ten_cases_and_six_unique_valid_ids():
    payload = load_manifest(MANIFEST)
    assert len(payload["cases"]) == 10
    assert all(len(set(case["arxiv_ids"])) == 6 for case in payload["cases"])


def test_ndcg_rewards_ideal_order_and_penalizes_reversed_order():
    assert ndcg([3, 2, 1, 0, 0]) == pytest.approx(1.0)
    assert ndcg([0, 0, 1, 2, 3]) < 1.0


def test_manifest_rejects_wrong_case_count(tmp_path):
    invalid = {"version": "x", "cases": []}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly ten"):
        load_manifest(path)

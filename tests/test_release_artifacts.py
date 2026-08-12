"""Cross-check the current compact result bundle against its summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.make_paper_figures import build_abc_xyz_segmentation


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "current"


def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_release_summary_matches_forecast_and_selection_artifacts():
    summary = _json("release_summary.json")
    index = summary["headline_run"]
    forecast = _json("forecast_metrics.json")
    coverage = _json("interval_coverage.json")
    selection = _json("inventory_selection.json")

    assert summary["software_release"] == "v1.0.0"
    assert index["forecast"]["calibrated_wape_pct"] == pytest.approx(
        forecast["ensemble"]["WAPE"]
    )
    assert index["forecast"]["calibrated_wpe_pct"] == pytest.approx(
        forecast["ensemble"]["WPE"]
    )
    assert index["forecast"]["q10_q90_coverage_pct"] == pytest.approx(
        coverage["overall_interval_coverage"] * 100
    )
    assert coverage["nominal_interval_coverage"] == pytest.approx(0.8)
    assert selection["preselected_policy"] == "Q90 Direct"
    assert selection["newsvendor_critical_ratio"]["value"] == pytest.approx(1.1 / 1.9)
    assert selection["fill_rate_floor_sensitivity"] == {
        "0.85": "Newsvendor-Normal (CR=0.579)",
        "0.90": "Q90 Direct",
        "0.95": "SS (0.75σ)",
    }


def test_selected_policy_and_profit_inference_are_synchronized():
    selection = _json("inventory_selection.json")
    inventory = {row["Policy"]: row for row in _csv_rows("inventory_results.csv")}
    selected = inventory[selection["preselected_policy"]]

    assert float(selected["preselected"]) == 1
    assert float(selected["Avg Profit"]) == pytest.approx(
        selection["test_metrics_reported"]["Avg Profit"], abs=5e-5
    )
    assert float(selected["Fill Rate"]) == pytest.approx(
        selection["test_metrics_reported"]["Fill Rate"], abs=5e-5
    )
    assert "Newsvendor-Normal (CR=0.579)" in inventory
    assert all("CR=0.625" not in name for name in inventory)

    paired = {
        row["comparison"]: row for row in _csv_rows("profit_paired_differences.csv")
    }
    static = paired["selected vs static baseline"]
    index_diff = _json("release_summary.json")["headline_run"]["selection"][
        "paired_profit_difference_vs_static"
    ]
    assert float(static["diff_point"]) == pytest.approx(index_diff["point"], abs=5e-7)
    assert float(static["diff_ci95_lo"]) == pytest.approx(index_diff["ci95"][0], abs=5e-7)
    assert float(static["diff_ci95_hi"]) == pytest.approx(index_diff["ci95"][1], abs=5e-7)


def test_release_documentation_points_to_current_artifacts():
    readme = (ROOT / "README.md").read_text()
    lock = (ROOT / "requirements-lock.txt").read_text()
    assert "output/headline" in readme
    assert "output" + "_v" not in readme
    assert "requirements-lock.txt" in readme
    assert "numpy==" in lock and "lightgbm==" in lock and "pytest==" in lock


def test_public_documentation_is_release_neutral():
    paths = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "docs" / "run_manifest.md",
        ROOT / "results" / "current" / "README.md",
    ]
    paths.extend((ROOT / "results" / "current").glob("*.json"))
    text = "\n".join(path.read_text().lower() for path in paths)
    assert "round" + "1" not in text
    assert "round" + " 1" not in text
    assert "revi" + "sion" not in text


def test_controlled_runtime_artifact_supports_manuscript_claims():
    runtime = _json("recovery_runtime.json")
    assert runtime["n_sp"] == 2000
    assert runtime["boundary"].startswith("recovery/imputation call only")
    assert runtime["timings"]["heuristic_imr"]["median_s"] == pytest.approx(0.295)
    assert runtime["timings"]["tobit_mle"]["median_s"] == pytest.approx(6.2)
    assert runtime["timings"]["saits"]["median_s"] == pytest.approx(23.0)
    assert runtime["timings"]["timesnet"]["median_s"] == pytest.approx(531.1)


def test_abc_xyz_segmentation_is_derived_from_training_sales():
    import pandas as pd

    frame = pd.DataFrame({
        "sp": [1, 1, 2, 2, 3, 3],
        "sale_amount": [40.0, 40.0, 5.0, 10.0, 0.0, 5.0],
    })
    segmented = build_abc_xyz_segmentation(frame)

    assert segmented["sp"].tolist() == [1, 2, 3]
    assert segmented["ABC"].tolist() == ["A", "B", "C"]
    assert set(segmented["XYZ"]).issubset({"X", "Y", "Z"})
    assert segmented["Segment"].str.len().eq(2).all()

    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        segmented,
        build_abc_xyz_segmentation(shuffled),
    )

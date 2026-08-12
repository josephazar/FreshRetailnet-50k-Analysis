"""Guard the tutorial notebooks against stockout and cost-model errors."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> str:
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_stockout_hour_count_is_not_treated_as_availability():
    source = _source("01_data_exploration_and_censored_demand.ipynb")
    assert "Number of out-of-stock hours" in source
    assert "df['stock_hour6_22_cnt'] > 0" in source
    assert "df['stock_hour6_22_cnt'] < 16" not in source
    assert "df['stock_hour6_22_cnt'] == 0" in source


def test_inventory_tutorial_uses_full_objective_critical_ratio():
    source = _source("05_inventory_optimization.ipynb")
    assert 'underage = cfg["unit_revenue"] - cfg["unit_cost"] + cfg["p"]' in source
    assert 'overage = cfg["unit_cost"] + cfg["h"] + cfg["w"]' in source
    assert "critical_ratio = underage / (underage + overage)" in source
    assert "cfg['p'] / (cfg['p'] + cfg['h'] + cfg['w'])" not in source
    assert "CR = 0.625" not in source
    assert 'RESULTS / "inventory_results.csv"' in source


def test_public_forecasting_tutorials_use_current_result_bundle():
    for name in [
        "03_forecasting_models_and_validation.ipynb",
        "04_forecast_evaluation_and_error_analysis.ipynb",
        "05_inventory_optimization.ipynb",
    ]:
        source = _source(name)
        assert 'ROOT / "results" / "current"' in source
        assert "output" + "_v" not in source
        assert "make_features(combined)" not in source


def test_saved_tutorials_do_not_expose_local_paths():
    for name in [
        "03_forecasting_models_and_validation.ipynb",
        "04_forecast_evaluation_and_error_analysis.ipynb",
        "05_inventory_optimization.ipynb",
    ]:
        raw = (ROOT / "notebooks" / name).read_text()
        assert "/Users/" not in raw
        assert "sys.executable" not in raw


def test_interval_tutorial_uses_committed_coverage_artifact():
    source = _source("04_forecast_evaluation_and_error_analysis.ipynb")
    assert 'RESULTS / "interval_coverage.json"' in source
    assert "nominal_interval_coverage" in source

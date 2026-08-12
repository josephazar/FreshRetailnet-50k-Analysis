"""Guard against silent stale-data fallbacks in analysis scripts."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
STATSIG = ROOT / 'scripts' / 'statistical_significance.py'


def test_statsig_default_preds_is_authoritative_run():
    src = STATSIG.read_text()
    assert "default='output/benchmark/benchmark_per_sp_predictions.csv'" in src, (
        "--preds default must point at output/benchmark"
    )
    assert 'fallback path' not in src, (
        "silent fallback-path behavior must stay absent"
    )


def test_statsig_hard_errors_on_missing_preds(tmp_path):
    result = subprocess.run(
        [sys.executable, str(STATSIG),
         '--preds', 'does/not/exist.csv',
         '--out', str(tmp_path / 'statsig_out'),
         '--n_boot', '10'],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0, (
        "script must exit non-zero when the predictions file is missing "
        f"(stdout: {result.stdout!r})"
    )
    out_dir = tmp_path / 'statsig_out'
    written = list(out_dir.glob('*.csv')) if out_dir.exists() else []
    assert not written, f"no output CSVs may be written on failure, got {written}"

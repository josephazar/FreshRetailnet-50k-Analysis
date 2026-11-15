"""Synthetic panel whose stock status varies and carries signal.

Stockouts follow a persistent two-state process and cut that day's sales, so a
model trained on the panel splits on the stockout-history features (so1, sor7,
soh7, ...). A test that must notice evaluation-week stock status reaching the
forecasts needs such a panel: with a constant stock status, as in the other
fixtures, that channel cannot change anything.
"""
import numpy as np
import pandas as pd

EVAL_START = '2024-03-25'


def make_stock_panel(seed: int = 3, n_days: int = 90):
    """Return (train, eval) frames as load_data would, without cens/dem_rec."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    rows = []
    for s in range(1, 4):
        for p in range(1, 4):
            out = False
            for dt in dates:
                out = rng.random() < (0.7 if out else 0.15)
                hours = int(rng.integers(4, 17)) if out else 0
                base = float(rng.poisson(4 + s + p))
                rows.append({
                    'sp': s * 10000 + p,
                    'store_id': s, 'product_id': p, 'city_id': 1,
                    'management_group_id': 1,
                    'first_category_id': 10,
                    'second_category_id': 20 + (p % 2),
                    'third_category_id': 30 + (p % 3),
                    'dt': dt,
                    'sale_amount': base * (1 - hours / 16),
                    'stock_hour6_22_cnt': hours,
                    'discount': 0.1, 'holiday_flag': 0, 'activity_flag': 0,
                    'precpt': 0.5, 'avg_temperature': 15.0,
                    'avg_humidity': 60.0, 'avg_wind_level': 2.0,
                })
    df = pd.DataFrame(rows)
    return df[df['dt'] < EVAL_START].copy(), df[df['dt'] >= EVAL_START].copy()


def flip_stock_status(frame: pd.DataFrame, mask=None) -> pd.DataFrame:
    """Turn in-stock days into full-day stockouts and vice versa (rows in mask)."""
    out = frame.copy()
    mask = np.ones(len(out), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    hours = out['stock_hour6_22_cnt'].to_numpy()
    out.loc[mask, 'stock_hour6_22_cnt'] = np.where(hours[mask] > 0, 0, 16)
    return out

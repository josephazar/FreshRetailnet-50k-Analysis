"""Direct multi-horizon forecasting (horizon-as-feature), leak-free.

Alternative to recursive multi-step inference: a single model per objective is
trained on (origin, horizon) samples and predicts day T+h directly from
information available at origin T. No predictions are fed back into lag
features, so recursive error compounding and write-back shrinkage cannot occur.

Construction
------------
The 117 FeatureBuilder features split exactly into two groups:

  * HISTORY features (68): functions of the sale/dem_rec/censoring history
    through the day before the row's date (lags, rollings, EWMAs, diffs,
    stockout history, derived ratios). For an (origin T, target T+h) sample
    these must reflect information through T only, which equals the feature
    row of day T+1 -- implemented as a per-SP shift(h-1) of the HISTORY block.
  * TARGET-DAY features (49): calendar encodings, exogenous covariates
    (weather, discount, flags), and train-only lookup aggregates keyed by the
    target day's day-of-week. These are known in advance for the target day
    and contain no sales information, so they are taken from the target row.

A sample for horizon h therefore is:
    [HISTORY shifted by (h-1)] + [TARGET-DAY of day t] + [h]  ->  y_t
For h=1 this is exactly the baseline one-step design.

Leakage argument: HISTORY at the first evaluation day references training days
only; TARGET-DAY columns are sales-free; evaluation actuals never enter any
feature. tests/test_direct_forecast.py asserts this by mutation.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import lightgbm as lgb


# Longest lag/rolling window in FeatureBuilder. The first 28 days of the
# training file only seed lag features and are never used as training targets;
# the evaluation path carries exactly this much training history so that
# lag-28/rolling-28 features exist for the first forecast day.
WARMUP_DAYS = 28

HISTORY_FEATURES = [
    # lags
    's_l1', 's_l2', 's_l3', 's_l5', 's_l7', 's_l14', 's_l21', 's_l28',
    'r_l1', 'r_l2', 'r_l3', 'r_l5', 'r_l7', 'r_l14', 'r_l21', 'r_l28',
    # diffs
    's_d1', 's_d7', 'r_d1',
    # rolling stats
    's_m3', 's_m7', 's_m14', 's_m28', 's_sd3', 's_sd7', 's_sd14', 's_sd28',
    's_mx7', 's_mx28', 's_mn7', 's_mn28', 's_md7', 's_md28',
    'r_m3', 'r_m7', 'r_m14', 'r_m28', 'r_sd3', 'r_sd7', 'r_sd14', 'r_sd28',
    'r_mx7', 'r_mx28', 'r_mn7', 'r_mn28', 'r_md7', 'r_md28',
    # rolling quantiles
    's_q25_14', 's_q25_28', 's_q75_14', 's_q75_28',
    # EWMA
    's_ew7', 's_ew14', 'r_ew7', 'r_ew14',
    # stockout history
    'so1', 'so7', 'sor7', 'sor14', 'soh7', 'soh14',
    # derived from history
    'cv7', 'cv28', 'tr_7_28', 'tr_3_14', 'l1_m7', 'm7_m28', 'rec_obs_ratio',
]

TARGETDAY_FEATURES = [
    # calendar
    'dow', 'dom', 'woy', 'month', 'wknd', 'doy',
    'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos', 'woy_sin', 'woy_cos',
    # exogenous covariates of the target day
    'discount', 'holiday_flag', 'activity_flag', 'precpt',
    'avg_temperature', 'avg_humidity', 'avg_wind_level',
    # train-only lookups (static or keyed by target-day dow)
    'sp_m', 'sp_s', 'pd_m', 'pd_s', 'st_m', 'st_s', 'ct_m', 'ct_s',
    'cat1_m', 'cat2_m', 'cat3_m', 'cat3_s',
    'dow_prof', 'dow_prof_r', 'store_daily_vol', 'store_so_rate',
    'cat2_daily_m', 'sp_vs_cat2', 'sp_vs_cat3', 'sp_vs_store',
    'cluster_id', 'cluster_m', 'cluster_s', 'temp_dev', 'hum_dev',
    # cross features of the target day
    'd_h', 'd_a', 't_p', 'w_h', 'h_a',
]

HORIZON_COL = 'h'


def classify_feature_cols(fcols: list) -> tuple[list, list]:
    """Split fcols into (history, target-day); hard-fail on unknown columns
    so a future FeatureBuilder change cannot silently mis-route a feature."""
    hist = [c for c in fcols if c in HISTORY_FEATURES]
    tday = [c for c in fcols if c in TARGETDAY_FEATURES]
    unknown = sorted(set(fcols) - set(hist) - set(tday))
    if unknown:
        raise ValueError(
            f"unclassified feature columns for direct forecasting: {unknown}; "
            f"add them to HISTORY_FEATURES or TARGETDAY_FEATURES"
        )
    return hist, tday


def direct_feature_cols(fcols: list) -> list:
    return list(fcols) + [HORIZON_COL]


def build_direct_table(
    feat: pd.DataFrame,
    fcols: list,
    horizons: int = 7,
    target_col: str = 'sale_amount',
) -> pd.DataFrame:
    """Long-format direct training table.

    For each horizon h, the HISTORY block is shifted (h-1) rows within each SP
    (daily-contiguous panel), so a row for target day t carries information
    through day t-h only. Rows whose shift window precedes the SP's first
    observation are dropped.

    Returns a frame with columns: sp, dt, h, <fcols>, target, cens.
    """
    hist, _tday = classify_feature_cols(fcols)
    feat = feat.sort_values(['sp', 'dt']).reset_index(drop=True)
    parts = []
    g = feat.groupby('sp')
    for h in range(1, horizons + 1):
        part = feat[['sp', 'dt', target_col, 'cens'] + fcols].copy()
        if h > 1:
            # `part` is an index-preserving copy of `feat`, so the groupby
            # built on `feat` aligns row-for-row with `part`.
            part[hist] = g[hist].shift(h - 1)
        part[HORIZON_COL] = np.int8(h)
        if h > 1:
            # Rows whose (h-1)-shift ran past the SP's first observation have
            # NaN history; s_l1 (hist[0]) is non-null on every valid row, so
            # it identifies exactly those rows.
            part = part[part[hist[0]].notna()]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    # Release the per-horizon parts before the astype below materialises its
    # block; together they are the second-largest transient of a full-data
    # run.
    del parts
    gc.collect()
    out[hist] = out[hist].astype('float32')
    return out


def _origin_design(
    feat: pd.DataFrame,
    fcols: list,
    first_forecast_day,
    target_days: list,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Design matrix for predicting target_days from one origin.

    first_forecast_day: pass the FIRST FORECAST DAY (origin + 1), NOT the
    origin itself. That day's lag/rolling features are computed from data
    through the origin day, so reusing its HISTORY block for every horizon
    restricts all predictions to information available at the origin. Both
    callers pass val_days[0] / eval_days[0].

    Returns (X, meta) with meta = sp/dt/h aligned to X rows, sorted (sp, dt).
    """
    hist, tday = classify_feature_cols(fcols)
    origin_rows = feat[feat['dt'] == first_forecast_day][['sp'] + hist]
    target_rows = feat[feat['dt'].isin(target_days)][['sp', 'dt'] + tday]
    day_index = {d: i + 1 for i, d in enumerate(sorted(target_days))}
    merged = target_rows.merge(origin_rows, on='sp', how='left', validate='many_to_one')
    merged[HORIZON_COL] = merged['dt'].map(day_index).astype('int8')
    merged = merged.sort_values(['sp', 'dt']).reset_index(drop=True)
    X = merged[direct_feature_cols(fcols)].fillna(0).values
    return X, merged[['sp', 'dt', HORIZON_COL]]


def train_direct_cv(
    feat: pd.DataFrame,
    fcols: list,
    objectives: dict,
    base_params: dict,
    cv_folds: int = 4,
    val_size: int = 7,
    target_col: str = 'sale_amount',
    num_boost_round: int = 1500,
    early_stopping: int = 50,
):
    """Expanding-window CV for the direct design, mirroring train_cv.

    Validation for a fold with train_end T uses origin T exactly like the
    final evaluation uses the last training day: HISTORY from day T+1's rows
    (information through T), TARGET-DAY columns from each validation day.
    """
    dates = sorted(feat['dt'].unique())
    warmup_date = dates[WARMUP_DAYS - 1]
    available = [d for d in dates if d > warmup_date]
    n_avail = len(available)

    # One training sample per horizon 1..val_size: the model must cover
    # exactly the validation (and final evaluation) window, so the number of
    # trained horizons equals val_size by construction.
    table = build_direct_table(feat, fcols, horizons=val_size, target_col=target_col)
    dcols = direct_feature_cols(fcols)

    best_iters = {name: [] for name in objectives}
    cv_val_frames = []
    cv_metrics = []
    for i in range(cv_folds):
        val_end_idx = n_avail - (cv_folds - 1 - i) * val_size
        val_start_idx = val_end_idx - val_size
        train_end = available[val_start_idx - 1]
        val_days = available[val_start_idx:val_end_idx]

        tr_mask = (table['dt'] > warmup_date) & (table['dt'] <= train_end)
        Xtr = table.loc[tr_mask, dcols].fillna(0).values
        ytr = table.loc[tr_mask, target_col].values
        wtr = np.ones(len(ytr), dtype='float32')
        # Censored days carry partial demand information; the 0.5 down-weight
        # matches pipeline_leakfree.train_cv so direct and recursive arms
        # differ only in design, not in weighting.
        wtr[table.loc[tr_mask, 'cens'].values == 1] = 0.5

        Xvl, vl_meta = _origin_design(feat, fcols, val_days[0], val_days)
        vl_lookup = feat[feat['dt'].isin(val_days)][['sp', 'dt', target_col]]
        vl_meta2 = vl_meta.merge(vl_lookup, on=['sp', 'dt'], how='left')
        yvl = vl_meta2[target_col].values

        print(f"  [direct] fold {i+1}/{cv_folds}: train<= {pd.Timestamp(train_end).date()}, "
              f"Ntr={len(ytr):,} Nvl={len(yvl):,}")
        val_preds = {}
        for name, obj in objectives.items():
            params = {**base_params, **obj}
            d_tr = lgb.Dataset(Xtr, label=ytr, weight=wtr)
            d_vl = lgb.Dataset(Xvl, label=yvl, reference=d_tr)
            mdl = lgb.train(
                params, d_tr, num_boost_round=num_boost_round,
                valid_sets=[d_vl],
                callbacks=[lgb.early_stopping(early_stopping), lgb.log_evaluation(0)],
            )
            best_iters[name].append(mdl.best_iteration)
            val_preds[name] = np.clip(mdl.predict(Xvl), 0, None).astype('float32')
        ens = [n for n in ('mae', 'huber', 'q50') if n in val_preds]
        if ens:
            val_preds['ensemble'] = np.mean([val_preds[n] for n in ens], axis=0).astype('float32')
        cv_val_frames.append({
            'fold': i + 1,
            'sp': vl_meta2['sp'].values,
            'dt': vl_meta2['dt'].values,
            'y': yvl,
            'preds': val_preds,
        })
        ref = val_preds.get('ensemble', next(iter(val_preds.values())))
        cv_metrics.append({'fold': i + 1,
                           'mae': float(np.mean(np.abs(yvl - ref))),
                           'n_train': int(len(ytr)), 'n_val': int(len(yvl))})
        # Free this fold's design before the next fold materialises its own;
        # at full data each Xtr is several GB and holding two doubles the
        # peak.
        del Xtr, ytr, wtr, Xvl, d_tr, d_vl, mdl
        gc.collect()
    return cv_metrics, best_iters, cv_val_frames


def retrain_direct_final(
    feat: pd.DataFrame,
    fcols: list,
    objectives: dict,
    base_params: dict,
    best_iters: dict,
    target_col: str = 'sale_amount',
    horizons: int = 7,
):
    dates = sorted(feat['dt'].unique())
    warmup_date = dates[WARMUP_DAYS - 1]
    table = build_direct_table(feat, fcols, horizons=horizons, target_col=target_col)
    dcols = direct_feature_cols(fcols)
    mask = table['dt'] > warmup_date
    Xtr = table.loc[mask, dcols].fillna(0).values
    ytr = table.loc[mask, target_col].values
    wtr = np.ones(len(ytr), dtype='float32')
    # Same 0.5 censored-row down-weight as pipeline_leakfree (see train_direct_cv).
    wtr[table.loc[mask, 'cens'].values == 1] = 0.5
    # The table is not needed once the design is extracted; at full data it
    # is ~14 GB and would otherwise sit idle through six boosting runs.
    del table
    gc.collect()
    final = {}
    for name, obj in objectives.items():
        params = {**base_params, **obj}
        nround = max(50, int(np.mean(best_iters[name])))
        mdl = lgb.train(params, lgb.Dataset(Xtr, label=ytr, weight=wtr),
                        num_boost_round=nround)
        final[name] = mdl
        print(f"  [direct] final {name}: num_boost_round={nround}")
    return final


def predict_eval_direct(
    fbuilder,
    models: dict,
    fcols: list,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
):
    """Direct prediction of the evaluation window from the last training day.

    Builds the combined feature frame exactly like predict_eval_single_origin
    (evaluation sales zeroed -- irrelevant here because HISTORY columns are
    taken from the first evaluation day, whose lags reference training days
    only, and TARGET-DAY columns contain no sales), then assembles the
    (origin, horizon) design and predicts each model.

    Returns (predictions dict, eval_sorted) aligned to sort (sp, dt), matching
    predict_eval_recursive's contract.
    """
    # Carry exactly the WARMUP_DAYS-day tail of training history so that
    # lag-28/rolling-28 features exist for the first forecast day.
    hist = train_df[train_df['dt'] >= train_df['dt'].max()
                    - pd.Timedelta(days=WARMUP_DAYS - 1)].copy()
    hist['cens'] = (hist['stock_hour6_22_cnt'] > 0).astype('int8')
    if 'dem_rec' not in hist.columns:
        hist['dem_rec'] = hist['sale_amount'].astype('float32')

    ev = eval_df.sort_values(['sp', 'dt']).reset_index(drop=True).copy()
    eval_sorted = ev.copy()
    ev['cens'] = (ev['stock_hour6_22_cnt'] > 0).astype('int8')
    ev['sale_amount'] = 0.0
    ev['dem_rec'] = 0.0

    combined = pd.concat([hist, ev], ignore_index=True)
    feat = fbuilder.transform(combined)
    eval_days = sorted(ev['dt'].unique())
    X, meta = _origin_design(feat, fcols, eval_days[0], eval_days)

    # meta is sorted (sp, dt) == eval_sorted order; verify alignment hard.
    if not (meta['sp'].values == eval_sorted['sp'].values).all() or \
       not (pd.to_datetime(meta['dt']).values == eval_sorted['dt'].values).all():
        raise RuntimeError("direct eval design misaligned with eval_sorted")

    predictions = {
        name: np.clip(mdl.predict(X), 0, None).astype('float32')
        for name, mdl in models.items()
    }
    return predictions, eval_sorted

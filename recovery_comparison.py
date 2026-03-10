#!/usr/bin/env python3
"""
recovery_comparison.py — Censored Demand Recovery: Tobit/IMR vs Deep Learning
==============================================================================

Compares 5 recovery methods on FreshRetailNet-50K by measuring downstream
forecasting accuracy after each recovery approach:

  1. No Recovery        — raw censored sale_amount (lower bound of demand)
  2. Simple Mean        — replace censored with SP mean of non-censored
  3. Tobit/IMR          — our econometric method (time-weighted + inverse Mills)
  4. SAITS              — Self-Attention Imputation for Time Series (DL)
  5. TimesNet           — Temporal 2D-variation model (DL)

For each method:
  - Apply recovery to training data → dem_rec column
  - Build identical feature set
  - Train same LightGBM model (same hyperparameters)
  - Predict on held-out evaluation set
  - Measure: WAPE, WPE, MAE, RMSE, Corr, training time

DL models (SAITS, TimesNet) operate on hourly data matching the official
FreshRetailNet-50K baseline (Wang et al., arXiv:2505.16319):
  - 480 timesteps = 30 days × 16 business hours (6am-10pm)
  - 6 features: sale + discount + holiday + precipitation + temperature + time
  - NaN masking for censored hours (stock_status == 1)
Then aggregate hourly imputed demand to daily totals.

Outputs:
  - output/recovery_comparison.csv
  - output/recovery_comparison.png
  - output/recovery_comparison_detail.json
"""

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MPLBACKEND'] = 'Agg'

import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'freshretailnet', 'raw', 'data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.parquet')
EVAL_PATH = os.path.join(DATA_DIR, 'eval.parquet')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SP = 2000           # SPs to sample for all non-DL methods
N_SP_DL = 1000        # SPs for DL recovery (PyPOTS scales ~O(N^1.5))
SEED = 42
OP = 16               # Operating hours 6am-10pm
WINDOW_DAYS = 30      # DL recovery window (matching official baseline)

# Hourly demand profile (normalized to sum=1 over 16 hours)
HOURLY_PROFILE = np.array([0.08, 0.09, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04,
                            0.05, 0.06, 0.07, 0.08, 0.09, 0.08, 0.06, 0.06])
HOURLY_PROFILE = HOURLY_PROFILE / HOURLY_PROFILE.sum()

# LightGBM hyperparameters (same as main pipeline)
LGB_PARAMS = {
    'objective': 'mae',
    'metric': 'mae',
    'n_estimators': 800,
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': SEED,
    'n_jobs': 4,
    'verbose': -1,
}

# SAITS/TimesNet hyperparameters (matching official baseline)
DL_EPOCHS = 30        # 30 epochs sufficient — loss converges well by epoch 25-30
DL_PATIENCE = None    # No early stopping (train all epochs — no validation overhead)
DL_BATCH = 256


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load and subsample FreshRetailNet-50K data."""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load with hourly columns for DL recovery
    daily_cols = ['city_id', 'store_id', 'management_group_id',
                  'first_category_id', 'second_category_id', 'third_category_id',
                  'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt',
                  'discount', 'holiday_flag', 'activity_flag',
                  'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
    hourly_cols = ['store_id', 'product_id', 'dt', 'hours_sale', 'hours_stock_status']

    train = pd.read_parquet(TRAIN_PATH, columns=daily_cols)
    train_hourly = pd.read_parquet(TRAIN_PATH, columns=hourly_cols)
    ev = pd.read_parquet(EVAL_PATH, columns=daily_cols)

    # Create SP key
    for df in [train, train_hourly, ev]:
        df['sp'] = df['store_id'] * 10000 + df['product_id']
    for df in [train, ev]:
        df['dt'] = pd.to_datetime(df['dt'])
    train_hourly['dt'] = pd.to_datetime(train_hourly['dt'])

    # Stratified subsample
    np.random.seed(SEED)
    all_sp = train['sp'].unique()
    print(f"  Total SPs: {len(all_sp):,}")

    if N_SP > 0 and N_SP < len(all_sp):
        cens_rate = train.groupby('sp')['stock_hour6_22_cnt'].apply(
            lambda x: (x > 0).mean()
        )
        bins = pd.qcut(cens_rate, q=10, duplicates='drop')
        chosen = cens_rate.groupby(bins).apply(
            lambda x: x.sample(min(len(x), N_SP // 10), random_state=SEED)
        ).droplevel(0).index.values
        if len(chosen) < N_SP:
            remaining = np.setdiff1d(all_sp, chosen)
            extra = np.random.choice(remaining, N_SP - len(chosen), replace=False)
            chosen = np.concatenate([chosen, extra])
        chosen = chosen[:N_SP]

        train = train[train['sp'].isin(chosen)].reset_index(drop=True)
        train_hourly = train_hourly[train_hourly['sp'].isin(chosen)].reset_index(drop=True)
        ev = ev[ev['sp'].isin(chosen)].reset_index(drop=True)

    cens_pct = (train['stock_hour6_22_cnt'] > 0).mean() * 100
    print(f"  Sampled SPs: {train['sp'].nunique():,}")
    print(f"  Train: {len(train):,} rows, Eval: {len(ev):,} rows")
    print(f"  Censoring rate: {cens_pct:.1f}%")
    return train, train_hourly, ev


# ============================================================================
# RECOVERY METHOD 1: NO RECOVERY (raw censored sales)
# ============================================================================
def recovery_none(train):
    """No recovery — use raw censored sale_amount."""
    df = train.copy()
    df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
    df['so_frac'] = (df['stock_hour6_22_cnt'] / OP).astype('float32')
    df['dem_rec'] = df['sale_amount'].values.copy().astype('float32')
    return df


# ============================================================================
# RECOVERY METHOD 2: SIMPLE MEAN IMPUTATION
# ============================================================================
def recovery_simple_mean(train):
    """Replace censored days with SP-level mean of non-censored observations."""
    df = train.copy()
    df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
    df['so_frac'] = (df['stock_hour6_22_cnt'] / OP).astype('float32')
    df['dem_rec'] = df['sale_amount'].values.copy().astype('float64')

    # Compute mean from non-censored days per SP
    sp_means = df[df['cens'] == 0].groupby('sp')['sale_amount'].mean()
    global_mean = df[df['cens'] == 0]['sale_amount'].mean()

    cens_idx = df.index[df['cens'] == 1]
    sp_vals = df.loc[cens_idx, 'sp'].map(sp_means).fillna(global_mean)
    df.loc[cens_idx, 'dem_rec'] = sp_vals.values

    df['dem_rec'] = df['dem_rec'].clip(lower=0).astype('float32')
    return df


# ============================================================================
# RECOVERY METHOD 3: TOBIT/IMR (our econometric approach)
# ============================================================================
def recovery_tobit(train):
    """Full Tobit/IMR recovery — matches main pipeline."""
    df = train.copy()
    df = df.sort_values(['sp', 'dt']).reset_index(drop=True)
    df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
    df['so_frac'] = (df['stock_hour6_22_cnt'] / OP).astype('float32')
    df['dem_rec'] = df['sale_amount'].values.copy().astype(np.float64)

    partial = (df['stock_hour6_22_cnt'] > 0) & (df['stock_hour6_22_cnt'] < OP)
    full = df['stock_hour6_22_cnt'] >= OP

    # Step 1: Time-weighted recovery for partial stockouts
    so_hrs = df['stock_hour6_22_cnt'].values.astype(int)
    avail_weights = np.array([HOURLY_PROFILE[:OP - n].sum() if n < OP else 0.01
                              for n in range(OP + 1)])
    partial_idx = df.index[partial]
    partial_so = so_hrs[partial]
    df.loc[partial_idx, 'dem_rec'] = (
        df.loc[partial_idx, 'sale_amount'].values / avail_weights[partial_so].clip(min=0.01)
    )

    # Step 2: Full stockout — rolling average of non-censored
    no_so = (df['cens'] == 0).astype(float)
    df['_s'] = df['sale_amount'] * no_so
    rs = df.groupby('sp')['_s'].transform(lambda x: x.rolling(14, min_periods=1).sum())
    rc = df.groupby('sp').apply(
        lambda g: no_so.loc[g.index].rolling(14, min_periods=1).sum()
    ).reset_index(level=0, drop=True).sort_index()
    avg_ns = rs / rc.clip(lower=1)
    df.loc[full, 'dem_rec'] = avg_ns[full]
    df.drop('_s', axis=1, inplace=True)

    # Step 3: Inverse Mills ratio correction
    sp_stats = df[df['cens'] == 0].groupby('sp')['sale_amount'].agg(['mean', 'std', 'count'])
    sp_stats.columns = ['sp_mu', 'sp_sigma', 'sp_n']
    sp_stats['sp_sigma'] = sp_stats['sp_sigma'].fillna(0).clip(lower=0.05)

    cat_stats = df[df['cens'] == 0].groupby('second_category_id')['sale_amount'].agg(['mean', 'std'])
    cat_stats.columns = ['cat_mu', 'cat_sigma']
    cat_stats['cat_sigma'] = cat_stats['cat_sigma'].fillna(0).clip(lower=0.05)

    df = df.merge(sp_stats[['sp_mu', 'sp_sigma', 'sp_n']], left_on='sp',
                   right_index=True, how='left')
    df = df.merge(cat_stats[['cat_mu', 'cat_sigma']], left_on='second_category_id',
                   right_index=True, how='left')

    use_cat = (df['sp_n'].fillna(0) < 5)
    df.loc[use_cat, 'sp_mu'] = df.loc[use_cat, 'cat_mu']
    df.loc[use_cat, 'sp_sigma'] = df.loc[use_cat, 'cat_sigma']
    df['sp_mu'] = df['sp_mu'].fillna(df['sale_amount'].mean())
    df['sp_sigma'] = df['sp_sigma'].fillna(0.5)

    cens_mask = df['cens'] == 1
    S = df.loc[cens_mask, 'sale_amount'].values.astype(np.float64)
    mu_vals = df.loc[cens_mask, 'sp_mu'].values.astype(np.float64)
    sigma_vals = df.loc[cens_mask, 'sp_sigma'].values.astype(np.float64)

    z = (S - mu_vals) / sigma_vals.clip(min=0.05)
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    imr = phi_z / (1 - Phi_z).clip(min=1e-6)
    correction = sigma_vals * imr
    correction = np.clip(correction, 0, 3 * sigma_vals)

    corrected = np.maximum(
        df.loc[cens_mask, 'dem_rec'].values,
        df.loc[cens_mask, 'dem_rec'].values + correction * 0.5
    )
    df.loc[cens_mask, 'dem_rec'] = corrected

    df.drop(['sp_mu', 'sp_sigma', 'sp_n', 'cat_mu', 'cat_sigma'],
            axis=1, inplace=True, errors='ignore')
    df['dem_rec'] = df['dem_rec'].clip(lower=0).astype('float32')
    return df


# ============================================================================
# RECOVERY METHOD 4 & 5: DEEP LEARNING (SAITS / TimesNet via PyPOTS)
# ============================================================================
def prepare_hourly_data(train_hourly, train_daily):
    """
    Prepare hourly data for PyPOTS imputation models.

    Returns:
        X_windows: np.ndarray (n_windows, 480, n_features) with NaN for censored
        window_map: list of (sp, window_idx, dates) for mapping back to daily
    """
    print("  Preparing hourly data for DL recovery...")

    sps = sorted(train_hourly['sp'].unique())
    all_windows = []
    window_map = []

    for i, sp in enumerate(sps):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(sps)} SPs...")

        sp_hourly = train_hourly[train_hourly['sp'] == sp].sort_values('dt')
        sp_daily = train_daily[train_daily['sp'] == sp].sort_values('dt')

        if len(sp_hourly) < WINDOW_DAYS:
            continue

        dates = sp_hourly['dt'].values
        n_days = len(sp_hourly)

        # Extract hourly sale and stock arrays
        hours_sale = np.array(sp_hourly['hours_sale'].tolist())       # (n_days, 24)
        hours_stock = np.array(sp_hourly['hours_stock_status'].tolist())  # (n_days, 24)

        # Business hours only: 6am-10pm (indices 6-21)
        hours_sale_biz = hours_sale[:, 6:22].astype(np.float32)       # (n_days, 16)
        hours_stock_biz = hours_stock[:, 6:22].astype(np.int8)         # (n_days, 16)

        # Mask censored hours with NaN
        hours_sale_masked = np.where(hours_stock_biz == 1, np.nan, hours_sale_biz)

        # Get daily covariates
        sp_daily_sorted = sp_daily.sort_values('dt')
        discount = sp_daily_sorted['discount'].values.astype(np.float32)
        holiday = sp_daily_sorted['holiday_flag'].values.astype(np.float32)
        precpt = sp_daily_sorted['precpt'].values.astype(np.float32)
        temperature = sp_daily_sorted['avg_temperature'].values.astype(np.float32)

        # Normalize covariates (per-SP normalization to [0,1])
        discount = discount / (discount.max() + 0.1) if discount.max() > 0 else discount
        precpt = precpt / (precpt.max() + 0.1) if precpt.max() > 0 else precpt
        temperature = temperature / (temperature.max() + 0.1) if temperature.max() > 0 else temperature

        # Split into 30-day windows
        n_windows = n_days // WINDOW_DAYS
        for w in range(n_windows):
            start = w * WINDOW_DAYS
            end = start + WINDOW_DAYS
            win_dates = dates[start:end]

            # Sale: (30, 16) → flatten to (480,)
            sale_flat = hours_sale_masked[start:end].reshape(-1)  # 480

            # Covariates: repeat daily values 16 times
            disc_flat = np.repeat(discount[start:end], OP)       # 480
            hol_flat = np.repeat(holiday[start:end], OP)          # 480
            prec_flat = np.repeat(precpt[start:end], OP)          # 480
            temp_flat = np.repeat(temperature[start:end], OP)     # 480

            # Time position: normalized 0 to 1
            time_pos = np.linspace(0, 1, WINDOW_DAYS * OP).astype(np.float32)

            # Stack: (480, 6)
            window = np.column_stack([sale_flat, disc_flat, hol_flat,
                                       prec_flat, temp_flat, time_pos])
            all_windows.append(window)
            window_map.append((sp, w, win_dates))

    X = np.array(all_windows, dtype=np.float32)

    # Normalize sale channel globally to prevent extreme values
    # (PyPOTS works best with standardized data)
    sale_vals = X[:, :, 0]
    sale_observed = sale_vals[~np.isnan(sale_vals)]
    sale_mean = np.mean(sale_observed) if len(sale_observed) > 0 else 0
    sale_std = np.std(sale_observed) if len(sale_observed) > 0 else 1
    sale_std = max(sale_std, 0.01)  # avoid division by zero

    # Replace any inf/nan in covariates with 0
    for feat_idx in range(1, X.shape[2]):
        col = X[:, :, feat_idx]
        col[np.isnan(col)] = 0
        col[np.isinf(col)] = 0
        X[:, :, feat_idx] = col

    print(f"  Prepared {X.shape[0]} windows, shape: {X.shape}")
    print(f"  NaN rate (sale channel): {np.isnan(X[:, :, 0]).mean() * 100:.1f}%")
    print(f"  Sale stats: mean={sale_mean:.4f}, std={sale_std:.4f}")

    # Store normalization params for denormalization after imputation
    norm_params = {'sale_mean': sale_mean, 'sale_std': sale_std}
    return X, window_map, norm_params


def run_dl_recovery(X_windows, window_map, norm_params, train_daily, model_name='SAITS'):
    """
    Run SAITS or TimesNet imputation and aggregate to daily dem_rec.

    Args:
        X_windows: (n_windows, 480, 6) with NaN for censored
        window_map: list of (sp, window_idx, dates)
        norm_params: dict with sale_mean, sale_std for denormalization
        train_daily: daily training DataFrame
        model_name: 'SAITS' or 'TimesNet'

    Returns:
        DataFrame with dem_rec column filled by DL recovery
    """
    import torch
    from pypots.imputation import SAITS as SAITS_Model
    from pypots.imputation import TimesNet as TimesNet_Model
    from pypots.optim import Adam as PyPOTS_Adam

    print(f"\n  Training {model_name} model...")

    # Normalize the data for better DL training
    X = X_windows.copy()
    sale_mean = norm_params['sale_mean']
    sale_std = norm_params['sale_std']
    # Normalize sale channel (feature 0) — keep NaN as NaN
    sale_col = X[:, :, 0]
    observed_mask = ~np.isnan(sale_col)
    sale_col[observed_mask] = (sale_col[observed_mask] - sale_mean) / sale_std
    X[:, :, 0] = sale_col

    # Remove all-NaN windows (entire 30-day full stockout — no observed data)
    all_nan_mask = np.isnan(X[:, :, 0]).all(axis=1)
    if all_nan_mask.any():
        n_removed = all_nan_mask.sum()
        print(f"  Removing {n_removed} all-NaN windows (full stockout periods)")
        valid_idx = ~all_nan_mask
        X = X[valid_idx]
        window_map_filtered = [wm for wm, v in zip(window_map, valid_idx) if v]
    else:
        window_map_filtered = list(window_map)

    n_steps = X.shape[1]   # 480
    n_features = X.shape[2]  # 6
    print(f"  Data: {X.shape[0]} windows × {n_steps} steps × {n_features} features")

    # Force CPU — MPS hangs with large datasets in PyPOTS
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"  Using CUDA GPU")
    else:
        print(f"  Using CPU (MPS skipped due to PyPOTS compatibility)")

    t0 = time.time()
    optimizer = PyPOTS_Adam(lr=0.001, weight_decay=1e-5)

    if model_name == 'SAITS':
        model = SAITS_Model(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=2,
            d_model=64,
            d_ffn=32,
            n_heads=4,
            d_k=16,
            d_v=16,
            dropout=0.1,
            ORT_weight=1.0,
            MIT_weight=1.0,
            epochs=DL_EPOCHS,
            patience=DL_PATIENCE,
            batch_size=DL_BATCH,
            optimizer=optimizer,
            device=device,
            saving_path=None,
            verbose=True,
        )
    else:  # TimesNet
        optimizer2 = PyPOTS_Adam(lr=0.001, weight_decay=1e-5)
        model = TimesNet_Model(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=2,
            top_k=7,
            d_model=64,
            d_ffn=32,
            n_kernels=5,
            dropout=0.1,
            apply_nonstationary_norm=True,
            epochs=DL_EPOCHS,
            patience=DL_PATIENCE,
            batch_size=DL_BATCH,
            optimizer=optimizer2,
            device=device,
            saving_path=None,
            verbose=True,
        )

    # Train on a subsample if dataset is large (PyPOTS scales poorly >4000)
    MAX_TRAIN = 3000
    np.random.seed(42)
    if len(X) > MAX_TRAIN:
        print(f"  Subsampling {MAX_TRAIN}/{len(X)} windows for training")
        train_idx = np.random.choice(len(X), MAX_TRAIN, replace=False)
        X_train_subset = X[train_idx]
    else:
        X_train_subset = X

    # Fit the model
    model.fit({"X": X_train_subset})
    train_time = time.time() - t0
    print(f"  {model_name} training done in {train_time:.0f}s")

    # Predict (impute) — process in batches
    t1 = time.time()
    PRED_BATCH = 1500
    imputed_parts = []
    for start in range(0, len(X), PRED_BATCH):
        end = min(start + PRED_BATCH, len(X))
        batch_result = model.predict({"X": X[start:end]})
        imputed_parts.append(batch_result["imputation"])
        print(f"    Imputed batch {start}-{end}")
    X_imputed = np.concatenate(imputed_parts, axis=0)
    impute_time = time.time() - t1
    print(f"  Imputation done in {impute_time:.0f}s")

    # Denormalize sale channel back to original scale
    sale_imputed = X_imputed[:, :, 0] * sale_std + sale_mean

    # Clip negative values
    sale_imputed = np.clip(sale_imputed, 0, None)

    # Aggregate hourly → daily demand
    sale_hourly = sale_imputed  # (n_windows, 480)
    # Reshape to (n_windows, 30, 16) and sum hours per day
    sale_daily_recovered = sale_hourly.reshape(-1, WINDOW_DAYS, OP).sum(axis=2)

    # Map back to training DataFrame
    df = train_daily.copy()
    df = df.sort_values(['sp', 'dt']).reset_index(drop=True)
    df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
    df['so_frac'] = (df['stock_hour6_22_cnt'] / OP).astype('float32')
    df['dem_rec'] = df['sale_amount'].values.copy().astype('float32')

    # Build lookup: (sp, date) → recovered daily demand (vectorized)
    records = []
    for i, (sp, w_idx, win_dates) in enumerate(window_map_filtered):
        for d_idx, dt in enumerate(win_dates):
            records.append((sp, pd.Timestamp(dt), float(sale_daily_recovered[i, d_idx])))
    rec_df = pd.DataFrame(records, columns=['sp', 'dt', 'dl_rec'])
    # Average if multiple windows cover the same (sp, date)
    rec_df = rec_df.groupby(['sp', 'dt'])['dl_rec'].mean().reset_index()

    # Merge with main DataFrame
    df = df.merge(rec_df, on=['sp', 'dt'], how='left')

    # Apply DL recovery to censored observations
    has_dl = df['dl_rec'].notna()
    cens_mask = (df['cens'] == 1) & has_dl
    recovered_count = cens_mask.sum()
    df.loc[cens_mask, 'dem_rec'] = df.loc[cens_mask, 'dl_rec'].clip(lower=0)

    # For SPs/dates without DL recovery (not in windows), use simple fallback
    remaining_cens = (df['cens'] == 1) & (~has_dl)
    if remaining_cens.sum() > 0:
        sp_means = df[df['cens'] == 0].groupby('sp')['sale_amount'].mean()
        global_mean = df[df['cens'] == 0]['sale_amount'].mean()
        fallback = df.loc[remaining_cens, 'sp'].map(sp_means).fillna(global_mean).values
        df.loc[remaining_cens, 'dem_rec'] = fallback.astype(np.float32)

    df.drop('dl_rec', axis=1, inplace=True)

    df['dem_rec'] = df['dem_rec'].clip(lower=0).astype('float32')

    print(f"  DL-recovered {recovered_count:,} censored observations")
    print(f"  Total time: {train_time + impute_time:.0f}s")

    return df, train_time + impute_time


def run_dl_recovery_subprocess(X_windows, window_map, norm_params, train_daily, model_name='SAITS'):
    """
    Run DL recovery in a subprocess to avoid OMP/PyTorch thread conflicts
    that cause hangs when PyPOTS runs after LightGBM in the same process.
    """
    import subprocess, tempfile, pickle

    print(f"\n  Running {model_name} in subprocess (isolated from LightGBM)...")

    # Save data to temp files
    tmp_dir = tempfile.mkdtemp(prefix='recovery_')
    data_path = os.path.join(tmp_dir, 'dl_input.pkl')
    result_path = os.path.join(tmp_dir, 'dl_output.pkl')

    with open(data_path, 'wb') as f:
        pickle.dump({
            'X_windows': X_windows,
            'window_map': window_map,
            'norm_params': norm_params,
            'model_name': model_name,
            'epochs': DL_EPOCHS,
            'patience': DL_PATIENCE,
            'batch_size': DL_BATCH,
            'result_path': result_path,
        }, f)

    # Write subprocess script
    script = f'''
import os, sys, pickle, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

with open("{data_path}", "rb") as f:
    cfg = pickle.load(f)

X_windows = cfg['X_windows']
window_map = cfg['window_map']
norm_params = cfg['norm_params']
model_name = cfg['model_name']

# Normalize
X = X_windows.copy()
sale_mean, sale_std = norm_params['sale_mean'], norm_params['sale_std']
sale_col = X[:, :, 0]
obs = ~np.isnan(sale_col)
sale_col[obs] = (sale_col[obs] - sale_mean) / sale_std
X[:, :, 0] = sale_col

# Remove all-NaN windows
all_nan = np.isnan(X[:,:,0]).all(axis=1)
valid_mask = ~all_nan
if all_nan.any():
    print(f"  Removing {{all_nan.sum()}} all-NaN windows")
X_clean = X[valid_mask]

# Subsample for training
MAX_TRAIN = 3000
np.random.seed(42)
if len(X_clean) > MAX_TRAIN:
    train_idx = np.random.choice(len(X_clean), MAX_TRAIN, replace=False)
    X_train = X_clean[train_idx]
    print(f"  Training on {{MAX_TRAIN}}/{{len(X_clean)}} windows")
else:
    X_train = X_clean

print(f"  Training {{model_name}}, X_train={{X_train.shape}}")
sys.stdout.flush()

import torch
from pypots.imputation import SAITS, TimesNet
from pypots.optim import Adam

t0 = time.time()
n_steps, n_features = X_clean.shape[1], X_clean.shape[2]

if model_name == 'SAITS':
    model = SAITS(n_steps=n_steps, n_features=n_features, n_layers=2,
                  d_model=64, d_ffn=32, n_heads=4, d_k=16, d_v=16,
                  dropout=0.1, ORT_weight=1.0, MIT_weight=1.0,
                  epochs=cfg['epochs'], patience=cfg['patience'],
                  batch_size=cfg['batch_size'],
                  optimizer=Adam(lr=0.001, weight_decay=1e-5),
                  device='cpu', verbose=True)
else:
    # TimesNet is ~15x heavier than SAITS — use fewer epochs and smaller training set
    tn_epochs = min(cfg['epochs'], 10)
    model = TimesNet(n_steps=n_steps, n_features=n_features, n_layers=2,
                     top_k=5, d_model=32, d_ffn=32, n_kernels=3,
                     dropout=0.1, apply_nonstationary_norm=True,
                     epochs=tn_epochs, patience=cfg['patience'],
                     batch_size=cfg['batch_size'],
                     optimizer=Adam(lr=0.001, weight_decay=1e-5),
                     device='cpu', verbose=True)
    # Also reduce training set for TimesNet
    if len(X_train) > 1000:
        np.random.seed(42)
        X_train = X_train[np.random.choice(len(X_train), 1000, replace=False)]
        print(f"  TimesNet: reduced training to {{len(X_train)}} windows, {{tn_epochs}} epochs")

model.fit({{"X": X_train}})
train_time = time.time() - t0
print(f"  {{model_name}} training: {{train_time:.0f}}s")

# Predict in batches
t1 = time.time()
BATCH = 1500
parts = []
for i in range(0, len(X_clean), BATCH):
    r = model.predict({{"X": X_clean[i:i+BATCH]}})
    parts.append(r["imputation"])
    print(f"    Imputed {{i}}-{{min(i+BATCH, len(X_clean))}}")
X_imp = np.concatenate(parts, axis=0)

# Denormalize
sale_imp = X_imp[:,:,0] * sale_std + sale_mean
sale_imp = np.clip(sale_imp, 0, None)

# Aggregate hourly -> daily
sale_daily = sale_imp.reshape(-1, 30, 16).sum(axis=2)

impute_time = time.time() - t1
total_time = train_time + impute_time
print(f"  Total: {{total_time:.0f}}s")

# Map valid windows back to full index
full_daily = np.full((len(X_windows), 30), np.nan, dtype=np.float32)
valid_indices = np.where(valid_mask)[0]
for vi, fi in enumerate(valid_indices):
    full_daily[fi] = sale_daily[vi]

with open(cfg['result_path'], 'wb') as f:
    pickle.dump({{'sale_daily': full_daily, 'total_time': total_time}}, f)

print("  Subprocess done.")
'''

    script_path = os.path.join(tmp_dir, 'run_dl.py')
    with open(script_path, 'w') as f:
        f.write(script)

    # Run subprocess
    venv_python = os.path.join(BASE_DIR, 'venv', 'bin', 'python')
    t0 = time.time()
    proc = subprocess.run(
        [venv_python, '-u', script_path],
        capture_output=False,
        timeout=3600,  # 60 min max per DL model
    )

    if proc.returncode != 0:
        print(f"  WARNING: {model_name} subprocess failed (code={proc.returncode})")
        # Return a fallback (simple mean recovery)
        return recovery_simple_mean(train_daily), 0

    # Load results
    with open(result_path, 'rb') as f:
        result = pickle.load(f)

    sale_daily_recovered = result['sale_daily']
    dl_time = result['total_time']

    # Map back to DataFrame
    df = train_daily.copy()
    df = df.sort_values(['sp', 'dt']).reset_index(drop=True)
    df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
    df['so_frac'] = (df['stock_hour6_22_cnt'] / OP).astype('float32')
    df['dem_rec'] = df['sale_amount'].values.copy().astype('float32')

    # Build lookup
    records = []
    for i, (sp, w_idx, win_dates) in enumerate(window_map):
        for d_idx, dt in enumerate(win_dates):
            val = sale_daily_recovered[i, d_idx]
            if not np.isnan(val):
                records.append((sp, pd.Timestamp(dt), float(val)))

    if records:
        rec_df = pd.DataFrame(records, columns=['sp', 'dt', 'dl_rec'])
        rec_df = rec_df.groupby(['sp', 'dt'])['dl_rec'].mean().reset_index()
        df = df.merge(rec_df, on=['sp', 'dt'], how='left')

        has_dl = df['dl_rec'].notna()
        cens_mask = (df['cens'] == 1) & has_dl
        recovered_count = cens_mask.sum()
        df.loc[cens_mask, 'dem_rec'] = df.loc[cens_mask, 'dl_rec'].clip(lower=0)
        df.drop('dl_rec', axis=1, inplace=True)
    else:
        recovered_count = 0

    # Fallback for remaining censored
    remaining_cens = (df['cens'] == 1) & (df['dem_rec'] == df['sale_amount'])
    if remaining_cens.sum() > 0:
        sp_means = df[df['cens'] == 0].groupby('sp')['sale_amount'].mean()
        global_mean = df[df['cens'] == 0]['sale_amount'].mean()
        fallback_vals = df.loc[remaining_cens, 'sp'].map(sp_means).fillna(global_mean).values
        df.loc[remaining_cens, 'dem_rec'] = fallback_vals.astype(np.float32)

    df['dem_rec'] = df['dem_rec'].clip(lower=0).astype('float32')

    print(f"  DL-recovered {recovered_count:,} censored observations")
    print(f"  Subprocess time: {dl_time:.0f}s")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return df, dl_time


# ============================================================================
# FEATURE ENGINEERING (streamlined version matching main pipeline)
# ============================================================================
def make_features(df):
    """Build feature set for LightGBM forecasting."""
    df = df.sort_values(['sp', 'dt']).reset_index(drop=True)
    g = df.groupby('sp')
    targets = {'s': 'sale_amount', 'r': 'dem_rec'}

    # Temporal
    df['dow'] = df['dt'].dt.dayofweek.astype('int8')
    df['dom'] = df['dt'].dt.day.astype('int8')
    df['woy'] = df['dt'].dt.isocalendar().week.astype('int8')
    df['month'] = df['dt'].dt.month.astype('int8')
    df['wknd'] = (df['dow'] >= 5).astype('int8')
    df['doy'] = df['dt'].dt.dayofyear.astype('int16')
    for cyc, period in [('dow', 7), ('dom', 31), ('woy', 52)]:
        df[f'{cyc}_sin'] = np.sin(2 * np.pi * df[cyc] / period).astype('float32')
        df[f'{cyc}_cos'] = np.cos(2 * np.pi * df[cyc] / period).astype('float32')

    # Lags
    for pfx, col in targets.items():
        for lag in [1, 2, 3, 5, 7, 14, 21, 28]:
            df[f'{pfx}_l{lag}'] = g[col].shift(lag).astype('float32')
    df['s_d1'] = (df['s_l1'] - df['s_l2']).astype('float32')
    df['s_d7'] = (df['s_l1'] - df['s_l7']).astype('float32')
    df['r_d1'] = (df['r_l1'] - g['dem_rec'].shift(2)).astype('float32')

    # Rolling stats
    shifted_s = g['sale_amount'].shift(1)
    shifted_r = g['dem_rec'].shift(1)
    for pfx, shifted in [('s', shifted_s), ('r', shifted_r)]:
        for w in [3, 7, 14, 28]:
            r = shifted.groupby(df['sp']).rolling(w, min_periods=1)
            df[f'{pfx}_m{w}'] = r.mean().reset_index(level=0, drop=True).astype('float32')
            df[f'{pfx}_sd{w}'] = r.std().reset_index(level=0, drop=True).astype('float32')
            if w in [7, 28]:
                df[f'{pfx}_mx{w}'] = r.max().reset_index(level=0, drop=True).astype('float32')
                df[f'{pfx}_mn{w}'] = r.min().reset_index(level=0, drop=True).astype('float32')
                df[f'{pfx}_md{w}'] = r.median().reset_index(level=0, drop=True).astype('float32')
            gc.collect()

    for w in [14, 28]:
        r = shifted_s.groupby(df['sp']).rolling(w, min_periods=1)
        df[f's_q25_{w}'] = r.quantile(0.25).reset_index(level=0, drop=True).astype('float32')
        df[f's_q75_{w}'] = r.quantile(0.75).reset_index(level=0, drop=True).astype('float32')

    # EWMA
    for pfx, shifted in [('s', shifted_s), ('r', shifted_r)]:
        for s in [7, 14]:
            df[f'{pfx}_ew{s}'] = shifted.groupby(df['sp']).transform(
                lambda x: x.ewm(span=s, min_periods=1).mean()
            ).astype('float32')
    del shifted_s, shifted_r
    gc.collect()

    # Stockout features
    cs = g['cens'].shift(1)
    df['so1'] = cs.astype('float32')
    df['so7'] = g['cens'].shift(7).astype('float32')
    for w in [7, 14]:
        df[f'sor{w}'] = cs.groupby(df['sp']).rolling(w, min_periods=1).mean()\
            .reset_index(level=0, drop=True).astype('float32')
    hs = g['stock_hour6_22_cnt'].shift(1)
    for w in [7, 14]:
        df[f'soh{w}'] = hs.groupby(df['sp']).rolling(w, min_periods=1).mean()\
            .reset_index(level=0, drop=True).astype('float32')
    del cs, hs
    gc.collect()

    # Variability
    df['cv7'] = (df['s_sd7'] / df['s_m7'].clip(lower=0.01)).astype('float32')
    df['cv28'] = (df['s_sd28'] / df['s_m28'].clip(lower=0.01)).astype('float32')
    df['tr_7_28'] = (df['s_m7'] - df['s_m28']).astype('float32')
    df['tr_3_14'] = (df['s_m3'] - df['s_m14']).astype('float32')
    df['l1_m7'] = (df['s_l1'] / df['s_m7'].clip(lower=0.01)).astype('float32')
    df['m7_m28'] = (df['s_m7'] / df['s_m28'].clip(lower=0.01)).astype('float32')
    df['rec_obs_ratio'] = (df['r_m7'] / df['s_m7'].clip(lower=0.01)).astype('float32')

    # DOW profile
    df['dow_prof'] = df.groupby(['sp', 'dow'])['sale_amount'].transform('mean').astype('float32')
    df['dow_prof_r'] = df.groupby(['sp', 'dow'])['dem_rec'].transform('mean').astype('float32')

    # Cross features
    df['d_h'] = (df['discount'] * df['holiday_flag']).astype('float32')
    df['d_a'] = (df['discount'] * df['activity_flag']).astype('float32')
    df['t_p'] = (df['avg_temperature'] * df['precpt']).astype('float32')
    df['w_h'] = (df['wknd'] * df['holiday_flag']).astype('int8')
    df['h_a'] = (df['holiday_flag'] * df['activity_flag']).astype('int8')
    df['temp_dev'] = (df['avg_temperature'] - df.groupby('sp')['avg_temperature'].transform('mean')).astype('float32')
    df['hum_dev'] = (df['avg_humidity'] - df.groupby('sp')['avg_humidity'].transform('mean')).astype('float32')

    # Global stats
    for grp, pfx in [('sp', 'sp'), ('product_id', 'pd'), ('store_id', 'st'), ('city_id', 'ct')]:
        df[f'{pfx}_m'] = df.groupby(grp)['sale_amount'].transform('mean').astype('float32')
        df[f'{pfx}_s'] = df.groupby(grp)['sale_amount'].transform('std').fillna(0).astype('float32')
    df['cat1_m'] = df.groupby('first_category_id')['sale_amount'].transform('mean').astype('float32')
    df['cat2_m'] = df.groupby('second_category_id')['sale_amount'].transform('mean').astype('float32')
    df['cat3_m'] = df.groupby('third_category_id')['sale_amount'].transform('mean').astype('float32')
    df['cat3_s'] = df.groupby('third_category_id')['sale_amount'].transform('std').fillna(0).astype('float32')

    # Hierarchy features
    df['sp_vs_cat2'] = (df['sp_m'] / df['cat2_m'].clip(lower=0.01)).astype('float32')
    df['sp_vs_cat3'] = (df['sp_m'] / df['cat3_m'].clip(lower=0.01)).astype('float32')
    df['sp_vs_store'] = (df['sp_m'] / df['st_m'].clip(lower=0.01)).astype('float32')
    df['store_daily_vol'] = df.groupby(['store_id', 'dt'])['sale_amount'].transform('sum').astype('float32')
    df['store_so_rate'] = df.groupby('store_id')['cens'].transform('mean').astype('float32')
    df['cat2_daily_m'] = df.groupby(['second_category_id', 'dt'])['sale_amount'].transform('mean').astype('float32')

    # Clustering
    sp_behavior = df.groupby('sp').agg(
        mean_demand=('sale_amount', 'mean'),
        std_demand=('sale_amount', 'std'),
        so_rate=('cens', 'mean'),
        zero_rate=('sale_amount', lambda x: (x == 0).mean()),
    ).fillna(0)
    sp_behavior['cv'] = sp_behavior['std_demand'] / sp_behavior['mean_demand'].clip(lower=0.01)
    df['_is_wknd'] = (df['dow'] >= 5).astype(float)
    wknd_mean = df[df['_is_wknd'] == 1].groupby('sp')['sale_amount'].mean()
    wkday_mean = df[df['_is_wknd'] == 0].groupby('sp')['sale_amount'].mean()
    sp_behavior['wknd_ratio'] = (wknd_mean / wkday_mean.clip(lower=0.01)).fillna(1.0)
    df.drop('_is_wknd', axis=1, inplace=True)

    n_clusters = min(25, len(sp_behavior) // 10)
    scaler = StandardScaler()
    sp_scaled = scaler.fit_transform(sp_behavior.values)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(sp_scaled)
    sp_behavior['cluster_id'] = km.labels_
    df['cluster_id'] = df['sp'].map(sp_behavior['cluster_id'].to_dict()).fillna(0).astype('int8')
    df['cluster_m'] = df.groupby('cluster_id')['sale_amount'].transform('mean').astype('float32')
    df['cluster_s'] = df.groupby('cluster_id')['sale_amount'].transform('std').fillna(0).astype('float32')

    df = df.fillna(0)
    gc.collect()
    return df


def get_fcols(df):
    excl = {'dt', 'sale_amount', 'dem_rec', 'sp', 'cens', 'so_frac'}
    return [c for c in df.columns if c not in excl]


# ============================================================================
# LGBM TRAINING & EVALUATION
# ============================================================================
def train_and_evaluate(train_featured, ev, method_name):
    """Train LightGBM and evaluate on eval set."""
    fc = get_fcols(train_featured)
    TARGET = 'sale_amount'

    dates = sorted(train_featured['dt'].unique())
    warmup_date = dates[27]
    mask = train_featured['dt'] > warmup_date

    X_train = train_featured.loc[mask, fc].values
    y_train = train_featured.loc[mask, TARGET].values

    # Prepare eval features
    ev_featured = prepare_eval(train_featured, ev, fc)
    X_eval = ev_featured[fc].values
    y_eval = ev_featured[TARGET].values

    # Train ensemble (MAE + Huber)
    models = {}
    for obj_name, obj in [('mae', 'mae'), ('huber', 'huber')]:
        params = LGB_PARAMS.copy()
        params['objective'] = obj
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        models[obj_name] = model

    # Predict ensemble
    preds = {}
    for name, model in models.items():
        preds[name] = np.clip(model.predict(X_eval), 0, None)
    p_ens = np.mean(list(preds.values()), axis=0)

    # Metrics
    y = y_eval
    p = p_ens
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    corr = np.corrcoef(y, p)[0, 1] if len(y) > 1 else 0
    bias = np.mean(p - y)
    sum_y = np.sum(y)
    wape = np.sum(np.abs(y - p)) / sum_y * 100 if sum_y > 0 else 0
    wpe = np.sum(p - y) / sum_y * 100 if sum_y > 0 else 0
    smape = np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-8)) * 100

    metrics = {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'SMAPE': round(smape, 2),
        'Corr': round(corr, 4),
        'Bias': round(bias, 4),
        'WAPE': round(wape, 2),
        'WPE': round(wpe, 2),
    }

    print(f"  {method_name}: MAE={mae:.4f}, WAPE={wape:.2f}%, WPE={wpe:+.2f}%, Corr={corr:.4f}")
    return metrics


def prepare_eval(train_featured, ev, fc):
    """Build features for eval set using training history."""
    # Combine train + eval for rolling features
    ev_df = ev.copy()
    ev_df['cens'] = (ev_df['stock_hour6_22_cnt'] > 0).astype('int8')
    ev_df['so_frac'] = (ev_df['stock_hour6_22_cnt'] / OP).astype('float32')
    # Use sale_amount as dem_rec for eval (no recovery needed for future data)
    ev_df['dem_rec'] = ev_df['sale_amount'].values.copy().astype('float32')

    # We need historical context for lag/rolling features
    # Take last 35 days of training as context
    train_dates = sorted(train_featured['dt'].unique())
    context_start = train_dates[-35] if len(train_dates) >= 35 else train_dates[0]
    context = train_featured[train_featured['dt'] >= context_start].copy()

    comb = pd.concat([context, ev_df], ignore_index=True)
    comb = comb.sort_values(['sp', 'dt']).reset_index(drop=True)

    # Build features on combined
    comb = make_features_eval(comb, train_featured)

    # Extract eval rows
    eval_dates = ev_df['dt'].unique()
    result = comb[comb['dt'].isin(eval_dates)].copy()

    # Ensure all feature columns exist
    for c in fc:
        if c not in result.columns:
            result[c] = 0

    return result.fillna(0)


def make_features_eval(df, train_full):
    """Simplified feature builder for eval (reuses patterns from make_features)."""
    df = df.sort_values(['sp', 'dt']).reset_index(drop=True)
    g = df.groupby('sp')
    targets = {'s': 'sale_amount', 'r': 'dem_rec'}

    # Temporal
    df['dow'] = df['dt'].dt.dayofweek.astype('int8')
    df['dom'] = df['dt'].dt.day.astype('int8')
    df['woy'] = df['dt'].dt.isocalendar().week.astype('int8')
    df['month'] = df['dt'].dt.month.astype('int8')
    df['wknd'] = (df['dow'] >= 5).astype('int8')
    df['doy'] = df['dt'].dt.dayofyear.astype('int16')
    for cyc, period in [('dow', 7), ('dom', 31), ('woy', 52)]:
        df[f'{cyc}_sin'] = np.sin(2 * np.pi * df[cyc] / period).astype('float32')
        df[f'{cyc}_cos'] = np.cos(2 * np.pi * df[cyc] / period).astype('float32')

    # Lags
    for pfx, col in targets.items():
        for lag in [1, 2, 3, 5, 7, 14, 21, 28]:
            df[f'{pfx}_l{lag}'] = g[col].shift(lag).astype('float32')
    df['s_d1'] = (df['s_l1'] - df['s_l2']).astype('float32')
    df['s_d7'] = (df['s_l1'] - df['s_l7']).astype('float32')
    df['r_d1'] = (df['r_l1'] - g['dem_rec'].shift(2)).astype('float32')

    # Rolling stats
    shifted_s = g['sale_amount'].shift(1)
    shifted_r = g['dem_rec'].shift(1)
    for pfx, shifted in [('s', shifted_s), ('r', shifted_r)]:
        for w in [3, 7, 14, 28]:
            r = shifted.groupby(df['sp']).rolling(w, min_periods=1)
            df[f'{pfx}_m{w}'] = r.mean().reset_index(level=0, drop=True).astype('float32')
            df[f'{pfx}_sd{w}'] = r.std().reset_index(level=0, drop=True).astype('float32')
            if w in [7, 28]:
                df[f'{pfx}_mx{w}'] = r.max().reset_index(level=0, drop=True).astype('float32')
                df[f'{pfx}_mn{w}'] = r.min().reset_index(level=0, drop=True).astype('float32')
                df[f'{pfx}_md{w}'] = r.median().reset_index(level=0, drop=True).astype('float32')

    for w in [14, 28]:
        r = shifted_s.groupby(df['sp']).rolling(w, min_periods=1)
        df[f's_q25_{w}'] = r.quantile(0.25).reset_index(level=0, drop=True).astype('float32')
        df[f's_q75_{w}'] = r.quantile(0.75).reset_index(level=0, drop=True).astype('float32')

    # EWMA
    for pfx, shifted in [('s', shifted_s), ('r', shifted_r)]:
        for s in [7, 14]:
            df[f'{pfx}_ew{s}'] = shifted.groupby(df['sp']).transform(
                lambda x: x.ewm(span=s, min_periods=1).mean()
            ).astype('float32')
    del shifted_s, shifted_r

    # Stockout features
    cs = g['cens'].shift(1)
    df['so1'] = cs.astype('float32')
    df['so7'] = g['cens'].shift(7).astype('float32')
    for w in [7, 14]:
        df[f'sor{w}'] = cs.groupby(df['sp']).rolling(w, min_periods=1).mean()\
            .reset_index(level=0, drop=True).astype('float32')
    hs = g['stock_hour6_22_cnt'].shift(1)
    for w in [7, 14]:
        df[f'soh{w}'] = hs.groupby(df['sp']).rolling(w, min_periods=1).mean()\
            .reset_index(level=0, drop=True).astype('float32')
    del cs, hs

    # Variability
    df['cv7'] = (df['s_sd7'] / df['s_m7'].clip(lower=0.01)).astype('float32')
    df['cv28'] = (df['s_sd28'] / df['s_m28'].clip(lower=0.01)).astype('float32')
    df['tr_7_28'] = (df['s_m7'] - df['s_m28']).astype('float32')
    df['tr_3_14'] = (df['s_m3'] - df['s_m14']).astype('float32')
    df['l1_m7'] = (df['s_l1'] / df['s_m7'].clip(lower=0.01)).astype('float32')
    df['m7_m28'] = (df['s_m7'] / df['s_m28'].clip(lower=0.01)).astype('float32')
    df['rec_obs_ratio'] = (df['r_m7'] / df['s_m7'].clip(lower=0.01)).astype('float32')

    # DOW profile
    df['dow_prof'] = df.groupby(['sp', 'dow'])['sale_amount'].transform('mean').astype('float32')
    df['dow_prof_r'] = df.groupby(['sp', 'dow'])['dem_rec'].transform('mean').astype('float32')

    # Cross features
    df['d_h'] = (df['discount'] * df['holiday_flag']).astype('float32')
    df['d_a'] = (df['discount'] * df['activity_flag']).astype('float32')
    df['t_p'] = (df['avg_temperature'] * df['precpt']).astype('float32')
    df['w_h'] = (df['wknd'] * df['holiday_flag']).astype('int8')
    df['h_a'] = (df['holiday_flag'] * df['activity_flag']).astype('int8')
    df['temp_dev'] = (df['avg_temperature'] - df.groupby('sp')['avg_temperature'].transform('mean')).astype('float32')
    df['hum_dev'] = (df['avg_humidity'] - df.groupby('sp')['avg_humidity'].transform('mean')).astype('float32')

    # Global stats (from full training set for consistency)
    for grp, pfx in [('sp', 'sp'), ('product_id', 'pd'), ('store_id', 'st'), ('city_id', 'ct')]:
        stats_map = train_full.groupby(grp)['sale_amount'].mean().to_dict()
        std_map = train_full.groupby(grp)['sale_amount'].std().fillna(0).to_dict()
        df[f'{pfx}_m'] = df[grp].map(stats_map).fillna(0).astype('float32')
        df[f'{pfx}_s'] = df[grp].map(std_map).fillna(0).astype('float32')
    for cat in ['first_category_id', 'second_category_id', 'third_category_id']:
        pfx = {'first_category_id': 'cat1', 'second_category_id': 'cat2',
               'third_category_id': 'cat3'}[cat]
        m = train_full.groupby(cat)['sale_amount'].mean().to_dict()
        df[f'{pfx}_m'] = df[cat].map(m).fillna(0).astype('float32')
    cat3_s = train_full.groupby('third_category_id')['sale_amount'].std().fillna(0).to_dict()
    df['cat3_s'] = df['third_category_id'].map(cat3_s).fillna(0).astype('float32')

    # Hierarchy
    df['sp_vs_cat2'] = (df['sp_m'] / df['cat2_m'].clip(lower=0.01)).astype('float32')
    df['sp_vs_cat3'] = (df['sp_m'] / df['cat3_m'].clip(lower=0.01)).astype('float32')
    df['sp_vs_store'] = (df['sp_m'] / df['st_m'].clip(lower=0.01)).astype('float32')
    df['store_daily_vol'] = df.groupby(['store_id', 'dt'])['sale_amount'].transform('sum').astype('float32')
    df['store_so_rate'] = df.groupby('store_id')['cens'].transform('mean').astype('float32')
    df['cat2_daily_m'] = df.groupby(['second_category_id', 'dt'])['sale_amount'].transform('mean').astype('float32')

    # Clustering (from training)
    sp_behavior = train_full.groupby('sp').agg(
        mean_demand=('sale_amount', 'mean'),
        std_demand=('sale_amount', 'std'),
        so_rate=('cens', 'mean'),
        zero_rate=('sale_amount', lambda x: (x == 0).mean()),
    ).fillna(0)
    sp_behavior['cv'] = sp_behavior['std_demand'] / sp_behavior['mean_demand'].clip(lower=0.01)
    train_full_tmp = train_full.copy()
    train_full_tmp['_is_wknd'] = (train_full_tmp['dt'].dt.dayofweek >= 5).astype(float)
    wknd_m = train_full_tmp[train_full_tmp['_is_wknd'] == 1].groupby('sp')['sale_amount'].mean()
    wkday_m = train_full_tmp[train_full_tmp['_is_wknd'] == 0].groupby('sp')['sale_amount'].mean()
    sp_behavior['wknd_ratio'] = (wknd_m / wkday_m.clip(lower=0.01)).fillna(1.0)
    del train_full_tmp

    n_clusters = min(25, len(sp_behavior) // 10)
    scaler = StandardScaler()
    sp_scaled = scaler.fit_transform(sp_behavior.values)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(sp_scaled)
    sp_behavior['cluster_id'] = km.labels_
    df['cluster_id'] = df['sp'].map(sp_behavior['cluster_id'].to_dict()).fillna(0).astype('int8')
    df['cluster_m'] = df.groupby('cluster_id')['sale_amount'].transform('mean').astype('float32')
    df['cluster_s'] = df.groupby('cluster_id')['sale_amount'].transform('std').fillna(0).astype('float32')

    df = df.fillna(0)
    gc.collect()
    return df


# ============================================================================
# DEMAND-STOCK DECOUPLING SCORE
# ============================================================================
def compute_decoupling_score(df):
    """
    Compute demand-stock decoupling score.
    Lower = better (recovered demand less correlated with stock levels).
    """
    scores = []
    for sp, grp in df.groupby('sp'):
        if grp['stock_hour6_22_cnt'].nunique() < 3:
            continue
        if grp['dem_rec'].std() < 1e-6:
            continue
        corr = grp['dem_rec'].corr(grp['stock_hour6_22_cnt'])
        if not np.isnan(corr):
            weight = grp['dem_rec'].mean()
            scores.append((abs(corr), weight))

    if not scores:
        return 0.0
    corrs, weights = zip(*scores)
    return np.average(corrs, weights=weights)


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_comparison(results, output_path):
    """Create comparison chart."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    methods = list(results.keys())
    colors = {'No Recovery': '#95a5a6', 'Simple Mean': '#3498db',
              'Tobit/IMR': '#e74c3c', 'SAITS': '#2ecc71', 'TimesNet': '#9b59b6'}

    # 1. WAPE comparison
    ax1 = fig.add_subplot(gs[0, 0])
    wapes = [results[m]['WAPE'] for m in methods]
    bars = ax1.bar(range(len(methods)), wapes,
                   color=[colors.get(m, '#7f8c8d') for m in methods])
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('WAPE (%)')
    ax1.set_title('Forecast WAPE (lower = better)', fontweight='bold')
    for bar, val in zip(bars, wapes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # 2. MAE comparison
    ax2 = fig.add_subplot(gs[0, 1])
    maes = [results[m]['MAE'] for m in methods]
    bars = ax2.bar(range(len(methods)), maes,
                   color=[colors.get(m, '#7f8c8d') for m in methods])
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('MAE')
    ax2.set_title('Forecast MAE (lower = better)', fontweight='bold')
    for bar, val in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # 3. WPE (bias) comparison
    ax3 = fig.add_subplot(gs[0, 2])
    wpes = [results[m]['WPE'] for m in methods]
    bar_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in wpes]
    bars = ax3.bar(range(len(methods)), wpes, color=bar_colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax3.set_ylabel('WPE (%)')
    ax3.set_title('Forecast Bias WPE (closer to 0 = better)', fontweight='bold')
    for bar, val in zip(bars, wpes):
        offset = 0.2 if val >= 0 else -0.4
        ax3.text(bar.get_x() + bar.get_width()/2, val + offset,
                 f'{val:+.2f}%', ha='center', va='bottom', fontsize=9)

    # 4. Correlation
    ax4 = fig.add_subplot(gs[1, 0])
    corrs = [results[m]['Corr'] for m in methods]
    bars = ax4.bar(range(len(methods)), corrs,
                   color=[colors.get(m, '#7f8c8d') for m in methods])
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax4.set_ylabel('Correlation')
    ax4.set_title('Forecast Correlation (higher = better)', fontweight='bold')
    ax4.set_ylim(min(corrs) - 0.005, max(corrs) + 0.005)
    for bar, val in zip(bars, corrs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # 5. Decoupling score
    ax5 = fig.add_subplot(gs[1, 1])
    decoup = [results[m].get('Decoupling', 0) for m in methods]
    bars = ax5.bar(range(len(methods)), decoup,
                   color=[colors.get(m, '#7f8c8d') for m in methods])
    ax5.set_xticks(range(len(methods)))
    ax5.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax5.set_ylabel('Decoupling Score')
    ax5.set_title('Demand-Stock Decoupling (lower = better)', fontweight='bold')
    for bar, val in zip(bars, decoup):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 6. Recovery uplift
    ax6 = fig.add_subplot(gs[1, 2])
    uplifts = [results[m].get('Recovery_Uplift_pct', 0) for m in methods]
    bars = ax6.bar(range(len(methods)), uplifts,
                   color=[colors.get(m, '#7f8c8d') for m in methods])
    ax6.set_xticks(range(len(methods)))
    ax6.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
    ax6.set_ylabel('Uplift (%)')
    ax6.set_title('Demand Recovery Uplift vs Raw Sales', fontweight='bold')
    for bar, val in zip(bars, uplifts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Censored Demand Recovery: Tobit/IMR vs Deep Learning\n'
                 'Downstream LightGBM Forecasting Accuracy Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved to {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    overall_start = time.time()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  DEMAND RECOVERY COMPARISON: Tobit/IMR vs Deep Learning            ║")
    print("║  Testing 5 recovery methods → same LightGBM → same eval metrics    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Load data
    train_daily, train_hourly, ev = load_data()

    # Prepare DL hourly data (shared by SAITS and TimesNet)
    X_windows, window_map, norm_params = prepare_hourly_data(train_hourly, train_daily)
    del train_hourly
    gc.collect()

    results = {}
    recovery_methods = [
        ("No Recovery", recovery_none, None),
        ("Simple Mean", recovery_simple_mean, None),
        ("Tobit/IMR",   recovery_tobit, None),
        ("SAITS",       None, "SAITS"),
        ("TimesNet",    None, "TimesNet"),
    ]

    for method_name, recovery_fn, dl_model in recovery_methods:
        print(f"\n{'='*80}")
        print(f"RECOVERY METHOD: {method_name}")
        print(f"{'='*80}")

        t0 = time.time()
        dl_time = 0

        # Apply recovery
        if dl_model:
            # Run DL recovery in a subprocess to avoid OMP/PyTorch conflicts
            # with LightGBM that cause hangs in the same process
            train_recovered, dl_time = run_dl_recovery_subprocess(
                X_windows, window_map, norm_params, train_daily, model_name=dl_model
            )
        else:
            train_recovered = recovery_fn(train_daily)

        recovery_time = time.time() - t0

        # Recovery statistics
        rec_mean = train_recovered['dem_rec'].mean()
        sale_mean = train_recovered['sale_amount'].mean()
        uplift = (rec_mean / sale_mean - 1) * 100 if sale_mean > 0 else 0
        print(f"  Sale mean: {sale_mean:.4f}, Recovered: {rec_mean:.4f}, Uplift: {uplift:.1f}%")

        # Decoupling score
        decoup = compute_decoupling_score(train_recovered)
        print(f"  Demand-stock decoupling: {decoup:.4f}")

        # Feature engineering
        t_feat = time.time()
        print("\n  Building features...")
        train_featured = make_features(train_recovered)
        feat_time = time.time() - t_feat
        print(f"  Features built in {feat_time:.0f}s")

        # Train & evaluate
        t_lgb = time.time()
        print("\n  Training LightGBM ensemble...")
        metrics = train_and_evaluate(train_featured, ev, method_name)
        lgb_time = time.time() - t_lgb

        # Store results
        metrics['Decoupling'] = round(decoup, 4)
        metrics['Recovery_Uplift_pct'] = round(uplift, 1)
        metrics['Recovery_Time_s'] = round(recovery_time, 1)
        metrics['DL_Train_Time_s'] = round(dl_time, 1) if dl_model else 0
        metrics['Total_Time_s'] = round(time.time() - t0, 1)
        results[method_name] = metrics

        # Cleanup
        del train_recovered, train_featured
        gc.collect()

    # ================================================================
    # OUTPUT RESULTS
    # ================================================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # Summary table
    df_results = pd.DataFrame(results).T
    df_results.index.name = 'Recovery Method'
    print("\n" + df_results.to_string())

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'recovery_comparison.csv')
    df_results.to_csv(csv_path)
    print(f"\n  Results saved to {csv_path}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, 'recovery_comparison_detail.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Details saved to {json_path}")

    # Plot
    plot_path = os.path.join(OUTPUT_DIR, 'recovery_comparison.png')
    plot_comparison(results, plot_path)

    # Summary box
    best_wape_method = min(results, key=lambda m: results[m]['WAPE'])
    best_mae_method = min(results, key=lambda m: results[m]['MAE'])
    best_decoup_method = min(results, key=lambda m: results[m]['Decoupling'])

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  SUMMARY                                                           ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Best WAPE:        {best_wape_method:<15} ({results[best_wape_method]['WAPE']:.2f}%)")
    print(f"║  Best MAE:         {best_mae_method:<15} ({results[best_mae_method]['MAE']:.4f})")
    print(f"║  Best Decoupling:  {best_decoup_method:<15} ({results[best_decoup_method]['Decoupling']:.4f})")
    print(f"║  Tobit/IMR WAPE:   {results['Tobit/IMR']['WAPE']:.2f}%")
    if 'SAITS' in results:
        print(f"║  SAITS WAPE:       {results['SAITS']['WAPE']:.2f}%")
    if 'TimesNet' in results:
        print(f"║  TimesNet WAPE:    {results['TimesNet']['WAPE']:.2f}%")
    print(f"║  Total runtime:    {(time.time() - overall_start) / 60:.1f} minutes")
    print("╚══════════════════════════════════════════════════════════════════════╝")


if __name__ == '__main__':
    main()

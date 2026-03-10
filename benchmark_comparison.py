#!/usr/bin/env python3
"""
Benchmark Comparison: Alternative Forecasting Approaches on FreshRetailNet-50K
==============================================================================
Compares 7 forecasting approaches on the SAME data, SAME split, SAME metrics:

1. Naive Seasonal     — weekday average from last 4 weeks (floor baseline)
2. SSA                — Similar Scenario Average (statistical, weighted history)
3. DLinear            — Simple neural decomposition-linear (AAAI 2023)
4. LightGBM-Minimal   — LightGBM with ~15 basic features (ablation: no censoring/hierarchy)
5. XGBoost-Minimal    — XGBoost with same ~15 basic features
6. CatBoost-Minimal   — CatBoost with same ~15 basic features
7. Full Pipeline      — Our 120-feature LightGBM ensemble + Tobit correction (reference)

All predictions are evaluated with identical forecast metrics AND run through the
same Empirical Newsvendor inventory policy for downstream profit comparison.

Usage:
    python benchmark_comparison.py
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import json
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'freshretailnet', 'raw', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SP = 3000  # Same as main pipeline

COLS = [
    'city_id', 'store_id', 'management_group_id',
    'first_category_id', 'second_category_id', 'third_category_id',
    'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt',
    'discount', 'holiday_flag', 'activity_flag',
    'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',
]

# Cost configuration (identical to main pipeline)
INV_CFG = {
    'h': 0.10,
    'p': 0.50,
    'w': 0.30,
    'unit_revenue': 1.0,
    'unit_cost': 0.40,
}
Co = INV_CFG['h'] + INV_CFG['w']
Cu = INV_CFG['p']
CR = Cu / (Cu + Co)


# ============================================================================
# Section 1: Data Loading (shared)
# ============================================================================
def load_data():
    """Load train/eval data with stratified SP sampling (same as main pipeline)."""
    print("=" * 70)
    print("SECTION 1: Loading Data")
    print("=" * 70)

    train_path = os.path.join(DATA_DIR, 'train.parquet')
    eval_path = os.path.join(DATA_DIR, 'eval.parquet')

    train = pd.read_parquet(train_path, columns=COLS)
    train['sp'] = train['store_id'] * 10000 + train['product_id']
    train['dt'] = pd.to_datetime(train['dt'])

    ev = pd.read_parquet(eval_path, columns=COLS)
    ev['sp'] = ev['store_id'] * 10000 + ev['product_id']
    ev['dt'] = pd.to_datetime(ev['dt'])

    # Stratified sampling by stockout rate (same logic as main pipeline)
    so_rate = train.groupby('sp')['stock_hour6_22_cnt'].apply(
        lambda x: (x > 0).mean()
    ).reset_index()
    so_rate.columns = ['sp', 'so_rate']
    so_rate['bin'] = pd.qcut(so_rate['so_rate'], q=5, labels=False, duplicates='drop')

    sampled = so_rate.groupby('bin').apply(
        lambda x: x.sample(min(len(x), N_SP // 5), random_state=42)
    ).reset_index(drop=True)['sp'].values

    if len(sampled) < N_SP:
        extra = np.random.choice(
            list(set(so_rate['sp']) - set(sampled)),
            N_SP - len(sampled), replace=False
        )
        sampled = np.concatenate([sampled, extra])
    sp_set = set(sampled[:N_SP])

    train = train[train['sp'].isin(sp_set)].copy()
    ev = ev[ev['sp'].isin(sp_set)].copy()

    print(f"  Train: {len(train):,} rows, {train['sp'].nunique():,} SPs")
    print(f"  Eval:  {len(ev):,} rows, {ev['sp'].nunique():,} SPs")
    print(f"  Train dates: {train['dt'].min().date()} to {train['dt'].max().date()}")
    print(f"  Eval dates:  {ev['dt'].min().date()} to {ev['dt'].max().date()}")
    return train, ev


# ============================================================================
# Shared Utilities
# ============================================================================
def compute_metrics(y_true, y_pred, name=""):
    """Compute forecast accuracy metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    bias = np.mean(y_pred - y_true)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

    # SMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_vals = np.where(denom > 1e-8, np.abs(y_true - y_pred) / denom, 0)
    smape = np.mean(smape_vals) * 100

    # WMAE (weighted by actual volume)
    weights = np.abs(y_true) / max(np.sum(np.abs(y_true)), 1e-8)
    wmae = np.sum(weights * np.abs(y_true - y_pred))

    return {
        'Model': name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'SMAPE(%)': round(smape, 2),
        'WMAE': round(wmae, 4),
        'Bias': round(bias, 4),
        'Corr': round(corr, 4),
    }


def eval_policy(Q, D, cfg, name=""):
    """Evaluate an inventory policy (identical to main pipeline)."""
    Q = np.asarray(Q, dtype=float)
    D = np.asarray(D, dtype=float)
    sold = np.minimum(Q, D)
    over = np.maximum(Q - D, 0)
    under = np.maximum(D - Q, 0)
    revenue = sold * cfg['unit_revenue']
    procurement = Q * cfg['unit_cost']
    holding = cfg['h'] * over
    stockout = cfg['p'] * under
    waste = cfg['w'] * over
    total_cost = procurement + holding + stockout + waste
    profit = revenue - total_cost
    return {
        'Model': name,
        'Avg Profit': round(np.mean(profit), 4),
        'SL(Type1)': round(np.mean(Q >= D), 4),
        'Fill Rate': round(np.sum(sold) / max(np.sum(D), 1), 4),
        'SO Rate': round(np.mean(D > Q), 4),
        'Waste%': round(np.sum(over) / max(np.sum(Q), 1) * 100, 2),
    }


def run_empirical_newsvendor(mu, sp_eval, train, cfg):
    """Apply prediction-based Newsvendor to model predictions.

    Q = prediction + z_CR * sigma_hist  (per SP)
    where z_CR is the normal quantile at critical ratio,
    and sigma_hist is the per-SP historical demand std.
    This makes Q depend on each model's predictions.
    """
    z_cr = norm.ppf(CR)  # ~0.74 for CR=0.555
    Q = np.copy(mu)
    for sp in np.unique(sp_eval):
        sp_mask_eval = sp_eval == sp
        sp_hist = train.loc[train['sp'] == sp, 'sale_amount'].values
        sigma = np.std(sp_hist) if len(sp_hist) >= 5 else 0.5
        Q[sp_mask_eval] = mu[sp_mask_eval] + z_cr * sigma
    return np.maximum(Q, 0)


# ============================================================================
# Section 2: Naive Seasonal Baseline
# ============================================================================
def run_naive_seasonal(train, ev):
    """Predict eval demand as same-weekday average from last 4 weeks of training."""
    print("\n" + "=" * 70)
    print("SECTION 2: Naive Seasonal Baseline")
    print("=" * 70)
    t0 = time.time()

    # Last 28 days of training
    cutoff = train['dt'].max() - pd.Timedelta(days=27)
    recent = train[train['dt'] >= cutoff].copy()
    recent['dow'] = recent['dt'].dt.dayofweek

    # Mean sale_amount by SP × day-of-week
    dow_avg = recent.groupby(['sp', 'dow'])['sale_amount'].mean().reset_index()
    dow_avg.columns = ['sp', 'dow', 'pred']

    # Map to eval
    ev_pred = ev[['sp', 'dt']].copy()
    ev_pred['dow'] = ev_pred['dt'].dt.dayofweek
    ev_pred = ev_pred.merge(dow_avg, on=['sp', 'dow'], how='left')

    # Fallback: SP mean if weekday not seen
    sp_mean = recent.groupby('sp')['sale_amount'].mean()
    ev_pred['pred'] = ev_pred['pred'].fillna(ev_pred['sp'].map(sp_mean)).fillna(0)

    preds = np.clip(ev_pred['pred'].values, 0, None)
    print(f"  Done in {time.time() - t0:.1f}s")
    return preds


# ============================================================================
# Section 3: SSA (Similar Scenario Average)
# ============================================================================
def run_ssa(train, ev):
    """Similar Scenario Average: weighted historical average."""
    print("\n" + "=" * 70)
    print("SECTION 3: SSA (Similar Scenario Average)")
    print("=" * 70)
    t0 = time.time()

    train_c = train.copy()
    train_c['dow'] = train_c['dt'].dt.dayofweek
    train_c['day_num'] = (train_c['dt'] - train_c['dt'].min()).dt.days

    ev_c = ev[['sp', 'dt', 'discount']].copy()
    ev_c['dow'] = ev_c['dt'].dt.dayofweek
    ev_c['day_num'] = (ev_c['dt'] - train_c['dt'].min()).dt.days

    preds = np.zeros(len(ev_c))

    # Process per SP for efficiency
    sp_groups_train = {sp: g for sp, g in train_c.groupby('sp')}

    for i, row in ev_c.iterrows():
        sp = row['sp']
        if sp not in sp_groups_train:
            continue
        hist = sp_groups_train[sp]

        # Weights
        date_dist = np.abs(hist['day_num'].values - row['day_num'])
        w_date = np.exp(-date_dist / 30.0)

        dow_match = (hist['dow'].values == row['dow']).astype(float) * 2.0 + 1.0

        disc_diff = np.abs(hist['discount'].values - row['discount'])
        w_disc = np.exp(-disc_diff)

        weights = w_date * dow_match * w_disc
        w_sum = weights.sum()
        if w_sum > 1e-8:
            preds[ev_c.index.get_loc(i)] = np.dot(weights, hist['sale_amount'].values) / w_sum

    preds = np.clip(preds, 0, None)
    print(f"  Done in {time.time() - t0:.1f}s")
    return preds


# ============================================================================
# Section 4: DLinear (Simple Neural Model)
# ============================================================================
class MovingAvg(nn.Module):
    """Moving average block for series decomposition."""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        # Pad front and back with edge values
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        # AvgPool expects [batch, channels, length]
        x_t = x_pad.permute(0, 2, 1)
        out = self.avg(x_t).permute(0, 2, 1)
        return out[:, :x.shape[1], :]


class DLinearModel(nn.Module):
    """DLinear: Decomposition-Linear for time series forecasting (AAAI 2023)."""
    def __init__(self, seq_len=28, pred_len=7, n_channels=7, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels

        self.decomp = MovingAvg(kernel_size)

        # Per-channel linear layers
        self.linear_seasonal = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(n_channels)
        ])
        self.linear_trend = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(n_channels)
        ])

    def forward(self, x):
        # x: [batch, seq_len, channels]
        trend = self.decomp(x)
        seasonal = x - trend

        # Per-channel projection
        seasonal_out = torch.zeros(x.shape[0], self.pred_len, self.n_channels,
                                   device=x.device)
        trend_out = torch.zeros_like(seasonal_out)

        for i in range(self.n_channels):
            seasonal_out[:, :, i] = self.linear_seasonal[i](seasonal[:, :, i])
            trend_out[:, :, i] = self.linear_trend[i](trend[:, :, i])

        return seasonal_out + trend_out


class SlidingWindowDataset(Dataset):
    """Create sliding windows from per-SP time series."""
    def __init__(self, data, seq_len=28, pred_len=7):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.windows = []
        self.targets = []

        for sp, group in data.groupby('sp'):
            group = group.sort_values('dt')
            vals = group[['sale_amount', 'discount', 'holiday_flag',
                          'activity_flag', 'precpt', 'avg_temperature',
                          'avg_humidity']].values.astype(np.float32)
            total = seq_len + pred_len
            for i in range(len(vals) - total + 1):
                self.windows.append(vals[i:i + seq_len])
                self.targets.append(vals[i + seq_len:i + total, 0])  # sale_amount only

        self.windows = np.array(self.windows)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx]), torch.tensor(self.targets[idx])


def run_dlinear(train, ev):
    """Train DLinear model and predict eval period."""
    print("\n" + "=" * 70)
    print("SECTION 4: DLinear (Neural Baseline)")
    print("=" * 70)
    t0 = time.time()

    seq_len = 28
    pred_len = 7
    n_channels = 7

    # Build training dataset (sliding windows)
    print("  Building sliding window dataset...")
    ds = SlidingWindowDataset(train, seq_len=seq_len, pred_len=pred_len)
    print(f"  Training windows: {len(ds):,}")

    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)

    # Model
    device = torch.device('cpu')
    model = DLinearModel(seq_len=seq_len, pred_len=pred_len,
                         n_channels=n_channels, kernel_size=25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()  # MAE

    # Train
    n_epochs = 15
    print(f"  Training DLinear for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)[:, :, 0]  # Only sale_amount channel
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: MAE = {total_loss/n_batches:.4f}")

    # Predict eval: for each SP, use last seq_len days of training as input
    print("  Predicting eval period...")
    model.eval()
    ev_sorted = ev.sort_values(['sp', 'dt'])
    sp_eval = ev_sorted['sp'].values
    unique_sps = ev_sorted['sp'].unique()

    preds = np.zeros(len(ev_sorted))

    with torch.no_grad():
        for sp in unique_sps:
            sp_train = train[train['sp'] == sp].sort_values('dt')
            sp_ev = ev_sorted[ev_sorted['sp'] == sp]

            if len(sp_train) < seq_len:
                # Fallback: use mean
                preds[ev_sorted['sp'] == sp] = sp_train['sale_amount'].mean() \
                    if len(sp_train) > 0 else 0
                continue

            # Take last seq_len days of training
            input_vals = sp_train[['sale_amount', 'discount', 'holiday_flag',
                                   'activity_flag', 'precpt', 'avg_temperature',
                                   'avg_humidity']].values[-seq_len:].astype(np.float32)
            x = torch.tensor(input_vals).unsqueeze(0).to(device)
            pred = model(x)[0, :, 0].cpu().numpy()  # [pred_len]

            # Map to eval rows (up to pred_len days)
            n_eval = min(len(sp_ev), pred_len)
            idx = ev_sorted.index[ev_sorted['sp'] == sp][:n_eval]
            preds[ev_sorted.index.get_indexer(idx)] = pred[:n_eval]

    preds = np.clip(preds, 0, None)

    # Reorder to match original ev index
    result = pd.Series(preds, index=ev_sorted.index)
    result = result.reindex(ev.index).fillna(0).values

    print(f"  Done in {time.time() - t0:.1f}s")
    return result


# ============================================================================
# Section 5: LightGBM-Minimal (Ablation)
# ============================================================================
def build_minimal_features(df):
    """Build ~15 basic features — NO censoring, NO hierarchy, NO clustering."""
    df = df.sort_values(['sp', 'dt']).copy()
    g = df.groupby('sp')['sale_amount']

    # Lags
    df['lag_1'] = g.shift(1)
    df['lag_7'] = g.shift(7)
    df['lag_14'] = g.shift(14)

    # Rolling
    df['roll_m7'] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    df['roll_s7'] = g.transform(lambda x: x.shift(1).rolling(7, min_periods=3).std())

    # Calendar
    df['dow'] = df['dt'].dt.dayofweek
    df['is_wknd'] = (df['dow'] >= 5).astype(int)
    df['month'] = df['dt'].dt.month

    # Weather & promo (raw columns already present)
    # discount, holiday_flag, activity_flag, avg_temperature, avg_humidity, precpt

    fcols = ['lag_1', 'lag_7', 'lag_14', 'roll_m7', 'roll_s7',
             'dow', 'is_wknd', 'month',
             'discount', 'holiday_flag', 'activity_flag',
             'avg_temperature', 'avg_humidity', 'precpt']
    return df, fcols


def run_lgbm_minimal(train, ev):
    """Train LightGBM with minimal features (ablation baseline)."""
    print("\n" + "=" * 70)
    print("SECTION 5: LightGBM-Minimal (Ablation)")
    print("=" * 70)
    t0 = time.time()

    # Combine last 28 days of train with eval for feature computation
    warmup = train['dt'].max() - pd.Timedelta(days=27)
    comb = pd.concat([train, ev], ignore_index=True)
    comb, fcols = build_minimal_features(comb)

    # Split back
    train_dates = set(train['dt'].unique())
    eval_dates = set(ev['dt'].unique())

    t_mask = comb['dt'].isin(train_dates) & (comb['dt'] >= warmup)
    e_mask = comb['dt'].isin(eval_dates)

    X_train = comb.loc[t_mask, fcols].fillna(0).values
    y_train = comb.loc[t_mask, 'sale_amount'].values
    X_eval = comb.loc[e_mask, fcols].fillna(0).values

    print(f"  Features: {len(fcols)}")
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Eval samples: {len(X_eval):,}")

    # Train
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': 'mae',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 30,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    # Use last 7 days of training as validation
    val_cutoff = comb.loc[t_mask, 'dt'].max() - pd.Timedelta(days=6)
    v_mask = t_mask & (comb['dt'] >= val_cutoff)
    t_only = t_mask & (comb['dt'] < val_cutoff)

    dtrain = lgb.Dataset(comb.loc[t_only, fcols].fillna(0).values,
                         label=comb.loc[t_only, 'sale_amount'].values)
    dval = lgb.Dataset(comb.loc[v_mask, fcols].fillna(0).values,
                       label=comb.loc[v_mask, 'sale_amount'].values, reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=1500,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    preds = np.clip(model.predict(X_eval), 0, None)

    # Align back to ev index
    eval_idx = comb.loc[e_mask].index
    result = pd.Series(preds, index=eval_idx)
    # Match to original ev ordering
    ev_with_idx = ev.copy()
    ev_with_idx['_orig_idx'] = range(len(ev))
    comb_eval = comb.loc[e_mask].copy()
    comb_eval['_pred'] = preds

    # Simple approach: align by sp + dt
    ev_merged = ev[['sp', 'dt']].merge(
        comb_eval[['sp', 'dt', '_pred']], on=['sp', 'dt'], how='left'
    )
    final_preds = ev_merged['_pred'].fillna(0).values

    print(f"  Done in {time.time() - t0:.1f}s")
    return final_preds


# ============================================================================
# Section 5b: XGBoost-Minimal (same features as LightGBM-Minimal)
# ============================================================================
def run_xgboost_minimal(train, ev):
    """Train XGBoost with same minimal features as LightGBM ablation."""
    print("\n" + "=" * 70)
    print("SECTION 5b: XGBoost-Minimal")
    print("=" * 70)
    t0 = time.time()

    warmup = train['dt'].max() - pd.Timedelta(days=27)
    comb = pd.concat([train, ev], ignore_index=True)
    comb, fcols = build_minimal_features(comb)

    train_dates = set(train['dt'].unique())
    eval_dates = set(ev['dt'].unique())

    t_mask = comb['dt'].isin(train_dates) & (comb['dt'] >= warmup)
    e_mask = comb['dt'].isin(eval_dates)

    # Train/val split: last 7 days of training as validation
    val_cutoff = comb.loc[t_mask, 'dt'].max() - pd.Timedelta(days=6)
    t_only = t_mask & (comb['dt'] < val_cutoff)
    v_mask = t_mask & (comb['dt'] >= val_cutoff)

    X_tr = comb.loc[t_only, fcols].fillna(0).values.astype(np.float32)
    y_tr = comb.loc[t_only, 'sale_amount'].values.astype(np.float32)
    X_val = comb.loc[v_mask, fcols].fillna(0).values.astype(np.float32)
    y_val = comb.loc[v_mask, 'sale_amount'].values.astype(np.float32)
    X_eval = comb.loc[e_mask, fcols].fillna(0).values.astype(np.float32)

    print(f"  Features: {len(fcols)}")
    print(f"  Train samples: {len(X_tr):,}, Val: {len(X_val):,}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=fcols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=fcols)
    dtest = xgb.DMatrix(X_eval, feature_names=fcols)

    params = {
        'objective': 'reg:absoluteerror',
        'eval_metric': 'mae',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 30,
        'tree_method': 'hist',
        'seed': 42,
        'verbosity': 0,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=1500,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    preds = np.clip(model.predict(dtest), 0, None)

    # Align back to ev index
    comb_eval = comb.loc[e_mask].copy()
    comb_eval['_pred'] = preds
    ev_merged = ev[['sp', 'dt']].merge(
        comb_eval[['sp', 'dt', '_pred']], on=['sp', 'dt'], how='left'
    )
    final_preds = ev_merged['_pred'].fillna(0).values

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Done in {time.time() - t0:.1f}s")
    return final_preds


# ============================================================================
# Section 5c: CatBoost-Minimal (same features as LightGBM-Minimal)
# ============================================================================
def run_catboost_minimal(train, ev):
    """Train CatBoost with same minimal features as LightGBM ablation."""
    print("\n" + "=" * 70)
    print("SECTION 5c: CatBoost-Minimal")
    print("=" * 70)
    t0 = time.time()

    warmup = train['dt'].max() - pd.Timedelta(days=27)
    comb = pd.concat([train, ev], ignore_index=True)
    comb, fcols = build_minimal_features(comb)

    train_dates = set(train['dt'].unique())
    eval_dates = set(ev['dt'].unique())

    t_mask = comb['dt'].isin(train_dates) & (comb['dt'] >= warmup)
    e_mask = comb['dt'].isin(eval_dates)

    val_cutoff = comb.loc[t_mask, 'dt'].max() - pd.Timedelta(days=6)
    t_only = t_mask & (comb['dt'] < val_cutoff)
    v_mask = t_mask & (comb['dt'] >= val_cutoff)

    X_tr = comb.loc[t_only, fcols].fillna(0).values.astype(np.float32)
    y_tr = comb.loc[t_only, 'sale_amount'].values.astype(np.float32)
    X_val = comb.loc[v_mask, fcols].fillna(0).values.astype(np.float32)
    y_val = comb.loc[v_mask, 'sale_amount'].values.astype(np.float32)
    X_eval = comb.loc[e_mask, fcols].fillna(0).values.astype(np.float32)

    print(f"  Features: {len(fcols)}")
    print(f"  Train samples: {len(X_tr):,}, Val: {len(X_val):,}")

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        subsample=0.8,
        random_seed=42,
        loss_function='MAE',
        eval_metric='MAE',
        early_stopping_rounds=50,
        verbose=0,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=0,
    )

    preds = np.clip(model.predict(X_eval), 0, None)

    # Align back to ev index
    comb_eval = comb.loc[e_mask].copy()
    comb_eval['_pred'] = preds
    ev_merged = ev[['sp', 'dt']].merge(
        comb_eval[['sp', 'dt', '_pred']], on=['sp', 'dt'], how='left'
    )
    final_preds = ev_merged['_pred'].fillna(0).values

    print(f"  Best iteration: {model.get_best_iteration()}")
    print(f"  Done in {time.time() - t0:.1f}s")
    return final_preds


# ============================================================================
# Section 6: Load Full Pipeline Results
# ============================================================================
def load_full_pipeline_preds(ev):
    """Load predictions from the main pipeline output, or return None."""
    print("\n" + "=" * 70)
    print("SECTION 6: Loading Full Pipeline Results")
    print("=" * 70)

    # Check if we have saved predictions from the main pipeline
    forecast_json = os.path.join(OUTPUT_DIR, 'forecast_metrics.json')
    if os.path.exists(forecast_json):
        with open(forecast_json) as f:
            metrics = json.load(f)
        print(f"  Found forecast_metrics.json: ensemble MAE = {metrics.get('ensemble', {}).get('mae', 'N/A')}")
        print("  Note: Full pipeline predictions not stored as arrays — will skip direct comparison")
        print("  (Run the main pipeline to regenerate if needed)")
        return None
    else:
        print("  No forecast_metrics.json found. Run demand_forecast_and_inventory_optimization.py first.")
        return None


# ============================================================================
# Section 7 & 8: Unified Evaluation + Inventory Comparison
# ============================================================================
def run_evaluation(predictions, ev, train):
    """Evaluate all models on forecast metrics and inventory optimization."""
    print("\n" + "=" * 70)
    print("SECTION 7: Forecast Evaluation")
    print("=" * 70)

    y_true = ev['sale_amount'].values
    sp_eval = ev['sp'].values

    forecast_results = []
    inventory_results = []

    for name, preds in predictions.items():
        # Forecast metrics
        metrics = compute_metrics(y_true, preds, name)
        forecast_results.append(metrics)
        print(f"  {name:25s}  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"SMAPE={metrics['SMAPE(%)']:.1f}%  WMAE={metrics['WMAE']:.4f}  "
              f"Corr={metrics['Corr']:.4f}")

    print("\n" + "=" * 70)
    print("SECTION 8: Inventory Optimization (Empirical Newsvendor)")
    print("=" * 70)

    for name, preds in predictions.items():
        # Empirical Newsvendor: use historical per-SP demand quantile
        Q = run_empirical_newsvendor(preds, sp_eval, train, INV_CFG)
        inv_metrics = eval_policy(Q, y_true, INV_CFG, name)
        inventory_results.append(inv_metrics)
        print(f"  {name:25s}  Profit={inv_metrics['Avg Profit']:.4f}  "
              f"FR={inv_metrics['Fill Rate']:.3f}  "
              f"SL={inv_metrics['SL(Type1)']:.3f}  "
              f"Waste={inv_metrics['Waste%']:.1f}%")

    return pd.DataFrame(forecast_results), pd.DataFrame(inventory_results)


# ============================================================================
# Section 9: Visualization & Output
# ============================================================================
def create_comparison_plot(df_forecast, df_inventory):
    """Create multi-panel comparison chart."""
    print("\n" + "=" * 70)
    print("SECTION 9: Creating Comparison Plot")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    colors = ['#95a5a6', '#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#1abc9c']
    models = df_forecast['Model'].tolist()
    n = len(models)
    bar_colors = colors[:n]

    # Row 1: Forecast metrics
    for ax, metric, title in zip(
        axes[0],
        ['MAE', 'RMSE', 'WMAE'],
        ['MAE (lower is better)', 'RMSE (lower is better)', 'WMAE (lower is better)']
    ):
        vals = df_forecast[metric].values
        bars = ax.bar(range(n), vals, color=bar_colors, edgecolor='white', width=0.6)
        ax.set_xticks(range(n))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        # Annotate
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.4f}', ha='center', va='bottom', fontsize=7)

    # Row 2: Inventory metrics
    for ax, metric, title in zip(
        axes[1],
        ['Avg Profit', 'Fill Rate', 'Waste%'],
        ['Avg Profit (higher is better)', 'Fill Rate (higher is better)',
         'Waste % (lower is better)']
    ):
        vals = df_inventory[metric].values
        bars = ax.bar(range(n), vals, color=bar_colors, edgecolor='white', width=0.6)
        ax.set_xticks(range(n))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        fmt = '.4f' if metric == 'Avg Profit' else '.3f' if metric == 'Fill Rate' else '.1f'
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Benchmark Comparison: 6 Forecasting Models on FreshRetailNet-50K (3K SP sample)\n'
                 'Top row: Forecast accuracy  |  Bottom row: Inventory optimization (Empirical Newsvendor)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'benchmark_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("  BENCHMARK COMPARISON PIPELINE")
    print("  FreshRetailNet-50K: 7 Forecasting Approaches")
    print("=" * 70)
    t_start = time.time()

    # Load data
    train, ev = load_data()

    # Run all models
    predictions = {}

    predictions['Naive Seasonal'] = run_naive_seasonal(train, ev)
    predictions['SSA'] = run_ssa(train, ev)
    predictions['DLinear'] = run_dlinear(train, ev)
    predictions['LGBM-Minimal'] = run_lgbm_minimal(train, ev)
    predictions['XGBoost-Minimal'] = run_xgboost_minimal(train, ev)
    predictions['CatBoost-Minimal'] = run_catboost_minimal(train, ev)

    # Full pipeline (load if available)
    full_preds = load_full_pipeline_preds(ev)
    if full_preds is not None:
        predictions['Full Pipeline'] = full_preds

    # Evaluate all
    df_forecast, df_inventory = run_evaluation(predictions, ev, train)

    # Save results
    fc_path = os.path.join(OUTPUT_DIR, 'benchmark_forecast_comparison.csv')
    inv_path = os.path.join(OUTPUT_DIR, 'benchmark_inventory_comparison.csv')
    df_forecast.to_csv(fc_path, index=False)
    df_inventory.to_csv(inv_path, index=False)
    print(f"\n  Saved: {fc_path}")
    print(f"  Saved: {inv_path}")

    # Save per-SP predictions for statistical significance tests
    print("\n  Saving per-SP predictions for statistical tests...")
    y_true = ev['sale_amount'].values
    sp_eval = ev['sp'].values
    per_sp = pd.DataFrame({'sp': sp_eval, 'y_true': y_true})
    for name, preds in predictions.items():
        col_name = name.replace(' ', '_').replace('-', '_')
        per_sp[f'pred_{col_name}'] = preds
        per_sp[f'ae_{col_name}'] = np.abs(y_true - preds)
        # Compute per-row profit for each model
        Q = run_empirical_newsvendor(preds, sp_eval, train, INV_CFG)
        D = y_true
        sold = np.minimum(Q, D)
        over = np.maximum(Q - D, 0)
        under = np.maximum(D - Q, 0)
        profit = (sold * INV_CFG['unit_revenue']
                  - Q * INV_CFG['unit_cost']
                  - INV_CFG['h'] * over
                  - INV_CFG['p'] * under
                  - INV_CFG['w'] * over)
        per_sp[f'profit_{col_name}'] = profit
    per_sp_path = os.path.join(OUTPUT_DIR, 'benchmark_per_sp_predictions.csv')
    per_sp.to_csv(per_sp_path, index=False)
    print(f"  Saved: {per_sp_path}")

    # Plot
    create_comparison_plot(df_forecast, df_inventory)

    # Summary
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print("\nForecast Metrics:")
    print(df_forecast.to_string(index=False))
    print("\nInventory Metrics (Empirical Newsvendor):")
    print(df_inventory.to_string(index=False))

    best_fc = df_forecast.loc[df_forecast['MAE'].idxmin(), 'Model']
    best_inv = df_inventory.loc[df_inventory['Avg Profit'].idxmax(), 'Model']
    print(f"\n  Best forecast (MAE):     {best_fc}")
    print(f"  Best inventory (Profit): {best_inv}")
    print(f"\n  Total runtime: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()

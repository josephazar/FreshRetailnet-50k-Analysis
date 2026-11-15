"""Leakage-free feature construction with fit/transform separation.

``FeatureBuilder.fit`` learns aggregate lookups from the training window only,
and ``FeatureBuilder.transform`` applies those fixed lookups to later rows.
Evaluation-time lag and rolling features are generated recursively: each day's
prediction becomes history for the next day, so held-out sales never enter
the model inputs. Held-out stock status has no prediction to substitute, so
the stockout-history features are held at the forecast origin instead (see
``hold_stockout_history_at_origin``).
"""

from __future__ import annotations

import gc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class FeatureBuilder:
    """Stateful feature builder. Fit on training data, transform on any frame."""

    DEFAULT_LAGS = [1, 2, 3, 5, 7, 14, 21, 28]
    ROLLING_WINDOWS = [3, 7, 14, 28]
    EWMA_SPANS = [7, 14]
    N_CLUSTERS_DEFAULT = 50

    def __init__(self, n_clusters: int = N_CLUSTERS_DEFAULT, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed = seed
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        """Learn all aggregates from training data only."""
        df = train_df.copy()
        if 'cens' not in df.columns:
            df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')

        # ---- Per-group means / stds (sp, product_id, store_id, city_id) ----
        self.group_stats_ = {}
        for grp in ['sp', 'product_id', 'store_id', 'city_id']:
            m = df.groupby(grp)['sale_amount'].mean()
            s = df.groupby(grp)['sale_amount'].std().fillna(0)
            self.group_stats_[grp] = {'m': m.to_dict(), 's': s.to_dict()}

        # ---- Category levels (cat1, cat2, cat3) ----
        self.cat_stats_ = {}
        for cat in ['first_category_id', 'second_category_id', 'third_category_id']:
            m = df.groupby(cat)['sale_amount'].mean()
            s = df.groupby(cat)['sale_amount'].std().fillna(0)
            self.cat_stats_[cat] = {'m': m.to_dict(), 's': s.to_dict()}

        # ---- DOW profile per SP (training-only mean by (sp, dow)) ----
        dow = df['dt'].dt.dayofweek.astype('int8')
        self.dow_profile_ = df.assign(_dow=dow).groupby(['sp', '_dow'])['sale_amount'].mean().to_dict()
        self.dow_profile_r_ = None  # filled if dem_rec present
        if 'dem_rec' in df.columns:
            self.dow_profile_r_ = df.assign(_dow=dow).groupby(['sp', '_dow'])['dem_rec'].mean().to_dict()

        # ---- Store-level day-of-week aggregates from training history ----
        store_dt_tot = df.groupby(['store_id', 'dt'])['sale_amount'].sum().reset_index()
        store_dt_tot['_dow'] = store_dt_tot['dt'].dt.dayofweek
        self.store_dow_vol_ = store_dt_tot.groupby(['store_id', '_dow'])['sale_amount'].mean().to_dict()

        self.store_so_rate_ = df.groupby('store_id')['cens'].mean().to_dict()

        # ---- Category-level day-of-week aggregates from training history ----
        cat2_dt = df.groupby(['second_category_id', 'dt'])['sale_amount'].mean().reset_index()
        cat2_dt['_dow'] = cat2_dt['dt'].dt.dayofweek
        self.cat2_dow_m_ = cat2_dt.groupby(['second_category_id', '_dow'])['sale_amount'].mean().to_dict()

        # ---- Cluster model fit on training-only SP behavior ----
        sp_behavior = df.groupby('sp').agg(
            mean_demand=('sale_amount', 'mean'),
            std_demand=('sale_amount', 'std'),
            so_rate=('cens', 'mean'),
            zero_rate=('sale_amount', lambda x: (x == 0).mean()),
        ).fillna(0)
        sp_behavior['cv'] = sp_behavior['std_demand'] / sp_behavior['mean_demand'].clip(lower=0.01)

        # Weekend ratio from training only
        is_wknd = (df['dt'].dt.dayofweek >= 5).astype(float)
        wknd_m = df[is_wknd == 1].groupby('sp')['sale_amount'].mean()
        wkday_m = df[is_wknd == 0].groupby('sp')['sale_amount'].mean()
        sp_behavior['wknd_ratio'] = (wknd_m / wkday_m.clip(lower=0.01)).reindex(sp_behavior.index).fillna(1.0)

        self._cluster_scaler = StandardScaler()
        sp_feat_scaled = self._cluster_scaler.fit_transform(sp_behavior.values)
        self._cluster_model = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        self._cluster_model.fit(sp_feat_scaled)
        sp_behavior['cluster_id'] = self._cluster_model.labels_
        self.sp_cluster_map_ = sp_behavior['cluster_id'].to_dict()

        # Cluster-level demand stats (from training)
        df_with_cluster = df.assign(cluster_id=df['sp'].map(self.sp_cluster_map_))
        cluster_stats = df_with_cluster.groupby('cluster_id')['sale_amount'].agg(['mean', 'std']).fillna(0)
        self.cluster_m_ = cluster_stats['mean'].to_dict()
        self.cluster_s_ = cluster_stats['std'].to_dict()

        # ---- Per-SP temperature/humidity means (for deviation features) ----
        self.sp_temp_mean_ = df.groupby('sp')['avg_temperature'].mean().to_dict()
        self.sp_hum_mean_ = df.groupby('sp')['avg_humidity'].mean().to_dict()

        # ---- Store size info ----
        self.store_size_ = df.groupby('store_id')['sp'].nunique().to_dict()

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Lag / rolling / EWMA / calendar features (computed per-frame, SAFE
    # as long as the input frame's sale_amount does not contain leaked
    # future actuals. Callers responsible for masking eval actuals.)
    # ------------------------------------------------------------------
    @staticmethod
    def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
        df['dow'] = df['dt'].dt.dayofweek.astype('int8')
        df['dom'] = df['dt'].dt.day.astype('int8')
        df['woy'] = df['dt'].dt.isocalendar().week.astype('int8')
        df['month'] = df['dt'].dt.month.astype('int8')
        df['wknd'] = (df['dow'] >= 5).astype('int8')
        df['doy'] = df['dt'].dt.dayofyear.astype('int16')
        for cyc, period in [('dow', 7), ('dom', 31), ('woy', 52)]:
            df[f'{cyc}_sin'] = np.sin(2 * np.pi * df[cyc] / period).astype('float32')
            df[f'{cyc}_cos'] = np.cos(2 * np.pi * df[cyc] / period).astype('float32')
        return df

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby('sp')
        for lag in self.DEFAULT_LAGS:
            df[f's_l{lag}'] = g['sale_amount'].shift(lag).astype('float32')
            if 'dem_rec' in df.columns:
                df[f'r_l{lag}'] = g['dem_rec'].shift(lag).astype('float32')
        df['s_d1'] = (df['s_l1'] - df['s_l2']).astype('float32')
        df['s_d7'] = (df['s_l1'] - df['s_l7']).astype('float32')
        if 'dem_rec' in df.columns:
            df['r_d1'] = (df['r_l1'] - g['dem_rec'].shift(2)).astype('float32')
        return df

    def _add_rolling(self, df: pd.DataFrame) -> pd.DataFrame:
        shifted_s = df.groupby('sp')['sale_amount'].shift(1)
        has_r = 'dem_rec' in df.columns
        shifted_r = df.groupby('sp')['dem_rec'].shift(1) if has_r else None
        pairs = [('s', shifted_s)] + ([('r', shifted_r)] if has_r else [])
        for pfx, shifted in pairs:
            for w in self.ROLLING_WINDOWS:
                r = shifted.groupby(df['sp']).rolling(w, min_periods=1)
                df[f'{pfx}_m{w}'] = r.mean().reset_index(level=0, drop=True).astype('float32')
                df[f'{pfx}_sd{w}'] = r.std().reset_index(level=0, drop=True).astype('float32')
                if w in [7, 28]:
                    df[f'{pfx}_mx{w}'] = r.max().reset_index(level=0, drop=True).astype('float32')
                    df[f'{pfx}_mn{w}'] = r.min().reset_index(level=0, drop=True).astype('float32')
                    df[f'{pfx}_md{w}'] = r.median().reset_index(level=0, drop=True).astype('float32')
        # Quantiles
        for w in [14, 28]:
            r = shifted_s.groupby(df['sp']).rolling(w, min_periods=1)
            df[f's_q25_{w}'] = r.quantile(0.25).reset_index(level=0, drop=True).astype('float32')
            df[f's_q75_{w}'] = r.quantile(0.75).reset_index(level=0, drop=True).astype('float32')
        # EWMA
        for pfx, shifted in pairs:
            for s in self.EWMA_SPANS:
                df[f'{pfx}_ew{s}'] = shifted.groupby(df['sp']).transform(
                    lambda x: x.ewm(span=s, min_periods=1).mean()
                ).astype('float32')
        return df

    @staticmethod
    def _add_stockout(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby('sp')
        cs = g['cens'].shift(1)
        df['so1'] = cs.astype('float32')
        df['so7'] = g['cens'].shift(7).astype('float32')
        for w in [7, 14]:
            df[f'sor{w}'] = cs.groupby(df['sp']).rolling(w, min_periods=1).mean() \
                .reset_index(level=0, drop=True).astype('float32')
        hs = g['stock_hour6_22_cnt'].shift(1)
        for w in [7, 14]:
            df[f'soh{w}'] = hs.groupby(df['sp']).rolling(w, min_periods=1).mean() \
                .reset_index(level=0, drop=True).astype('float32')
        return df

    def _add_global_stats_from_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lookup-only aggregates from fit(). No groupby on current frame."""
        for grp, stats in self.group_stats_.items():
            pfx = {'sp': 'sp', 'product_id': 'pd', 'store_id': 'st', 'city_id': 'ct'}[grp]
            df[f'{pfx}_m'] = df[grp].map(stats['m']).fillna(0).astype('float32')
            df[f'{pfx}_s'] = df[grp].map(stats['s']).fillna(0).astype('float32')
        for cat, stats in self.cat_stats_.items():
            pfx = {'first_category_id': 'cat1', 'second_category_id': 'cat2',
                   'third_category_id': 'cat3'}[cat]
            df[f'{pfx}_m'] = df[cat].map(stats['m']).fillna(0).astype('float32')
            if cat == 'third_category_id':
                df['cat3_s'] = df['third_category_id'].map(stats['s']).fillna(0).astype('float32')
        # DOW profile — need (sp, dow) lookup
        key_df = list(zip(df['sp'].values, df['dow'].astype(int).values))
        df['dow_prof'] = pd.Series([self.dow_profile_.get(k, 0.0) for k in key_df],
                                    index=df.index).astype('float32')
        if self.dow_profile_r_ is not None:
            df['dow_prof_r'] = pd.Series([self.dow_profile_r_.get(k, 0.0) for k in key_df],
                                          index=df.index).astype('float32')
        # Store DOW volume
        key_sdow = list(zip(df['store_id'].values, df['dow'].astype(int).values))
        df['store_daily_vol'] = pd.Series([self.store_dow_vol_.get(k, 0.0) for k in key_sdow],
                                           index=df.index).astype('float32')
        df['store_so_rate'] = df['store_id'].map(self.store_so_rate_).fillna(0).astype('float32')
        # Cat2 DOW mean
        key_cat2_dow = list(zip(df['second_category_id'].values, df['dow'].astype(int).values))
        df['cat2_daily_m'] = pd.Series([self.cat2_dow_m_.get(k, 0.0) for k in key_cat2_dow],
                                        index=df.index).astype('float32')
        # Hierarchy ratios
        df['sp_vs_cat2'] = (df['sp_m'] / df['cat2_m'].clip(lower=0.01)).astype('float32')
        df['sp_vs_cat3'] = (df['sp_m'] / df['cat3_m'].clip(lower=0.01)).astype('float32')
        df['sp_vs_store'] = (df['sp_m'] / df['st_m'].clip(lower=0.01)).astype('float32')
        # Cluster
        df['cluster_id'] = df['sp'].map(self.sp_cluster_map_).fillna(0).astype('int8')
        df['cluster_m'] = df['cluster_id'].map(self.cluster_m_).fillna(0).astype('float32')
        df['cluster_s'] = df['cluster_id'].map(self.cluster_s_).fillna(0).astype('float32')
        # Temp / humidity deviation from SP mean (train-only)
        df['temp_dev'] = (df['avg_temperature']
                          - df['sp'].map(self.sp_temp_mean_).fillna(0)).astype('float32')
        df['hum_dev'] = (df['avg_humidity']
                         - df['sp'].map(self.sp_hum_mean_).fillna(0)).astype('float32')
        return df

    @staticmethod
    def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
        # Variability and trend
        df['cv7'] = (df['s_sd7'] / df['s_m7'].clip(lower=0.01)).astype('float32')
        df['cv28'] = (df['s_sd28'] / df['s_m28'].clip(lower=0.01)).astype('float32')
        df['tr_7_28'] = (df['s_m7'] - df['s_m28']).astype('float32')
        df['tr_3_14'] = (df['s_m3'] - df['s_m14']).astype('float32')
        df['l1_m7'] = (df['s_l1'] / df['s_m7'].clip(lower=0.01)).astype('float32')
        df['m7_m28'] = (df['s_m7'] / df['s_m28'].clip(lower=0.01)).astype('float32')
        if 'r_m7' in df.columns:
            df['rec_obs_ratio'] = (df['r_m7'] / df['s_m7'].clip(lower=0.01)).astype('float32')
        # Cross features
        df['d_h'] = (df['discount'] * df['holiday_flag']).astype('float32')
        df['d_a'] = (df['discount'] * df['activity_flag']).astype('float32')
        df['t_p'] = (df['avg_temperature'] * df['precpt']).astype('float32')
        df['w_h'] = (df['wknd'] * df['holiday_flag']).astype('int8')
        df['h_a'] = (df['holiday_flag'] * df['activity_flag']).astype('int8')
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features on the given frame (SPxDT panel).

        Caller's responsibility: ensure df['sale_amount'] and df['dem_rec'] for
        eval rows do NOT contain leaked future actuals. For inference, either
        pass only rows whose actuals are from strict past, or mask eval actuals
        to NaN and rely on recursive prediction (see predict_eval_recursive).
        Eval-row stock status still feeds the stockout-history features; the
        predict_eval_* helpers hold those at the origin
        (hold_stockout_history_at_origin).
        """
        import warnings as _warnings
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.fit() must be called before transform().")
        df = df.sort_values(['sp', 'dt']).reset_index(drop=True)
        # Ensure required auxiliary columns exist for downstream steps
        if 'cens' not in df.columns:
            df['cens'] = (df['stock_hour6_22_cnt'] > 0).astype('int8')
        if 'dem_rec' not in df.columns:
            df['dem_rec'] = df['sale_amount'].astype('float32')
        df = self._add_temporal(df)
        df = self._add_lags(df)
        df = self._add_rolling(df)
        df = self._add_stockout(df)
        with _warnings.catch_warnings():
            _warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
            df = self._add_global_stats_from_fit(df)
            df = self._add_derived(df)
        # Defragment
        df = df.copy()
        df = df.fillna(0)
        return df

    # ------------------------------------------------------------------
    # Column helper
    # ------------------------------------------------------------------
    @staticmethod
    def feature_cols(df: pd.DataFrame) -> list:
        excl = {'dt', 'sale_amount', 'dem_rec', 'sp', 'cens', 'so_frac',
                'city_id', 'store_id', 'management_group_id',
                'first_category_id', 'second_category_id', 'third_category_id',
                'product_id', 'stock_hour6_22_cnt'}
        return [c for c in df.columns if c not in excl]


# Stockout-history features built by FeatureBuilder._add_stockout, each mapped
# to the most recent lag (in days) it reads.
STOCKOUT_HISTORY_MIN_LAG = {'so1': 1, 'so7': 7, 'sor7': 1, 'sor14': 1,
                            'soh7': 1, 'soh14': 1}


def hold_stockout_history_at_origin(feat, day_mask, sp_day, day_idx, origin):
    """Keep realised evaluation-day stock status out of the model inputs.

    Stock status is observed during each evaluation day, so at the forecast
    origin it is as unknown as that day's sales. A stockout-history feature
    whose most recent lag reaches an evaluation day (day_idx >= lag, where
    day_idx is 0 on the first forecast day) is held at its first-forecast-day
    value, which reads training days only. predict_eval_direct applies the
    same rule to its whole history block.

    Modifies ``feat`` in place on the ``day_mask`` rows and returns the
    origin values, which are captured when day_idx == 0.
    """
    cols = list(STOCKOUT_HISTORY_MIN_LAG)
    if day_idx == 0:
        return feat.loc[day_mask, ['sp'] + cols].set_index('sp')
    held = [c for c, lag in STOCKOUT_HISTORY_MIN_LAG.items() if day_idx >= lag]
    if held:
        feat.loc[day_mask, held] = origin.reindex(sp_day)[held].to_numpy(dtype='float32')
    return origin


def predict_eval_recursive(
    fbuilder: FeatureBuilder,
    models: dict,
    fcols: list,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict:
    """Leakage-free recursive prediction over the eval window.

    For each eval day in ascending order:
      1. Build features on pd.concat([train, predicted_eval_so_far]) where
         predicted_eval_so_far has sale_amount filled with PREDICTIONS
         (not actuals) for all prior eval days.
      2. Slice to current eval day and hold its stockout-history features at
         the origin (eval-day stock status is realised, not known at T).
      3. Predict using each model.
      4. Write predictions into sale_amount slot (so next day's lag_1 is a
         prediction, not a leaked actual).

    Returns
    -------
    dict keyed by model name -> np.ndarray of predictions, aligned to
    eval_df.sort_values(['sp','dt']).
    """
    if not fbuilder._fitted:
        raise RuntimeError("FeatureBuilder must be fit first.")

    # Keep only the last 28 days of train as history (enough for all lags)
    hist = train_df[train_df['dt'] >= train_df['dt'].max() - pd.Timedelta(days=27)].copy()
    hist['cens'] = (hist['stock_hour6_22_cnt'] > 0).astype('int8')
    if 'dem_rec' not in hist.columns:
        hist['dem_rec'] = hist['sale_amount'].astype('float32')

    eval_sorted = eval_df.sort_values(['sp', 'dt']).reset_index(drop=True)
    eval_sorted['cens'] = (eval_sorted['stock_hour6_22_cnt'] > 0).astype('int8')
    eval_sorted['so_frac'] = (eval_sorted['stock_hour6_22_cnt'] / 16).astype('float32')
    # CRITICAL: zero out sale_amount/dem_rec; we'll fill predictions day by day
    eval_dates = sorted(eval_sorted['dt'].unique())
    eval_work = eval_sorted.copy()
    eval_work['sale_amount'] = 0.0  # placeholder, will be filled with predictions
    eval_work['dem_rec'] = 0.0

    predictions = {name: np.zeros(len(eval_sorted), dtype='float32') for name in models}

    origin_so = None
    for day_idx, day in enumerate(eval_dates):
        # Combine history with predicted-so-far eval days
        so_far_mask = eval_work['dt'] < day
        combined = pd.concat([hist, eval_work[so_far_mask | (eval_work['dt'] == day)]],
                             ignore_index=True)
        feat = fbuilder.transform(combined)
        today_mask = feat['dt'] == day
        sp_today = feat.loc[today_mask, 'sp'].values
        origin_so = hold_stockout_history_at_origin(
            feat, today_mask, sp_today, day_idx, origin_so)
        X_today = feat.loc[today_mask, fcols].fillna(0).values

        # Predict with each model
        today_preds = {}
        for name, mdl in models.items():
            today_preds[name] = np.clip(mdl.predict(X_today), 0, None).astype('float32')

        # Use ensemble (mae + huber + q50) / 3 as the value written back for lag features
        ens_source = ['mae', 'huber', 'q50']
        avail = [n for n in ens_source if n in today_preds]
        if avail:
            write_back = np.mean([today_preds[n] for n in avail], axis=0)
        else:
            # fallback to first model
            write_back = next(iter(today_preds.values()))

        # Write predictions back into eval_work for next day's lag computation
        day_rows = eval_work[eval_work['dt'] == day].index
        # Map: sp_today array is already sorted by sp within that day
        mapper = pd.Series(write_back, index=sp_today)
        eval_work.loc[day_rows, 'sale_amount'] = eval_work.loc[day_rows, 'sp'].map(mapper).values
        eval_work.loc[day_rows, 'dem_rec'] = eval_work.loc[day_rows, 'sale_amount']

        # Store per-model predictions in aligned position
        day_mask_in_eval_sorted = (eval_sorted['dt'] == day).values
        sp_in_eval = eval_sorted.loc[day_mask_in_eval_sorted, 'sp'].values
        for name, p in today_preds.items():
            # Map sp -> pred, then assign to eval_sorted positions
            pm = pd.Series(p, index=sp_today)
            predictions[name][day_mask_in_eval_sorted] = eval_sorted.loc[
                day_mask_in_eval_sorted, 'sp'
            ].map(pm).values

    gc.collect()
    return predictions, eval_sorted


def predict_eval_single_origin(
    fbuilder: FeatureBuilder,
    models: dict,
    fcols: list,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> dict:
    """Simpler single-origin alternative to recursive prediction.

    Features for ALL eval days are built using only training history (eval
    sale_amount set to 0 before computing lags; stockout-history features held
    at the origin, as in predict_eval_recursive). Each eval day gets features
    whose lag_k refers back exactly k days -- for day 1 that's the last
    training day (correct); for day 7 that's 7 days before last training
    (legit, and what lag_7 always was).

    The drawback vs recursive: lag_1 for eval day 2 is 0 instead of the
    day-1 prediction, which loses some signal. Use this when fast inference
    is needed and recursive is too slow.
    """
    hist = train_df[train_df['dt'] >= train_df['dt'].max() - pd.Timedelta(days=27)].copy()
    hist['cens'] = (hist['stock_hour6_22_cnt'] > 0).astype('int8')
    if 'dem_rec' not in hist.columns:
        hist['dem_rec'] = hist['sale_amount'].astype('float32')

    ev = eval_df.sort_values(['sp', 'dt']).reset_index(drop=True).copy()
    ev['cens'] = (ev['stock_hour6_22_cnt'] > 0).astype('int8')
    ev['so_frac'] = (ev['stock_hour6_22_cnt'] / 16).astype('float32')
    ev['sale_amount'] = 0.0
    ev['dem_rec'] = 0.0

    combined = pd.concat([hist, ev], ignore_index=True)
    feat = fbuilder.transform(combined)
    origin_so = None
    for day_idx, day in enumerate(sorted(ev['dt'].unique())):
        day_mask = feat['dt'] == day
        origin_so = hold_stockout_history_at_origin(
            feat, day_mask, feat.loc[day_mask, 'sp'].values, day_idx, origin_so)
    eval_mask = feat['dt'].isin(ev['dt'].unique())
    X_eval = feat.loc[eval_mask, fcols].fillna(0).values
    sp_eval = feat.loc[eval_mask, 'sp'].values

    predictions = {}
    for name, mdl in models.items():
        p = np.clip(mdl.predict(X_eval), 0, None).astype('float32')
        predictions[name] = p

    return predictions, feat.loc[eval_mask].reset_index(drop=True), sp_eval

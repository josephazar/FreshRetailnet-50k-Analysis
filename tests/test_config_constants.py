"""Pin the LightGBM configuration cited in the manuscript to a single source.

research/paper.tex (Model architecture) states these exact values; if this
test fails, either the paper or the constants must be updated together.
"""
import sys, os, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline_leakfree as pl


PAPER_CITED_PARAMS = {
    'num_leaves': 255,
    'learning_rate': 0.03,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'seed': 42,
}


def test_lgbm_params_match_paper_cited_values():
    for key, val in PAPER_CITED_PARAMS.items():
        assert pl.LGBM_BASE_PARAMS[key] == val, (
            f"LGBM_BASE_PARAMS[{key!r}] = {pl.LGBM_BASE_PARAMS[key]!r} "
            f"but the paper cites {val!r}"
        )
    assert 'max_depth' not in pl.LGBM_BASE_PARAMS, (
        "paper states no explicit depth limit"
    )
    assert set(pl.LGBM_OBJECTIVES) == {'mae', 'huber', 'q10', 'q50', 'q90'}


def test_train_and_retrain_share_single_param_source():
    cv_src = inspect.getsource(pl.train_cv)
    rt_src = inspect.getsource(pl.retrain_final)
    for src, fn in [(cv_src, 'train_cv'), (rt_src, 'retrain_final')]:
        assert 'LGBM_BASE_PARAMS' in src and 'LGBM_OBJECTIVES' in src, (
            f"{fn} must consume the module-level constants"
        )
        assert "'num_leaves':" not in src, (
            f"{fn} re-declares LightGBM params inline instead of using "
            f"LGBM_BASE_PARAMS"
        )

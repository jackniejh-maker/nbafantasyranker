"""Shared model zoo.

Every learned model is a *residual learner on top of a transparent baseline*:

    y_hat = baseline + sd * f(X),     f fitted on (y - baseline) / sd

(baseline = recency-weighted history for FPPM / minutes / availability).  If the features carry no
out-of-sample signal, strong regularisation shrinks f to ~0 and the model falls back to the baseline - it
cannot do much worse than the benchmark, which keeps Ridge / Elastic Net / GBM comparable and stable.

Hyper-parameters are chosen by an inner *chronological* CV (expanding window over seasons) on the
training seasons only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import log

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class FeatureCleaner(BaseEstimator, TransformerMixin):
    """Drops empty / constant columns, winsorises, median-imputes; returns an ndarray.

    Columns are matched BY NAME at transform time, so feature frames with extra / missing columns
    (e.g. a coach feature that is entirely NaN in one season) are handled consistently."""

    def __init__(self, winsor: float = 0.005):
        self.winsor = winsor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_in_ = list(X.columns)
        v = X.to_numpy(float)
        v = np.where(np.isfinite(v), v, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nn = np.isfinite(v).mean(0)
            sd = np.nanstd(v, 0)
            self.keep_ = (nn >= 0.05) & np.isfinite(sd) & (sd > 1e-9)
            vk = v[:, self.keep_]
            self.median_ = np.nanmedian(vk, 0)
            self.lo_ = np.nanquantile(vk, self.winsor, axis=0)
            self.hi_ = np.nanquantile(vk, 1 - self.winsor, axis=0)
        self.columns_ = [c for c, k in zip(self.columns_in_, self.keep_) if k]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reindex(columns=self.columns_in_)
        v = X.to_numpy(float)[:, self.keep_]
        v = np.where(np.isfinite(v), v, self.median_)
        return np.clip(v, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


class GBMRegressor(BaseEstimator, RegressorMixin):
    """LightGBM / XGBoost when installed, otherwise scikit-learn's HistGradientBoostingRegressor."""

    def __init__(self, backend="auto", max_depth=3, n_estimators=150, learning_rate=0.04, min_leaf=25,
                 random_state=0):
        self.backend = backend
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_leaf = min_leaf
        self.random_state = random_state

    def _build(self):
        be = self.backend
        if be in ("auto", "lightgbm"):
            try:
                import lightgbm as lgb
                return "lightgbm", lgb.LGBMRegressor(
                    n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
                    num_leaves=max(4, 2 ** self.max_depth - 1), min_child_samples=self.min_leaf, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=self.random_state,
                    verbose=-1, n_jobs=1)
            except ImportError:
                if be == "lightgbm":
                    raise
        if be in ("auto", "xgboost"):
            try:
                import xgboost as xgb
                return "xgboost", xgb.XGBRegressor(
                    n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
                    min_child_weight=self.min_leaf / 5.0, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                    random_state=self.random_state, n_jobs=1, verbosity=0)
            except ImportError:
                if be == "xgboost":
                    raise
        from sklearn.ensemble import HistGradientBoostingRegressor
        return "sklearn_hist", HistGradientBoostingRegressor(
            max_depth=self.max_depth, max_iter=self.n_estimators, learning_rate=self.learning_rate,
            min_samples_leaf=self.min_leaf, l2_regularization=5.0, random_state=self.random_state)

    def fit(self, X, y, sample_weight=None):
        self.backend_, self.model_ = self._build()
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        imp = getattr(self.model_, "feature_importances_", None)
        if imp is None:
            raise AttributeError("backend has no feature_importances_")
        return imp


def make_pipeline(kind: str, cfg) -> Pipeline:
    clean = FeatureCleaner(float(cfg.get("models.winsor_quantile", 0.005)))
    if kind == "ridge":
        return Pipeline([("clean", clean), ("scale", StandardScaler()), ("model", Ridge(alpha=100.0))])
    if kind == "elastic_net":
        return Pipeline([("clean", clean), ("scale", StandardScaler()),
                         ("model", ElasticNet(alpha=0.03, l1_ratio=0.6, max_iter=20000))])
    if kind == "gbm":
        return Pipeline([("clean", clean), ("model", GBMRegressor(
            backend=cfg.get("models.gbm_backend", "auto"), random_state=int(cfg.get("models.random_state", 0))))])
    raise ValueError(f"unknown model kind {kind!r}")


def param_grid(kind: str, cfg):
    m = cfg.get("models")
    if kind == "ridge":
        return {"model__alpha": list(m["ridge_alphas"])}
    if kind == "elastic_net":
        return {"model__alpha": list(m["enet_alphas"]), "model__l1_ratio": list(m["enet_l1_ratios"])}
    if kind == "gbm":
        return [{"model__max_depth": [g["max_depth"]], "model__n_estimators": [g["n_estimators"]],
                 "model__learning_rate": [g["learning_rate"]]} for g in m["gbm_grid"]]
    return None


def season_splits(years, min_train: int = 3):
    """Expanding-window chronological splits: train on seasons < s, validate on season s."""
    years = np.asarray(years)
    uniq = np.sort(np.unique(years))
    out = []
    for i in range(min_train, len(uniq)):
        out.append((np.flatnonzero(years < uniq[i]), np.flatnonzero(years == uniq[i])))
    return out


class TargetModel:
    """baseline + sd * f(X); `kind` in {'baseline', 'ridge', 'elastic_net', 'gbm'}."""

    def __init__(self, kind: str, cfg):
        self.kind = kind
        self.cfg = cfg
        self.est_ = None
        self.y_sd_ = 1.0
        self.best_params_: dict = {}

    def fit(self, X: pd.DataFrame, y, base, w=None, years=None) -> "TargetModel":
        y, base = np.asarray(y, float), np.asarray(base, float)
        if self.kind == "baseline":
            return self
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        resid = y - base
        mu = np.average(resid, weights=w)
        self.y_sd_ = float(np.sqrt(np.average((resid - mu) ** 2, weights=w))) or 1.0
        r = resid / self.y_sd_
        pipe, grid = make_pipeline(self.kind, self.cfg), param_grid(self.kind, self.cfg)
        splits = season_splits(years, int(self.cfg.get("validation.inner_cv_min_seasons", 3))) if years is not None else []
        if splits and grid:
            gs = GridSearchCV(pipe, grid, cv=splits, scoring="neg_mean_squared_error", refit=True, n_jobs=1)
            gs.fit(X, r, model__sample_weight=w)
            self.est_, self.best_params_ = gs.best_estimator_, {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        else:
            pipe.fit(X, r, model__sample_weight=w)
            self.est_ = pipe
        return self

    def adjustment(self, X: pd.DataFrame) -> np.ndarray:
        if self.est_ is None:
            return np.zeros(len(X))
        return self.y_sd_ * self.est_.predict(X)

    def predict(self, X: pd.DataFrame, base) -> np.ndarray:
        return np.asarray(base, float) + self.adjustment(X)

    @property
    def is_linear(self) -> bool:
        return self.kind in ("ridge", "elastic_net")


def cv_summary(kind: str, tm: TargetModel) -> str:
    return f"{kind}: {tm.best_params_}" if tm.best_params_ else kind

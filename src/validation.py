"""Strictly chronological validation (expanding window) and model selection.

For every test season S:   train on target seasons < S   ->   predict S.
There is never a random split; hyper-parameters are tuned by an inner chronological CV that only sees
the training seasons.  Metrics: RMSE, MAE, R^2, Pearson, Spearman, MAPE, plus the error among the top-N
projected players (what a fantasy drafter actually cares about).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance

from .feature_engineering import TrainingTable
from .models import TargetModel
from .utils import log

METRIC_COLS = ["rmse", "mae", "r2", "pearson", "spearman", "mape", "top_n_mae", "top_n_rmse"]
HIGHER_BETTER = {"r2", "pearson", "spearman"}
COMPLEXITY_ORDER = ["baseline", "ridge", "elastic_net", "gbm"]
MODEL_LABELS = {"baseline": "A_weighted_average", "fixed": "A0_fixed_weights", "ridge": "B_ridge",
                "elastic_net": "C_elastic_net", "gbm": "D_gradient_boosting"}


@dataclass
class TargetSpec:
    name: str
    y_col: str
    base_col: str
    elig_col: str
    weight_fn: Callable
    alt_col: str | None = None
    alt_label: str = "A0_fixed_weights"
    feature_cols: list | None = None
    min_rows: int = 40


@dataclass
class TargetResult:
    spec: TargetSpec
    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    selected: str
    final: TargetModel
    explain: TargetModel
    all_final: dict
    fold_models: dict = field(repr=False, default_factory=dict)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)


# ====================================================================== metrics
def regression_metrics(y, p, top_n: int = 100) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    out = {"n": int(len(y)), **{k: np.nan for k in METRIC_COLS}}
    if len(y) < 3:
        return out
    err = p - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae"] = float(np.mean(np.abs(err)))
    out["r2"] = float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
    if y.std() > 0 and p.std() > 0:
        out["pearson"] = float(np.corrcoef(y, p)[0, 1])
        out["spearman"] = float(stats.spearmanr(y, p).statistic)
    pos = y > 0
    out["mape"] = float(np.mean(np.abs(err[pos]) / y[pos])) if pos.any() else np.nan
    top = np.argsort(-p)[: min(top_n, len(p))]
    out["top_n_mae"] = float(np.mean(np.abs(err[top])))
    out["top_n_rmse"] = float(np.sqrt(np.mean(err[top] ** 2)))
    return out


def test_years_for(years, cfg) -> list[int]:
    ys = sorted(set(int(y) for y in years))
    n_test = int(cfg.get("validation.n_test_seasons", 5))
    min_train = int(cfg.get("validation.min_train_seasons", 4))
    n_test = min(n_test, len(ys))
    while n_test > 0 and len([y for y in ys if y < ys[-n_test]]) < min_train:
        n_test -= 1
    if n_test <= 0:
        raise ValueError(f"need at least {min_train + 1} seasons of features for chronological validation; got {ys}")
    return ys[-n_test:]


# ====================================================================== walk-forward
def walk_forward(tab: TrainingTable, spec: TargetSpec, kinds, test_years, cfg):
    X = tab.X if spec.feature_cols is None else tab.X[[c for c in spec.feature_cols if c in tab.X.columns]]
    meta, Y = tab.meta, tab.y
    years = meta["year"].to_numpy()
    yv = Y[spec.y_col].to_numpy(float)
    base = meta[spec.base_col].to_numpy(float)
    elig = Y[spec.elig_col].to_numpy(bool)
    w = np.asarray(spec.weight_fn(Y, cfg), float)
    parts, fold_models = [], {}
    for T in test_years:
        tr, te = elig & (years < T), years == T
        out = {"year": T, "player_id": meta["player_id"].to_numpy()[te], "y": yv[te], "base": base[te],
               "elig": elig[te], "y_min": Y["y_min"].to_numpy()[te]}
        for kind in kinds:
            if kind == "baseline":
                out["pred_baseline"] = base[te]
            elif kind == "fixed":
                out["pred_fixed"] = meta[spec.alt_col].to_numpy(float)[te] if spec.alt_col else base[te]
            elif tr.sum() < spec.min_rows:
                out[f"pred_{kind}"] = base[te]
            else:
                m = TargetModel(kind, cfg).fit(X[tr], yv[tr], base[tr], w[tr], years[tr])
                out[f"pred_{kind}"] = m.predict(X[te], base[te])
                fold_models[(T, kind)] = m
        parts.append(pd.DataFrame(out))
    return pd.concat(parts, ignore_index=True), fold_models


def fold_metric_table(oof: pd.DataFrame, kinds, spec: TargetSpec, cfg) -> pd.DataFrame:
    top_n = int(cfg.get("validation.top_n", 100))
    min_min = float(cfg.get("validation.eval_min_minutes", 500))
    rows = []
    for T, g in oof.groupby("year"):
        g = g[g["elig"] & (g["y_min"] >= min_min)]
        for k in kinds:
            rows.append({"target": spec.name, "model": k, "test_year": T, **regression_metrics(g["y"], g[f"pred_{k}"], top_n)})
    return pd.DataFrame(rows)


def select_model(summary: pd.DataFrame, cfg) -> str:
    # the weighted-average benchmark competes too: a learned model must beat it by more than `tol` to be used
    cands = [k for k in ["baseline"] + list(cfg.get("models.final_candidates")) if k in summary.index]
    metric = cfg.get("validation.selection_metric", "rmse")
    tol = float(cfg.get("validation.parsimony_tolerance", 0.01))
    sign = -1.0 if metric in HIGHER_BETTER else 1.0
    score = sign * summary.loc[cands, metric]
    best = score.min()
    for k in COMPLEXITY_ORDER:                     # simplest model that is within `tol` of the best
        if k in cands and score[k] <= best + abs(best) * tol:
            return k
    return score.idxmin()


def importance_table(res_fold_models: dict, final_models: dict, tab: TrainingTable, spec: TargetSpec, T_last: int,
                     cfg) -> pd.DataFrame:
    X = tab.X if spec.feature_cols is None else tab.X[[c for c in spec.feature_cols if c in tab.X.columns]]
    meta, Y = tab.meta, tab.y
    rows_mask = (meta["year"].to_numpy() == T_last) & Y[spec.elig_col].to_numpy(bool)
    rows = []
    for kind, fm in res_fold_models.items():
        if fm.est_ is None or rows_mask.sum() < 20:
            continue
        r = (Y[spec.y_col].to_numpy(float)[rows_mask] - meta[spec.base_col].to_numpy(float)[rows_mask]) / fm.y_sd_
        try:
            pi = permutation_importance(fm.est_, X[rows_mask], r, n_repeats=5, random_state=0,
                                        scoring="neg_mean_squared_error")
        except Exception as exc:
            log.warning("permutation importance failed for %s/%s: %s", spec.name, kind, exc)
            continue
        names = list(X.columns)
        d = pd.DataFrame({"feature": names, "perm_importance_mean": pi.importances_mean * fm.y_sd_ ** 2,
                          "perm_importance_std": pi.importances_std * fm.y_sd_ ** 2})
        d["target"], d["model"] = spec.name, kind
        ff = final_models.get(kind)
        d["coef_std"] = np.nan
        d["mean_abs_contribution"] = np.nan
        d["tree_importance"] = np.nan
        if ff is not None and ff.est_ is not None:
            clean, model = ff.est_.named_steps["clean"], ff.est_.named_steps["model"]
            cols = clean.columns_
            if ff.is_linear:
                coef = pd.Series(model.coef_ * ff.y_sd_, index=cols)        # effect of +1 SD of feature, target units
                Z = ff.est_.named_steps["scale"].transform(clean.transform(X[meta["year"].to_numpy() == T_last]))
                contrib = pd.Series(np.abs(Z * coef.to_numpy()).mean(0), index=cols)
                d["coef_std"] = d["feature"].map(coef)
                d["mean_abs_contribution"] = d["feature"].map(contrib)
            elif hasattr(model, "feature_importances_"):
                try:
                    d["tree_importance"] = d["feature"].map(pd.Series(model.feature_importances_, index=cols))
                except AttributeError:
                    pass
        d["group"] = d["feature"].str.split("__").str[0]
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_target(tab: TrainingTable, spec: TargetSpec, cfg, test_years, kinds=("baseline", "fixed", "ridge", "elastic_net", "gbm")) -> TargetResult:
    kinds = [k for k in kinds if not (k == "fixed" and spec.alt_col is None)]
    learned = [k for k in kinds if k in COMPLEXITY_ORDER and k != "baseline"]
    log.info("[%s] chronological validation on %s (train seasons < each test season)", spec.name, test_years)
    oof, fold_models = walk_forward(tab, spec, kinds, test_years, cfg)
    fm = fold_metric_table(oof, kinds, spec, cfg)
    summary = fm.groupby("model")[METRIC_COLS + ["n"]].mean(numeric_only=True)
    selected = select_model(summary, cfg)
    X = tab.X if spec.feature_cols is None else tab.X[[c for c in spec.feature_cols if c in tab.X.columns]]
    meta, Y = tab.meta, tab.y
    elig = Y[spec.elig_col].to_numpy(bool)
    yv, base = Y[spec.y_col].to_numpy(float), meta[spec.base_col].to_numpy(float)
    w = np.asarray(spec.weight_fn(Y, cfg), float)
    years = meta["year"].to_numpy()
    all_final = {}
    for k in learned:
        if elig.sum() >= spec.min_rows:
            all_final[k] = TargetModel(k, cfg).fit(X[elig], yv[elig], base[elig], w[elig], years[elig])
    final = all_final.get(selected, TargetModel("baseline", cfg))
    explain = final if final.is_linear else all_final.get("ridge", final)
    imp = importance_table({k: m for (T, k), m in fold_models.items() if T == test_years[-1]}, all_final, tab,
                           spec, test_years[-1], cfg)
    for k in kinds:
        s = summary.loc[k]
        log.info("   %-26s rmse=%.4f mae=%.4f r2=%.3f spearman=%.3f top%d_mae=%.4f%s", {**MODEL_LABELS, "fixed": spec.alt_label}[k], s["rmse"], s["mae"],
                 s["r2"], s["spearman"], int(cfg.get("validation.top_n", 100)), s["top_n_mae"],
                 "   <== selected" if k == selected else "")
    return TargetResult(spec=spec, oof=oof, fold_metrics=fm, summary=summary, selected=selected, final=final,
                        explain=explain, all_final=all_final, fold_models=fold_models, importance=imp)


# ====================================================================== composite pipeline evaluation
def reconcile_minutes(team: pd.Series, year: pd.Series, mpg: pd.Series, avail: pd.Series, mode: str, cfg) -> pd.Series:
    """Scale each team's projected minutes so that sum(mpg x availability) is consistent with 240 minutes/game."""
    rc = cfg.get("minutes_reconciliation")
    if mode == "none":
        return mpg.copy()
    em = mpg * avail
    s = em.groupby([year.to_numpy(), team.to_numpy()]).transform("sum")
    f = float(rc["team_minutes"]) / s.replace(0, np.nan)
    f = f.clip(lower=float(rc["cap_down"]), upper=1.0 if mode == "down" else float(rc["cap_up"])).fillna(1.0)
    return np.minimum(mpg * f, float(rc["max_mpg"]))


def assemble_oof(results: dict, tab: TrainingTable) -> pd.DataFrame:
    base = pd.concat([tab.meta[["year", "player_id", "team", "pathway", "age", "prior_min", "n_prior_seasons",
                                "ref_fp", "ref_fppg", "base_fppm", "base_mpg", "base_avail"]],
                      tab.y[["y_fppm", "y_mpg", "y_avail", "y_gp", "y_min", "y_fp", "y_fppg", "elig_fppm", "elig_mpg",
                             "elig_avail"]]], axis=1)
    for t in ("fppm", "mpg", "avail"):
        sel = results[t].selected
        o = results[t].oof[["year", "player_id", f"pred_{sel}"]].rename(columns={f"pred_{sel}": f"p_{t}"})
        base = base.merge(o, on=["year", "player_id"], how="inner")
    return base


def choose_reconciliation(df: pd.DataFrame, cfg) -> tuple[str, pd.DataFrame]:
    mode_cfg = cfg.get("minutes_reconciliation.mode", "auto")
    main = df[(df["pathway"] == "main") & (df["y_gp"] >= 10)]
    rows = []
    for mode in ("none", "down", "both"):
        rec = reconcile_minutes(df["team"], df["year"], df["p_mpg"], df["p_avail"], mode, cfg).loc[main.index]
        rows.append({"mode": mode, "mpg_mae": float(np.mean(np.abs(rec - main["y_mpg"]))),
                     "mpg_rmse": float(np.sqrt(np.mean((rec - main["y_mpg"]) ** 2)))})
    tab = pd.DataFrame(rows).set_index("mode")
    if mode_cfg != "auto":
        return mode_cfg, tab
    best = tab["mpg_mae"].idxmin()
    chosen = best if tab.loc[best, "mpg_mae"] < tab.loc["none", "mpg_mae"] * 0.99 else "none"
    return chosen, tab


def composite_metrics(df: pd.DataFrame, games_by_year: dict, mpg_col: str, cfg) -> pd.DataFrame:
    """Total / per-game fantasy-value accuracy of the combined FPPM x minutes x availability projection."""
    top_n = int(cfg.get("validation.top_n", 100))
    d = df[df["pathway"] == "main"].copy()
    d["games"] = d["year"].map(games_by_year).fillna(cfg.season_length)
    d["p_fppg"] = d["p_fppm"] * d[mpg_col]
    d["p_total"] = d["p_fppg"] * d["p_avail"] * d["games"]
    d["b_total"] = d["base_fppm"] * d["base_mpg"] * d["base_avail"] * d["games"]
    d["n_total"] = d["ref_fp"]                       # naive: last season's total fantasy points
    d["n_fppg"] = d["ref_fppg"]
    d["b_fppg"] = d["base_fppm"] * d["base_mpg"]
    rows = []
    for T, g in list(d.groupby("year")) + [("all", d)]:
        for model, pcol in (("pipeline_selected", "p_total"), ("baseline_weighted_history", "b_total"),
                            ("naive_last_season_total", "n_total")):
            r = regression_metrics(g["y_fp"], g[pcol], top_n)
            top_p = set(g.nlargest(top_n, pcol)["player_id"]) if len(g) else set()
            top_a = set(g.nlargest(top_n, "y_fp")["player_id"]) if len(g) else set()
            rows.append({"target": "total_fp", "model": model, "test_year": T, **r,
                         "top_n_overlap": len(top_p & top_a) / max(top_n, 1) if T != "all" else np.nan})
        gp = g[g["y_gp"] >= 10]
        for model, pcol in (("pipeline_selected", "p_fppg"), ("baseline_weighted_history", "b_fppg"),
                            ("naive_last_season_fppg", "n_fppg")):
            r = regression_metrics(gp["y_fppg"], gp[pcol], top_n)
            top_p = set(gp.nlargest(top_n, pcol)["player_id"]) if len(gp) else set()
            top_a = set(gp.nlargest(top_n, "y_fppg")["player_id"]) if len(gp) else set()
            rows.append({"target": "fppg", "model": model, "test_year": T, **r,
                         "top_n_overlap": len(top_p & top_a) / max(top_n, 1) if T != "all" else np.nan})
    return pd.DataFrame(rows)

"""Game-level modelling (player x game), features known BEFORE tip-off.

Each row is one player-game.  Features use only earlier games of that player / team:
  rolling 5/10/20-game FPPM, minutes and per-minute usage proxies (shift-by-one exclusive windows),
  season-to-date FPPM, prior-season FPPM, home/away, rest days, back-to-back, opponent pace and defensive
  rating *to date*, own team pace to date, and teammate availability (rotation players absent tonight,
  rotation = players averaging >= 18 minutes over their previous 20 games).

Uses
----
* learn how schedule/context moves fantasy production (outputs/game_context_effects.csv),
* chronological validation vs naive rolling-average baselines (rows target='game_fppm' in model_performance.csv),
* estimate sampling noise of FPPM and minutes -> empirical-Bayes strength K for in-season updating (simulation.eb_prior_strength).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models import FeatureCleaner
from .utils import log
from .validation import regression_metrics

CTX_FEATURES = ["home", "b2b", "rest_days", "opp_pace", "opp_drtg", "team_pace", "rot_missing_n", "rot_missing_min"]
FORM_FEATURES = ["r5_fppm", "r10_fppm", "r20_fppm", "std_fppm", "prior_fppm", "r5_mpg", "r10_mpg", "r20_mpg",
                 "std_mpg", "r20_fga_pm", "r20_ast_pm", "r20_reb_pm", "r20_stocks_pm", "r20_tov_pm", "log_games"]
GAME_FEATURES = FORM_FEATURES + CTX_FEATURES


def _roll(df: pd.DataFrame, cols: list[str], key: str, windows) -> pd.DataFrame:
    """Exclusive (previous-games-only) rolling sums via cumulative sums - no look-ahead.

    cs[i] = sum of x over the player's earlier games; the last-k-games window is cs[i] - cs[i-k]."""
    g = df.groupby(key, sort=False)
    pos = g.cumcount()
    cs = {c: g[c].cumsum() - df[c] for c in cols}
    out = {}
    for k in windows:
        for c in cols:
            lag = cs[c].groupby(df[key], sort=False).shift(k)
            out[f"{c}_s{k}"] = cs[c] - lag.fillna(0.0)
        out[f"n_s{k}"] = np.minimum(pos, k)
    for c in cols:
        out[f"{c}_all"] = cs[c]
    out["n_all"] = pos
    return pd.DataFrame(out, index=df.index)


def build_game_dataset(pdx, cfg) -> pd.DataFrame:
    pg = pdx.player_games.copy()
    pg["stocks"] = pg["stl"] + pg["blk"]
    pg = pg.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    cols = ["fp", "min", "fga", "ast", "reb", "stocks", "tov"]
    win = list(cfg.get("game_model.rolling_windows", [5, 10, 20]))
    r = _roll(pg, cols, "player_id", win)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in win:
            ok = r[f"n_s{k}"] >= 3
            pg[f"r{k}_fppm"] = np.where(ok, r[f"fp_s{k}"] / r[f"min_s{k}"], np.nan)
            pg[f"r{k}_mpg"] = np.where(ok, r[f"min_s{k}"] / r[f"n_s{k}"], np.nan)
        k20 = 20 if 20 in win else win[-1]
        for c, nm in (("fga", "fga"), ("ast", "ast"), ("reb", "reb"), ("stocks", "stocks"), ("tov", "tov")):
            pg[f"r20_{nm}_pm"] = np.where(r[f"n_s{k20}"] >= 3, r[f"{c}_s{k20}"] / r[f"min_s{k20}"], np.nan)
    # season-to-date (within season) + prior-season fallback
    pg["_key"] = pg["player_id"].astype(str) + "_" + pg["year"].astype(str)
    s = _roll(pg.assign(_k=pg["_key"]), ["fp", "min"], "_k", [])
    with np.errstate(divide="ignore", invalid="ignore"):
        pg["std_fppm"] = np.where(s["n_all"] >= 3, s["fp_all"] / s["min_all"], np.nan)
        pg["std_mpg"] = np.where(s["n_all"] >= 3, s["min_all"] / s["n_all"], np.nan)
    prior = pdx.player_seasons[["player_id", "year", "fppm"]].assign(year=lambda d: d["year"] + 1).rename(columns={"fppm": "prior_fppm"})
    pg = pg.merge(prior, on=["player_id", "year"], how="left")
    pg["prior_fppm"] = pg["prior_fppm"].fillna(pg["r20_fppm"])
    pg["log_games"] = np.log1p(s["n_all"].to_numpy())
    # ---- schedule / opponent context -------------------------------------------------------------------------
    tg = pdx.team_games.sort_values(["year", "team", "game_date"]).copy()
    tg["prev_date"] = tg.groupby(["year", "team"])["game_date"].shift()
    tg["rest_days"] = (tg["game_date"] - tg["prev_date"]).dt.days.fillna(7).clip(1, 7)
    for c in ("poss", "min", "opp_pts"):
        tg[f"cum_{c}"] = tg.groupby(["year", "team"])[c].cumsum() - tg[c]
    ng = tg.groupby(["year", "team"]).cumcount()
    with np.errstate(divide="ignore", invalid="ignore"):
        tg["pace_td"] = np.where(ng >= 8, 48 * tg["cum_poss"] / (tg["cum_min"] / 5), np.nan)
        tg["drtg_td"] = np.where(ng >= 8, 100 * tg["cum_opp_pts"] / tg["cum_poss"], np.nan)
    ts = pdx.team_seasons[["year", "team", "pace", "drtg"]].assign(year=lambda d: d["year"] + 1)
    tg = tg.merge(ts.rename(columns={"pace": "pace_prev", "drtg": "drtg_prev"}), on=["year", "team"], how="left")
    lg = pdx.league.set_index("year")["pace"]
    tg["pace_td"] = tg["pace_td"].fillna(tg["pace_prev"]).fillna(tg["year"].map(lg).fillna(lg.mean()))
    tg["drtg_td"] = tg["drtg_td"].fillna(tg["drtg_prev"]).fillna(tg["drtg_td"].mean())
    own = tg[["game_id", "team", "rest_days", "pace_td"]].rename(columns={"pace_td": "team_pace"})
    opp = tg[["game_id", "team", "pace_td", "drtg_td"]].rename(columns={"team": "opp", "pace_td": "opp_pace", "drtg_td": "opp_drtg"})
    pg = pg.merge(own, on=["game_id", "team"], how="left").merge(opp, on=["game_id", "opp"], how="left")
    pg["b2b"] = (pg["rest_days"] <= 1).astype(float)
    pg["home"] = pg["home"].astype(float)
    # ---- teammate availability: rotation players (r20 minutes >= 18) who sat out tonight ------------------------
    pg["rot_missing_n"], pg["rot_missing_min"] = 0.0, 0.0
    key = pg.set_index(["year", "team", "team_game_no"]).index
    miss_n, miss_m = {}, {}
    for (y, t), g in pg.groupby(["year", "team"], sort=False):
        games = np.sort(g["team_game_no"].unique())
        pl = g["player_id"].unique()
        gi = pd.Series(np.arange(len(games)), index=games)
        pi = pd.Series(np.arange(len(pl)), index=pl)
        A = np.zeros((len(games), len(pl)), bool)
        Mx = np.full((len(games), len(pl)), np.nan)
        ri, ci = gi.loc[g["team_game_no"]].to_numpy(), pi.loc[g["player_id"]].to_numpy()
        A[ri, ci] = True
        Mx[ri, ci] = g["r20_mpg"].to_numpy()
        Mf = pd.DataFrame(Mx).ffill().to_numpy()
        first, last = A.argmax(0), len(games) - 1 - A[::-1].argmax(0)
        idx = np.arange(len(games))[:, None]
        present = (idx >= first) & (idx <= last)
        miss = (np.nan_to_num(Mf) >= 18) & present & ~A
        for no, n_, m_ in zip(games, miss.sum(1), (np.nan_to_num(Mf) * miss).sum(1)):
            miss_n[(y, t, no)], miss_m[(y, t, no)] = n_, m_
    pg["rot_missing_n"] = [miss_n.get(k, 0.0) for k in key]
    pg["rot_missing_min"] = [miss_m.get(k, 0.0) for k in key]
    pg["g_fppm"] = pg["fp"] / pg["min"]
    return pg.drop(columns=["_key"])


def _fit_predict(kind: str, tr: pd.DataFrame, te: pd.DataFrame, cfg):
    X, y, w = tr[GAME_FEATURES], tr["g_fppm"].to_numpy(), tr["min"].to_numpy()
    if kind == "ridge":
        est = make_pipeline(FeatureCleaner(0.005), StandardScaler(), Ridge(alpha=300.0))
        est.fit(X, y, ridge__sample_weight=w)
    else:
        cap = int(cfg.get("game_model.gbm_max_rows", 100000))
        if len(tr) > cap:
            tr = tr.sample(cap, random_state=0)
            X, y, w = tr[GAME_FEATURES], tr["g_fppm"].to_numpy(), tr["min"].to_numpy()
        est = make_pipeline(FeatureCleaner(0.005), HistGradientBoostingRegressor(
            max_depth=3, max_iter=120, learning_rate=0.06, min_samples_leaf=200, l2_regularization=10.0, random_state=0))
        est.fit(X, y, histgradientboostingregressor__sample_weight=w)
    return est, est.predict(te[GAME_FEATURES])


def run_game_model(pdx, cfg, test_years) -> dict:
    """Validate game-level models chronologically; return metrics, context effects and noise estimates."""
    gd = build_game_dataset(pdx, cfg)
    d = gd[(gd["min"] >= 8) & gd["r20_fppm"].notna()].copy()
    rows = []
    for T in test_years:
        tr, te = d[d["year"] < T], d[d["year"] == T]
        if len(tr) < 5000 or len(te) < 500:
            continue
        te_s = te.sample(min(len(te), 30000), random_state=0)
        preds = {"naive_season_to_date": te_s["std_fppm"].fillna(te_s["prior_fppm"]).to_numpy(),
                 "naive_rolling20": te_s["r20_fppm"].to_numpy()}
        for kind in ("ridge", "gbm"):
            _, preds[kind] = _fit_predict(kind, tr, te_s, cfg)
        for name, p in preds.items():
            rows.append({"target": "game_fppm", "model": name, "test_year": T,
                         **regression_metrics(te_s["g_fppm"], p, top_n=2000)})
    final, _ = _fit_predict("ridge", d, d.head(10), cfg)
    clean, scale, ridge = final.steps[0][1], final.steps[1][1], final.steps[2][1]
    coef = pd.Series(ridge.coef_, index=clean.columns_)
    sd = pd.Series(scale.scale_, index=clean.columns_)
    ctx = pd.DataFrame({"feature": [f for f in CTX_FEATURES if f in coef.index]})
    ctx["fppm_per_unit"] = ctx["feature"].map(coef / sd)
    ctx["fp_per_game_at_30_min"] = ctx["fppm_per_unit"] * 30.0
    ctx["fppm_per_1sd"] = ctx["feature"].map(coef)
    # ---- sampling-noise estimates for empirical-Bayes in-season updating
    pgm = gd[gd["min"] > 0]
    gsum = pgm.groupby(["player_id", "year"]).agg(g_s=("game_id", "nunique"), fp_s=("fp", "sum"), mn=("min", "sum"))
    pgm = pgm.merge(gsum.reset_index(), on=["player_id", "year"])
    pgm = pgm[pgm["g_s"] >= 20]
    pgm["xbar"] = pgm["fp_s"] / pgm["mn"]
    pgm["ss"] = pgm["min"] * (pgm["g_fppm"] - pgm["xbar"]) ** 2
    noise = pgm.groupby(["player_id", "year"]).agg(ss=("ss", "sum"), g=("g_s", "first"))
    sigma2 = float((noise["ss"] / (noise["g"] - 1)).mean())
    vmin = pgm.groupby(["player_id", "year"])["min"].var().mean()
    log.info("game-level dataset: %d player-games; noise sigma^2/min=%.3f, var(game minutes)=%.1f", len(gd), sigma2, vmin)
    return {"metrics": pd.DataFrame(rows), "context_effects": ctx, "sigma2_per_min": sigma2,
            "var_game_minutes": float(vmin), "n_rows": int(len(gd))}


def estimate_eb(game_out: dict, oof: pd.DataFrame, cfg) -> dict:
    """Empirical-Bayes prior strength (minutes for FPPM, games for MPG) from game noise vs. season-level error."""
    from .simulation import eb_prior_strength
    ic = cfg.get("in_season")
    d = oof[(oof["pathway"] == "main")]
    f = d[d["elig_fppm"]]
    m = d[d["elig_mpg"]]
    k_min = eb_prior_strength(float(np.var(f["y_fppm"] - f["p_fppm"])), game_out["sigma2_per_min"],
                              float(f["y_min"].mean()), float(ic["k_minutes_default"]))
    k_gm = eb_prior_strength(float(np.var(m["y_mpg"] - m["p_mpg_rec"])), game_out["var_game_minutes"],
                             float(m["y_gp"].mean()), float(ic["k_games_default"]))
    log.info("in-season empirical-Bayes strength: K_minutes=%.0f (FPPM), K_games=%.1f (minutes)", k_min, k_gm)
    return {"k_minutes": k_min, "k_games": k_gm}

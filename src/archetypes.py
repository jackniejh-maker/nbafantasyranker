"""Statistical archetypes + fantasy-specific value analysis.

ESPN points scoring is NOT conventional basketball value: under the default settings every field-goal attempt
is positive expected value above a 25% (2PT) / 16.7% (3PT) make rate, free throws only break even at 50%, and
steals / blocks (+4) and assists (+2) are worth far more per event than points.  `value_analysis` quantifies this
from the league's own data; `classify` labels players from their weighted per-minute fantasy-value profile
(z-scores against all players with >= 600 minutes) - no subjective labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .fantasy_scoring import scoring_breakevens

COMPONENTS = {"scoring": "fp_scoring", "reb": "fp_rebounds", "ast": "fp_assists", "stocks": "fp_stocks"}
LABELS = {"scoring": "scoring-heavy", "reb": "rebounding-heavy", "ast": "playmaking-heavy", "stocks": "stocks-heavy"}


def profile(ps: pd.DataFrame, T: int, lookback: int = 3) -> pd.DataFrame:
    h = ps[(ps["year"] < T) & (ps["year"] >= T - lookback)]
    agg = {c: (c, "sum") for c in ("min", "fp", "fga", "fp_scoring", "fp_peripheral", "fp_rebounds", "fp_assists",
                                   "fp_stocks", "fp_tov_loss", "fp_miss_loss")}
    g = h.groupby("player_id").agg(**agg)
    for k, c in COMPONENTS.items():
        g[f"{k}_pm"] = g[c] / g["min"]
    g["fp_per_fga"] = g["fp"] / g["fga"].replace(0, np.nan)
    g["scoring_share"] = g["fp_scoring"] / g["fp"].replace(0, np.nan)
    g["peripheral_share"] = g["fp_peripheral"] / g["fp"].replace(0, np.nan)
    return g


def classify(ps: pd.DataFrame, T: int, ids, min_ref: float = 600.0, min_label: float = 300.0) -> pd.DataFrame:
    g = profile(ps, T)
    ref = g[g["min"] >= min_ref]
    cols = [f"{k}_pm" for k in COMPONENTS]
    z = (g[cols] - ref[cols].mean()) / ref[cols].std().replace(0, np.nan)
    zmax = z.max(axis=1)
    arg = z.idxmax(axis=1).str.replace("_pm", "", regex=False).map(LABELS)
    label = pd.Series(np.where(zmax < 0.6, "all-around", arg), index=g.index)
    z_eff = (g["fp_per_fga"] - ref["fp_per_fga"].mean()) / ref["fp_per_fga"].std()
    label[(z_eff >= 1.0) & (zmax < 1.0)] = "efficiency-driven"
    label[g["min"] < min_label] = "unclassified (limited history)"
    out = pd.DataFrame({"archetype": label, "scoring_share": g["scoring_share"], "peripheral_share": g["peripheral_share"]})
    out = out.reindex(ids)
    out["archetype"] = out["archetype"].fillna("unclassified (no NBA history)")
    return out


def value_analysis(ps: pd.DataFrame, cfg, T: int) -> tuple[pd.DataFrame, str]:
    """League-level facts about where ESPN fantasy value comes from (last completed season, >= 1000 minutes)."""
    rows = [{"section": "scoring_breakeven", "metric": k, "value": v, "note": ""} for k, v in scoring_breakevens(cfg.scoring).items()]
    q = ps[(ps["year"] == T - 1) & (ps["min"] >= 1000)].copy()
    if len(q) < 30:
        q = ps[(ps["year"] >= T - 3) & (ps["year"] < T) & (ps["min"] >= 1000)].copy()
    top = q["fppm"] >= q["fppm"].quantile(0.8)
    for col, label in (("fp_scoring", "scoring block (PTS+2FGM-FGA+FTM-FTA+3PM)"), ("fp_rebounds", "rebounds"),
                       ("fp_assists", "assists (x2)"), ("fp_stocks", "steals+blocks (x4)"),
                       ("fp_tov_loss", "turnovers (x-2)"), ("fp_miss_loss", "missed shots (attempt cost)")):
        pm = q[col] / q["min"]
        rows.append({"section": "fp_per_minute", "metric": label, "value": float(pm.mean()),
                     "note": f"top-20% FPPM players: {float(pm[top].mean()):.3f}; others: {float(pm[~top].mean()):.3f}"})
    q["stocks_pct"] = q["stl_pct"] + q["blk_pct"]
    for col in ("usg", "ts", "efg", "ast_pct", "reb_pct", "stocks_pct", "tov_pct", "fp_per_fga", "fp_per_poss", "mpg"):
        d = q[["fppm", col]].dropna()
        if len(d) > 20:
            rows.append({"section": "spearman_with_FPPM", "metric": col, "value": float(stats.spearmanr(d["fppm"], d[col]).statistic),
                         "note": "conventional efficiency (TS%, eFG%) vs fantasy value"})
    rows.append({"section": "totals", "metric": "share_of_fp_from_scoring_block", "value": float(q["fp_scoring"].sum() / q["fp"].sum()), "note": ""})
    rows.append({"section": "totals", "metric": "share_of_fp_from_peripherals", "value": float(q["fp_peripheral"].sum() / q["fp"].sum()), "note": ""})
    df = pd.DataFrame(rows)
    be = scoring_breakevens(cfg.scoring)
    md = ["# ESPN scoring: where fantasy value comes from", "",
          f"A made 2 = {be['value_made_2pt']:+.0f}, made 3 = {be['value_made_3pt']:+.0f}, made FT = {be['value_made_ft']:+.0f}, any missed shot = {be['value_missed_fg']:+.0f}.",
          f"Break-even make rates: 2PA {be['breakeven_2pt_pct']:.1%}, 3PA {be['breakeven_3pt_pct']:.1%}, FTA {be['breakeven_ft_pct']:.1%} - "
          "so volume scorers produce fantasy value even when 'inefficient' by TS%, while free-throw-heavy low-percentage shooters do not.", ""]
    for sec, g in df.groupby("section", sort=False):
        md += [f"## {sec}", ""] + [f"* {r.metric}: {r.value:.3f}  {r.note}" for r in g.itertuples()] + [""]
    return df, "\n".join(md)

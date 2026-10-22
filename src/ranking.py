"""Turns point projections + Monte-Carlo distributions into the three rankings and the scores.

rank_total          median simulated season total (DEFAULT)
rank_fppg           median simulated FPPG (fantasy points per game played)
rank_risk_adjusted  median(total) - lambda x downside_risk,  downside_risk = median - mean(worst 10% of seasons)

Breakout score = weighted z-score of measurable factors (projected FPPM gain, minutes gain, probability of a
role expansion, probability of beating last season's total, and - only when consensus data exist - the gap to
the market rank), converted to a 0-100 percentile among relevant players.  No player list is hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .consensus import compare
from .utils import pct_rank, zscore

ROLE_EDGES = [20, 25, 30, 35]
SPEC_COLUMNS = [
    "rank_total", "rank_fppg", "rank_risk_adjusted", "player_id", "player_name", "team", "position", "age",
    "projected_minutes", "projected_fppm", "projected_fppg", "projected_games", "projected_total_fantasy_points",
    "fppm_p10", "fppm_p25", "fppm_p50", "fppm_p75", "fppm_p90",
    "fppg_p10", "fppg_p25", "fppg_p50", "fppg_p75", "fppg_p90",
    "total_p10", "total_p25", "total_p50", "total_p75", "total_p90",
    "upside_score", "risk_score", "confidence_score",
    "historical_fppm", "historical_fppg", "minutes_change", "role_change",
    "team_effect", "teammate_effect", "coach_effect", "age_effect"]
EXTRA_COLUMNS = [
    "minutes_p10", "minutes_p25", "minutes_p50", "minutes_p75", "minutes_p90", "games_p10", "games_p50", "games_p90",
    "projected_avail", "total_mean", "downside_risk", "risk_adjusted_value", "vor_total", "role_effect", "form_effect", "history_adjustment",
    "hist_fppm_weighted", "breakout_score", "p_role_expansion", "p_beat_last_season", "rank_vs_last_season",
    "consensus_rank", "rank_diff_vs_consensus", "value_diff_vs_consensus", "archetype", "scoring_share",
    "peripheral_share", "pathway", "is_rookie", "confidence_label", "inj_note", "override_applied", "fp_to_date",
    "gp_to_date", "games_out"]


def role_index(mpg) -> np.ndarray:
    return np.digitize(np.asarray(mpg, float), ROLE_EDGES)


def assemble(ps, sim: pd.DataFrame, effects: pd.DataFrame, cfg) -> pd.DataFrame:
    """Merge point projections, simulation output, effects and history references into one table."""
    p, m = ps.proj, ps.meta
    df = pd.DataFrame(index=p.index)
    df["player_id"] = p.index
    for c in ("player_name", "team", "position", "age", "pathway", "is_rookie", "inj_note", "override_applied",
              "fp_to_date", "gp_to_date", "games_out"):
        df[c] = p[c]
    df["projected_minutes"] = p["mpg"]
    df["projected_fppm"] = p["fppm"]
    df["projected_fppg"] = p["fppm"] * p["mpg"]
    df["projected_avail"] = p["avail_exp"]
    df["projected_games"] = p["avail_exp"] * p["games_rem"]
    df["projected_total_fantasy_points"] = p["fp_to_date"] + df["projected_fppg"] * df["projected_games"]
    df = df.join(sim)
    df["historical_fppm"], df["historical_fppg"] = m["ref_fppm"], m["ref_fppg"]
    df["hist_fppm_weighted"] = m["base_fppm"]
    df["ref_mpg"], df["ref_gp"], df["ref_total"], df["ref_fppm"] = m["ref_mpg"], m["ref_gp"], m["ref_fp"], m["ref_fppm"]
    df["ref_team_games"] = float(cfg.season_length)
    df["minutes_change"] = df["projected_minutes"] - m["ref_mpg"]
    ri_ref = pd.Series(role_index(m["ref_mpg"].fillna(m["mpg_budget"])), index=df.index)
    df["role_change"] = role_index(df["projected_minutes"]) - ri_ref
    for c in effects.columns:
        df[c] = effects[c]
    df["p_beat_last_season"] = sim["p_beat_ref"]
    return df


def rank_and_score(df: pd.DataFrame, cfg, cons: pd.DataFrame | None) -> pd.DataFrame:
    df = df.copy()
    lam = float(cfg.get("ranking.risk_lambda", 0.5))
    df["rank_total"] = df["total_p50"].rank(ascending=False, method="first").astype(int)
    df["rank_fppg"] = df["fppg_p50"].rank(ascending=False, method="first").astype(int)
    df["downside_risk"] = (df["total_p50"] - df["total_cvar"]).clip(lower=0.0)
    df["risk_adjusted_value"] = df["total_p50"] - lam * df["downside_risk"]
    df["rank_risk_adjusted"] = df["risk_adjusted_value"].rank(ascending=False, method="first").astype(int)
    n_rep = int(cfg.get("league.teams", 12)) * int(cfg.get("league.roster_spots", 13))
    srt = np.sort(df["total_p50"].to_numpy())[::-1]
    df["vor_total"] = df["total_p50"] - srt[min(n_rep, len(srt) - 1)]
    df["upside_score"] = pct_rank(df["total_p90"] - df["total_p50"])
    df["risk_score"] = pct_rank(df["downside_risk"] / df["total_p50"].clip(lower=1.0))
    width = (df["total_p90"] - df["total_p10"]) / df["total_p50"].clip(lower=1.0)
    conf = 100.0 - pct_rank(width)
    conf = np.where(df["pathway"] == "newcomer", np.minimum(conf, 100.0) * 0.6, conf)
    df["confidence_score"] = conf
    df["confidence_label"] = pd.cut(df["confidence_score"], [-1, 33.3, 66.7, 101], labels=["Low", "Medium", "High"]).astype(str)
    df["rank_vs_last_season"] = df["ref_total"].rank(ascending=False, method="first") - df["rank_total"]
    df = compare(df, cons, "total_p50", "rank_total")
    df = breakout(df, cfg, has_market=cons is not None and len(cons) > 0)
    return df.sort_values("rank_total").reset_index(drop=True)


def breakout(df: pd.DataFrame, cfg, has_market: bool) -> pd.DataFrame:
    w = dict(cfg.get("breakout.weights"))
    rel = (df["projected_minutes"] >= 10) & (df["projected_games"] >= 15)
    comps = {
        "fppm_gain": df["projected_fppm"] - df["ref_fppm"],
        "minutes_gain": df["minutes_change"],
        "role_expansion_prob": df["p_role_expansion"],
        "upside": df["p_beat_last_season"],
        "market_gap": -df["rank_diff_vs_consensus"] if has_market else pd.Series(np.nan, index=df.index),
    }
    use = {k: v for k, v in w.items() if k in comps and (k != "market_gap" or has_market)}
    tot = sum(use.values())
    score = pd.Series(0.0, index=df.index)
    for k, wt in use.items():
        z = pd.Series(zscore(comps[k].where(rel)), index=df.index).fillna(0.0)
        score += (wt / tot) * z
    out = pd.Series(np.nan, index=df.index)
    out[rel] = pct_rank(score[rel])
    df["breakout_score"] = out
    return df


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Spec columns first (exact names/order), then the extra analytics columns."""
    extras = [c for c in EXTRA_COLUMNS if c in df.columns]
    return df[SPEC_COLUMNS + extras]

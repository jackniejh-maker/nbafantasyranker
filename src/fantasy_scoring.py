"""ESPN points-league scoring (values come from config/league_settings.json).

    FP = PTS + 2*FGM - FGA + FTM - FTA + 3PM + REB + 2*AST + 4*STL + 4*BLK - 2*TOV

`PTS` is the box-score point total (a made three is 3 PTS), so a made three is worth
3 + 2 - 1 + 1 = +5 fantasy points, a made two +3, a made free throw +1, and any miss -1.
"""
from __future__ import annotations

import pandas as pd

STAT_KEYS = ["pts", "fgm", "fga", "ftm", "fta", "fg3m", "reb", "ast", "stl", "blk", "tov"]
SCORING_KEYS = ["pts", "fgm", "fga", "ftm", "fta", "fg3m"]      # "scoring efficiency" block
PERIPHERAL_KEYS = ["reb", "ast", "stl", "blk", "tov"]           # "peripheral" block

DEFAULT_ESPN = {"pts": 1, "fgm": 2, "fga": -1, "ftm": 1, "fta": -1, "fg3m": 1,
                "reb": 1, "ast": 2, "stl": 4, "blk": 4, "tov": -2}


def get_weights(scoring: dict | None = None) -> dict:
    """Validate and return the 11 scoring weights."""
    scoring = DEFAULT_ESPN if scoring is None else scoring
    missing = [k for k in STAT_KEYS if k not in scoring]
    if missing:
        raise ValueError(f"missing scoring keys: {missing}")
    return {k: float(scoring[k]) for k in STAT_KEYS}


def fantasy_points(df: pd.DataFrame, w: dict):
    """Vectorised fantasy points; works on game rows or on season totals (linear)."""
    total = 0.0
    for k in STAT_KEYS:
        total = total + w[k] * df[k]
    return total


def add_fantasy_columns(df: pd.DataFrame, w: dict) -> pd.DataFrame:
    """Append `fp` and its decomposition (all linear in the totals)."""
    out = df.copy()
    out["fp"] = fantasy_points(out, w)
    out["fp_scoring"] = sum(w[k] * out[k] for k in SCORING_KEYS)
    out["fp_peripheral"] = sum(w[k] * out[k] for k in PERIPHERAL_KEYS)
    out["fp_stocks"] = w["stl"] * out["stl"] + w["blk"] * out["blk"]
    out["fp_assists"] = w["ast"] * out["ast"]
    out["fp_rebounds"] = w["reb"] * out["reb"]
    out["fp_tov_loss"] = w["tov"] * out["tov"]
    # points "lost" on missed shots: each missed FG / FT costs its attempt weight
    out["fp_miss_loss"] = w["fga"] * (out["fga"] - out["fgm"]) + w["fta"] * (out["fta"] - out["ftm"])
    return out


def scoring_breakevens(w: dict) -> dict:
    """Shooting percentage at which an attempt has zero expected fantasy value.

    Under the default ESPN scoring almost every shot attempt is positive-EV, which is why
    conventional "efficiency" and fantasy value diverge (see README / fantasy_analysis).
    """
    def be(made, miss):
        return float(-miss / (made - miss)) if (made - miss) != 0 else float("nan")

    made2 = 2 * w["pts"] + w["fgm"] + w["fga"]
    made3 = 3 * w["pts"] + w["fgm"] + w["fg3m"] + w["fga"]
    madeft = 1 * w["pts"] + w["ftm"] + w["fta"]
    return {
        "value_made_2pt": made2, "value_made_3pt": made3, "value_made_ft": madeft,
        "value_missed_fg": w["fga"], "value_missed_ft": w["fta"],
        "breakeven_2pt_pct": be(made2, w["fga"]),
        "breakeven_3pt_pct": be(made3, w["fga"]),
        "breakeven_ft_pct": be(madeft, w["fta"]),
    }

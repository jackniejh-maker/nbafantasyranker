"""External consensus rankings (ESPN ADP / rank, or a user CSV) - COMPARISON ONLY.

This module is imported only by the ranking/reporting stage.  Nothing here is ever a feature or a training
target, so there is no leakage of market opinion into the predictive model.

    rank_diff   = model_rank - consensus_rank           (negative: the model likes the player MORE than the market)
    value_diff  = projected_total - implied_value(consensus_rank)
where implied_value(r) is the projected total of the player the MODEL ranks r-th, i.e. what a player at that
market slot is worth on the model's own value curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import log, normalize_name


def match_names(names: pd.Series, teams: pd.Series | None, roster: pd.DataFrame, aliases: dict | None = None,
                espn_ids: pd.Series | None = None) -> pd.Series:
    """Map names (+ optional team) to player_id using the current roster; NaN where unmatched / ambiguous.

    `espn_ids` (only meaningful in the ESPN source profile, where player_id IS the ESPN athlete id) take priority."""
    r = roster[["player_id", "player_name", "team"]].copy()
    r["key"] = r["player_name"].map(lambda n: normalize_name(n, aliases))
    keys = names.map(lambda n: normalize_name(n, aliases))
    uniq = r.drop_duplicates("key", keep=False).set_index("key")["player_id"]       # unambiguous names
    out = keys.map(uniq)
    amb = out.isna() & keys.isin(r["key"])
    if amb.any() and teams is not None:                                              # duplicate names: use team
        by_team = r.set_index(["key", "team"])["player_id"]
        by_team = by_team[~by_team.index.duplicated(keep=False)]
        for i in names.index[amb]:
            try:
                out.loc[i] = by_team.loc[(keys.loc[i], teams.loc[i])]
            except KeyError:
                pass
    if espn_ids is not None:
        direct = pd.to_numeric(espn_ids, errors="coerce")
        out = direct.where(direct.isin(set(roster["player_id"]))).combine_first(out)
    return out


def load_consensus(pdx, cfg, roster: pd.DataFrame) -> pd.DataFrame | None:
    """Return player_id, consensus_rank (1..N), adp, source, or None when no consensus data is available."""
    ext = cfg.project_path(cfg.get("paths.external_consensus_csv", "data/external/consensus_rankings.csv"))
    df, src = None, None
    if ext.exists():
        df, src = pd.read_csv(ext), f"csv:{ext.name}"
    elif pdx.consensus is not None and len(pdx.consensus):
        df, src = pdx.consensus.copy(), "espn"
    if df is None or "player_name" not in df.columns:
        return None
    d = df.copy()
    rank = None
    for c in ("consensus_rank", "rank", "espn_rank"):
        if c in d.columns:
            rank = pd.to_numeric(d[c], errors="coerce")
            break
    adp = pd.to_numeric(d["adp"], errors="coerce") if "adp" in d.columns else None
    key = adp.where(adp > 0) if adp is not None else None
    d["_sort"] = key if key is not None and key.notna().sum() > 20 else rank
    d = d.dropna(subset=["_sort"])
    use_ids = cfg.get("source_profile") == "espn" and "espn_id" in d.columns
    d["player_id"] = match_names(d["player_name"], d["team"] if "team" in d.columns else None, roster, cfg.aliases,
                                 espn_ids=d["espn_id"] if use_ids else None)
    unmatched = d[d["player_id"].isna()]
    if len(unmatched):
        log.warning("consensus: %d of %d names could not be matched to the current roster (e.g. %s) - add them to "
                    "`name_aliases` in league_settings.json", len(unmatched), len(d), unmatched["player_name"].head(4).tolist())
    d = d.dropna(subset=["player_id"]).sort_values("_sort").drop_duplicates("player_id")
    d["player_id"] = d["player_id"].astype("int64")
    d["consensus_rank"] = np.arange(1, len(d) + 1)
    d["source"] = src
    return d[["player_id", "consensus_rank", "source"] + (["adp"] if "adp" in d.columns else [])]


def compare(rank_df: pd.DataFrame, cons: pd.DataFrame | None, value_col: str = "total_p50",
            rank_col: str = "rank_total") -> pd.DataFrame:
    """Append consensus_rank, rank_diff and value_diff to the ranking table."""
    out = rank_df.drop(columns=["consensus_rank", "rank_diff_vs_consensus", "value_diff_vs_consensus"], errors="ignore").copy()
    if cons is None or len(cons) == 0:
        for c in ("consensus_rank", "rank_diff_vs_consensus", "value_diff_vs_consensus"):
            out[c] = np.nan
        return out
    out = out.merge(cons[["player_id", "consensus_rank"]], on="player_id", how="left")
    curve = np.sort(out[value_col].to_numpy())[::-1]
    r = out["consensus_rank"]
    implied = pd.Series(np.where(r.notna(), curve[np.clip(r.fillna(1).astype(int) - 1, 0, len(curve) - 1)], np.nan),
                        index=out.index)
    out["rank_diff_vs_consensus"] = out[rank_col] - r
    out["value_diff_vs_consensus"] = out[value_col] - implied
    return out

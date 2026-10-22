"""Coach features.

The coach's *name* is never a feature.  For every team-season we measure style metrics from what
happened on the floor (pace, starter/star/bench minutes, rotation size, rookie / 2nd-year minutes,
lineup stability, double-big and small-ball frequency, usage concentration), express each as a
deviation from that season's league mean (removes era effects), and attach to a coach the
games-weighted average deviation over his PRIOR seasons, shrunk toward zero (= league average) by
`K` pseudo-games.  A coach with little history therefore gets ~league-average features, and
`coach__games_known` tells the model how much to trust them.

Lineup proxies: without per-game starter data, "starting lineup" = the five players with the most
minutes in that game; "big" = a player whose listed position contains C.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COACH_METRICS = ["pace", "starter_mpg", "star_mpg", "bench_mpg", "rot_size", "rookie_mpg", "y2_mpg",
                 "stability", "double_big", "small_ball", "usage_hhi"]
SHRINK_GAMES = 164.0  # two seasons of pseudo-games pulling coach tendencies toward the league mean


def lineup_style(pg: pd.DataFrame, ps_pos: pd.DataFrame) -> pd.DataFrame:
    """Per team-season lineup stability / double-big / small-ball from per-game top-5-by-minutes sets."""
    d = pg[["year", "team", "game_id", "game_date", "player_id", "min"]].merge(
        ps_pos[["player_id", "year", "position"]], on=["player_id", "year"], how="left")
    d["rk"] = d.groupby(["game_id", "team"])["min"].rank(method="first", ascending=False)
    t5 = d[d["rk"] <= 5].copy()
    t5["is_big"] = t5["position"].astype(str).str.contains("C", regex=False)
    g = t5.groupby(["year", "team", "game_id", "game_date"]).agg(
        n_big=("is_big", "sum"), ids=("player_id", lambda s: frozenset(s))).reset_index()
    g = g.sort_values(["year", "team", "game_date", "game_id"])
    rows = []
    for (y, t), grp in g.groupby(["year", "team"], sort=False):
        sets = grp["ids"].tolist()
        jac = [len(a & b) / len(a | b) for a, b in zip(sets[:-1], sets[1:]) if len(a | b)]
        nb = grp["n_big"].to_numpy()
        rows.append({"year": y, "team": t, "stability": float(np.mean(jac)) if jac else np.nan,
                     "double_big": float(np.mean(nb >= 2)), "small_ball": float(np.mean(nb == 0))})
    return pd.DataFrame(rows, columns=["year", "team", "stability", "double_big", "small_ball"])


def team_style(stints: pd.DataFrame, ps: pd.DataFrame, team_seasons: pd.DataFrame,
               pg: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (year, team) with the raw style metrics (no coach attached yet)."""
    st = stints[stints["gp"] >= 15].merge(ps[["player_id", "year", "exp"]], on=["player_id", "year"], how="left")
    st["mpg_s"] = st["min"] / st["gp"]
    rows = []
    for (y, t), g in st.groupby(["year", "team"]):
        m = np.sort(g["mpg_s"].to_numpy())[::-1]
        share = g["min"].to_numpy() / g["min"].sum()
        fsh = g["fga"].to_numpy() / max(g["fga"].sum(), 1.0)

        def wmean(mask):
            sub = g[mask]
            return float((sub["mpg_s"] * sub["min"]).sum() / sub["min"].sum()) if len(sub) else np.nan

        rows.append({"year": y, "team": t, "starter_mpg": m[:5].mean(), "star_mpg": m[:2].mean(),
                     "bench_mpg": m[5:10].mean() if len(m) > 5 else np.nan,
                     "rot_size": 1.0 / np.sum(share ** 2), "usage_hhi": float(np.sum(fsh ** 2)),
                     "rookie_mpg": wmean(g["exp"] == 0), "y2_mpg": wmean(g["exp"] == 1)})
    style = pd.DataFrame(rows).merge(team_seasons[["year", "team", "pace", "games"]], on=["year", "team"], how="left")
    if pg is not None and len(pg):
        style = style.merge(lineup_style(pg, ps), on=["year", "team"], how="left")
    for c in COACH_METRICS:
        if c not in style.columns:
            style[c] = np.nan
    return style


def coach_deviations(style: pd.DataFrame, head_coaches: pd.DataFrame) -> pd.DataFrame:
    """Attach the head coach and express every metric as a deviation from the league mean of that year."""
    d = style.merge(head_coaches[["year", "team", "coach_id"]], on=["year", "team"], how="left")
    for c in COACH_METRICS:
        d[f"dev_{c}"] = d[c] - d.groupby("year")[c].transform("mean")
    return d


def coach_features(T: int, dev: pd.DataFrame, coach_of_team: dict, prev_coach_of_team: dict,
                   shrink_games: float = SHRINK_GAMES) -> pd.DataFrame:
    """Features for target season T, indexed by team.  Uses ONLY seasons < T."""
    hist = dev[(dev["year"] < T) & dev["coach_id"].notna()]
    cols = [f"dev_{c}" for c in COACH_METRICS]
    out = {}
    if len(hist):
        g = hist.groupby("coach_id")
        games = g["games"].sum()
        for c in COACH_METRICS:
            ok = hist[f"dev_{c}"].notna()
            num = (hist["games"] * hist[f"dev_{c}"]).where(ok).groupby(hist["coach_id"]).sum()
            den = hist["games"].where(ok).groupby(hist["coach_id"]).sum()
            out[c] = num / (den.fillna(0) + shrink_games)
        table = pd.DataFrame(out)
        table["games_known"] = games
    else:
        table = pd.DataFrame(columns=COACH_METRICS + ["games_known"])
    rows = {}
    for team, cid in coach_of_team.items():
        if cid is not None and cid in table.index:
            r = table.loc[cid].to_dict()
        elif cid is not None:
            r = {c: 0.0 for c in COACH_METRICS}
            r["games_known"] = 0.0
        else:
            r = {c: np.nan for c in COACH_METRICS}
            r["games_known"] = np.nan
        r["new_to_team"] = float(cid is not None and prev_coach_of_team.get(team) not in (None, cid))
        rows[team] = r
    res = pd.DataFrame.from_dict(rows, orient="index")
    res.columns = [f"coach__{c}" for c in res.columns]
    return res

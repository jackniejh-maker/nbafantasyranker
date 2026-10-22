"""Team-environment features for the player's CURRENT team (target season T).

Allowed inputs: team results from seasons < T and the roster known before season T.
Projected pace / offensive / defensive rating = league mean + (estimated persistence) x (last season's
deviation); persistence is the through-origin regression slope of year-over-year team deviations
estimated on seasons < T.  Roster-turnover features quantify how many minutes / shots / assists /
rebounds left the team (vacated) and arrived (added) relative to last season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def persistence(ts: pd.DataFrame, T: int, col: str, default: float = 0.6) -> float:
    d = ts[ts["year"] < T][["year", "team", col]].dropna().copy()
    d["dev"] = d[col] - d.groupby("year")[col].transform("mean")
    prev = d[["year", "team", "dev"]].assign(year=lambda x: x["year"] + 1).rename(columns={"dev": "dev_prev"})
    m = d.merge(prev, on=["year", "team"])
    if len(m) < 20 or (m["dev_prev"] ** 2).sum() <= 0:
        return default
    return float(np.clip((m["dev"] * m["dev_prev"]).sum() / (m["dev_prev"] ** 2).sum(), 0.0, 1.0))


def team_context(T: int, roster: pd.DataFrame, base: pd.DataFrame, stints: pd.DataFrame, ps: pd.DataFrame,
                 ts: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    """Team-level features, indexed by team.  `base` is indexed by player_id (mpg_ref0, avail_ref, fga_pm_w...)."""
    teams = sorted(roster["team"].dropna().unique())
    out = pd.DataFrame(index=teams)
    prev = ts[ts["year"] == T - 1].set_index("team")
    for col in ("pace", "ortg", "drtg"):
        lg = ts.loc[ts["year"] == T - 1, col].mean() if (ts["year"] == T - 1).any() else np.nan
        out[f"{col}_l1"] = prev[col].reindex(teams) if col in prev else np.nan
        b = persistence(ts, T, col)
        out[f"{col}_proj"] = lg + b * (out[f"{col}_l1"] - lg)
        out[f"{col}_proj"] = out[f"{col}_proj"].fillna(lg)
    out["net_proj"] = out["ortg_proj"] - out["drtg_proj"]
    for src, dst in (("poss_pg", "poss_pg_l1"), ("fga_pg", "fga_pg_l1"), ("ast_pg", "ast_pg_l1"), ("reb_pg", "reb_pg_l1"),
                     ("tpar", "tpar_l1"), ("ftr", "ftr_l1"), ("ast_rate", "ast_rate_l1")):
        out[dst] = prev[src].reindex(teams) if src in prev else np.nan

    # ---- turnover versus last season's roster of the same team
    st_prev = stints[stints["year"] == T - 1]
    ps_prev = ps[ps["year"] == T - 1].set_index("player_id")
    lg_games = float(league.loc[league["year"] == T - 1, "games"].iloc[0]) if (league["year"] == T - 1).any() else np.nan
    members_by_team = roster.groupby("team")["player_id"].apply(set).to_dict()
    cols = ["continuity", "min_vacated", "min_added", "fga_vacated", "fga_added", "ast_vacated", "ast_added",
            "reb_vacated", "reb_added"]
    res = {c: [] for c in cols}
    for k in teams:
        members = members_by_team.get(k, set())
        s = st_prev[st_prev["team"] == k]
        g = prev["games"].get(k, np.nan) if "games" in prev else np.nan
        tot = s["min"].sum()
        stay = s[s["player_id"].isin(members)]
        left = s[~s["player_id"].isin(members)]
        arr_ids = [p for p in members if p in ps_prev.index and ps_prev.at[p, "last_team"] != k]
        a = ps_prev.loc[arr_ids] if arr_ids else ps_prev.iloc[0:0]
        res["continuity"].append(stay["min"].sum() / tot if tot > 0 else np.nan)
        res["min_vacated"].append(left["min"].sum() / g if g == g and g else np.nan)
        res["fga_vacated"].append(left["fga"].sum() / g if g == g and g else np.nan)
        res["ast_vacated"].append(left["ast"].sum() / g if g == g and g else np.nan)
        res["reb_vacated"].append(left["reb"].sum() / g if g == g and g else np.nan)
        res["min_added"].append(a["min"].sum() / lg_games if lg_games == lg_games else np.nan)
        res["fga_added"].append(a["fga"].sum() / lg_games if lg_games == lg_games else np.nan)
        res["ast_added"].append(a["ast"].sum() / lg_games if lg_games == lg_games else np.nan)
        res["reb_added"].append(a["reb"].sum() / lg_games if lg_games == lg_games else np.nan)
    for c in cols:
        out[c] = res[c]
    for what in ("min", "fga", "ast", "reb"):
        out[f"{what}_net"] = out[f"{what}_added"] - out[f"{what}_vacated"]

    # ---- projected roster structure (depth, rotation size, concentration, over-subscription)
    b = base.copy()
    b["exp_min"] = b["mpg_ref0"] * b["avail_ref"]
    tot = b.groupby("team")["exp_min"].transform("sum")
    b["share"] = b["exp_min"] / tot
    b["ush"] = b["fga_pm_w"] * b["exp_min"]
    b["ushare"] = b["ush"] / b.groupby("team")["ush"].transform("sum")
    grp = b.groupby("team")
    out["oversub"] = (grp["exp_min"].sum() / 240.0).reindex(teams)
    out["rot_size_exp"] = grp["share"].apply(lambda s: 1.0 / np.sum(s ** 2)).reindex(teams)
    out["depth15"] = grp["mpg_ref0"].apply(lambda s: float((s >= 15).sum())).reindex(teams)
    out["usage_conc"] = grp["ushare"].apply(lambda s: float(np.sum(s ** 2))).reindex(teams)
    out["top3_min_share"] = grp["share"].apply(lambda s: float(np.sort(s.to_numpy())[::-1][:3].sum())).reindex(teams)
    out.columns = [f"team__{c}" for c in out.columns]
    return out

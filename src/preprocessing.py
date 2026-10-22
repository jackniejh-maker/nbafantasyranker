"""Raw canonical tables -> processed modelling tables.

Everything is derived from box-score counting stats (no downloaded black-box metrics):
usage, assist/rebound/steal/block/turnover rates, TS%, eFG%, FT rate and 3PA rate use the standard
possession-based formulas, with team totals taken over the exact window each player spent with a team
(so traded players are handled per stint).  Each stint is preserved; season rows combine stints.

Tables produced (see README for column lists):
  player_games, team_games, stints, player_seasons, team_seasons, league, player_info,
  rosters (end-of-season), opening_rosters (preseason proxy), head_coaches, current_roster, ...
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_quality import (DataQuality, check_coverage, check_current, check_identity,
                           check_minutes_consistency, check_player_seasons)
from .fantasy_scoring import add_fantasy_columns, fantasy_points
from .utils import (load_table, log, normalize_name, position_group, save_table, season_ref_date,
                    stable_hash, table_exists)

PREPROCESS_VERSION = "1.0"
STAT_COLS = ["min", "fgm", "fga", "ftm", "fta", "fg3m", "fg3a", "oreb", "dreb", "reb", "ast", "stl", "blk",
             "tov", "pf", "pts"]
REQUIRED = ["min", "fgm", "fga", "ftm", "fta", "fg3m", "reb", "ast", "stl", "blk", "tov", "pts"]
TEAM_STATS = ["fgm", "fga", "ftm", "fta", "fg3m", "fg3a", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pts"]
PG_IDS = ["year", "game_id", "player_id", "player_name", "team"]
RATE_COLS = ["usg", "ast_pct", "reb_pct", "oreb_pct", "dreb_pct", "stl_pct", "blk_pct", "tov_pct"]


@dataclass
class ProcessedData:
    player_games: pd.DataFrame
    team_games: pd.DataFrame
    stints: pd.DataFrame
    player_seasons: pd.DataFrame
    team_seasons: pd.DataFrame
    league: pd.DataFrame
    player_info: pd.DataFrame
    rosters: pd.DataFrame
    opening_rosters: pd.DataFrame
    head_coaches: pd.DataFrame
    current_roster: pd.DataFrame | None = None
    current_coaches: pd.DataFrame | None = None
    injuries: pd.DataFrame | None = None
    consensus: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)


HIST_TABLES = ["player_games", "team_games", "stints", "player_seasons", "team_seasons", "league",
               "player_info", "rosters", "opening_rosters", "head_coaches"]


# ====================================================================== cleaning
def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return df


def clean_player_games(raw: pd.DataFrame, dq: DataQuality) -> pd.DataFrame:
    df = _coerce_numeric(raw.copy(), STAT_COLS + ["plus_minus"])
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["game_id"] = df["game_id"].astype(str)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    def drop(mask, reason):
        nonlocal df
        dq.drop("player_games", reason, df[mask], PG_IDS)
        df = df[~mask]

    drop(df["player_id"].isna() | df["game_date"].isna() | df["year"].isna() | (df["game_id"] == "nan"),
         "missing player_id / game_id / game_date")
    drop(df.duplicated(["game_id", "player_id"], keep="first"), "duplicate (game_id, player_id) row")
    drop(df[REQUIRED].isna().any(axis=1), "missing box-score statistic")
    drop((df["min"] <= 0) | (df["min"] > 70), "invalid minutes (<=0 or >70)")
    drop((df[REQUIRED] < 0).any(axis=1), "negative counting statistic")
    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["player_id"] = df["player_id"].astype("int64")
    for c in ("fg3a", "oreb", "dreb", "pf"):           # optional columns: fill, but never silently
        miss = df[c].isna()
        if miss.any():
            dq.transform("player_games", f"missing {c} filled", df[miss], PG_IDS)
            df.loc[miss, c] = df.loc[miss, "fg3m"] if c == "fg3a" else 0.0
    for made, att in (("fgm", "fga"), ("ftm", "fta"), ("fg3m", "fg3a")):
        bad = df[made] > df[att]
        if bad.any():
            dq.transform("player_games", f"{made} > {att}: attempts raised to makes", df[bad], PG_IDS,
                         detail="attempts were smaller than makes")
            df.loc[bad, att] = df.loc[bad, made]
    bad = df["fg3m"] > df["fgm"]
    if bad.any():
        dq.transform("player_games", "fg3m > fgm: fgm raised to fg3m", df[bad], PG_IDS)
        df.loc[bad, "fgm"] = df.loc[bad, "fg3m"]
        df.loc[bad, "fga"] = np.maximum(df.loc[bad, "fga"], df.loc[bad, "fgm"])
    pts_calc = 2 * (df["fgm"] - df["fg3m"]) + 3 * df["fg3m"] + df["ftm"]
    mism = (pts_calc - df["pts"]).abs() > 0.5
    dq.issue("box-score PTS inconsistent with FGM/3PM/FTM", "warning" if mism.sum() else "info", int(mism.sum()),
             "flagged only - raw PTS is kept for fantasy scoring")
    reb_bad = (df["oreb"] + df["dreb"] - df["reb"]).abs() > 0.5
    dq.issue("OREB + DREB != REB", "warning" if reb_bad.sum() else "info", int(reb_bad.sum()), "flagged only")
    return df.reset_index(drop=True)


def clean_team_games(raw: pd.DataFrame, dq: DataQuality) -> pd.DataFrame:
    df = _coerce_numeric(raw.copy(), STAT_COLS + ["plus_minus"])
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["game_id"] = df["game_id"].astype(str)
    ids = ["year", "game_id", "team"]
    bad = df["game_date"].isna() | df["team"].isna()
    dq.drop("team_games", "missing team / date", df[bad], ids)
    df = df[~bad]
    dup = df.duplicated(["game_id", "team"], keep="first")
    dq.drop("team_games", "duplicate (game_id, team) row", df[dup], ids)
    df = df[~dup]
    bad = df[["fgm", "fga", "fta", "tov", "pts", "reb"]].isna().any(axis=1) | (df["fga"] <= 0)
    dq.drop("team_games", "missing/invalid team statistics", df[bad], ids)
    df = df[~bad].copy()
    df["min"] = df["min"].where(df["min"] > 0, 240.0).fillna(240.0)
    for c in ("fg3a", "oreb", "dreb", "stl", "blk", "pf", "ast"):
        df[c] = df[c].fillna(0.0)
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def aggregate_team_games(pg: pd.DataFrame) -> pd.DataFrame:
    """Fallback when team logs are unavailable: sum the player rows of each team-game."""
    keys = ["year", "game_id", "game_date", "team", "matchup"]
    tg = pg.groupby(keys, as_index=False)[[c for c in TEAM_STATS]].sum()
    tg["min"] = 240.0
    tg["plus_minus"] = np.nan
    tg["team_id"] = np.nan
    pts = tg[["game_id", "team", "pts"]].rename(columns={"team": "opp_team", "pts": "opp_pts"})
    tg = tg.merge(pts, on="game_id")
    tg = tg[tg["team"] != tg["opp_team"]]
    tg["wl"] = np.where(tg["pts"] > tg["opp_pts"], "W", "L")
    return tg.drop(columns=["opp_team", "opp_pts"]).reset_index(drop=True)


# ====================================================================== team level
def enrich_team_games(tg: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["year", "team", "game_date", "game_id"]).reset_index(drop=True)
    tg["team_game_no"] = tg.groupby(["year", "team"]).cumcount() + 1
    tg["home"] = tg["matchup"].astype(str).str.contains("vs.", regex=False)
    tg["opp"] = tg["matchup"].astype(str).str.split().str[-1]
    o = tg[["game_id", "team"] + TEAM_STATS].rename(columns={"team": "opp_team", **{c: f"opp_{c}" for c in TEAM_STATS}})
    m = tg.merge(o, on="game_id", how="left")
    m = m[m["opp_team"].isna() | (m["team"] != m["opp_team"])].drop_duplicates(["game_id", "team"]).copy()
    have = m["opp_fga"].notna()
    with np.errstate(divide="ignore", invalid="ignore"):
        poss_t = m["fga"] + 0.4 * m["fta"] - 1.07 * (m["oreb"] / (m["oreb"] + m["opp_dreb"])) * (m["fga"] - m["fgm"]) + m["tov"]
        poss_o = m["opp_fga"] + 0.4 * m["opp_fta"] - 1.07 * (m["opp_oreb"] / (m["opp_oreb"] + m["dreb"])) * (m["opp_fga"] - m["opp_fgm"]) + m["opp_tov"]
    simple = m["fga"] + 0.44 * m["fta"] - m["oreb"] + m["tov"]
    poss = np.where(have, 0.5 * (poss_t + poss_o), simple)
    m["poss"] = pd.Series(poss, index=m.index).replace([np.inf, -np.inf], np.nan).fillna(simple)
    m = m.sort_values(["year", "team", "game_date", "game_id"]).reset_index(drop=True)
    return m


def build_team_seasons(tg: pd.DataFrame) -> pd.DataFrame:
    g = tg.groupby(["year", "team"])
    ts = g.agg(games=("game_id", "nunique"), min=("min", "sum"), poss=("poss", "sum"), pts=("pts", "sum"),
               opp_pts=("opp_pts", lambda s: s.sum(min_count=1)), fga=("fga", "sum"), fta=("fta", "sum"),
               fg3a=("fg3a", "sum"), ast=("ast", "sum"), reb=("reb", "sum"), tov=("tov", "sum"),
               fgm=("fgm", "sum")).reset_index()
    ts["pace"] = 48.0 * ts["poss"] / (ts["min"] / 5.0)
    ts["ortg"] = 100.0 * ts["pts"] / ts["poss"]
    ts["drtg"] = 100.0 * ts["opp_pts"] / ts["poss"]
    ts["netrtg"] = ts["ortg"] - ts["drtg"]
    ts["tpar"] = ts["fg3a"] / ts["fga"]            # team shot profile: 3PA share of FGA
    ts["ftr"] = ts["fta"] / ts["fga"]              # ... and free-throw rate
    ts["ast_rate"] = ts["ast"] / ts["fgm"]         # share of made baskets that were assisted (ball movement)
    for c in ("fga", "fta", "fg3a", "ast", "reb", "tov", "pts", "poss"):
        ts[f"{c}_pg"] = ts[c] / ts["games"]
    return ts


def build_league(ts: pd.DataFrame) -> pd.DataFrame:
    lg = ts.groupby("year").agg(games=("games", "median"), pace=("pace", "mean"), ortg=("ortg", "mean"),
                                teams=("team", "nunique")).reset_index()
    return lg


# ====================================================================== stints / seasons
def _advanced(st: pd.DataFrame) -> pd.DataFrame:
    for c in ("reb", "oreb", "dreb", "fga", "fg3a"):                  # symmetric fallback when no opponent data
        own = st[f"tm_{c}"]
        opp_c = {"oreb": "dreb", "dreb": "oreb"}.get(c, c)
        st[f"tm_opp_{c}"] = st[f"tm_opp_{c}"].fillna(st[f"tm_{opp_c}"]) if f"tm_opp_{c}" in st else own
    mins = st["min"]
    tmm5 = st["tm_min"] / 5.0
    used = st["fga"] + 0.44 * st["fta"] + st["tov"]
    tm_used = st["tm_fga"] + 0.44 * st["tm_fta"] + st["tm_tov"]
    with np.errstate(divide="ignore", invalid="ignore"):
        st["usg"] = 100 * used * tmm5 / (mins * tm_used)
        st["ast_pct"] = 100 * st["ast"] / ((mins / tmm5) * st["tm_fgm"] - st["fgm"])
        st["reb_pct"] = 100 * st["reb"] * tmm5 / (mins * (st["tm_reb"] + st["tm_opp_reb"]))
        st["oreb_pct"] = 100 * st["oreb"] * tmm5 / (mins * (st["tm_oreb"] + st["tm_opp_dreb"]))
        st["dreb_pct"] = 100 * st["dreb"] * tmm5 / (mins * (st["tm_dreb"] + st["tm_opp_oreb"]))
        st["stl_pct"] = 100 * st["stl"] * tmm5 / (mins * st["tm_poss"])
        st["blk_pct"] = 100 * st["blk"] * tmm5 / (mins * (st["tm_opp_fga"] - st["tm_opp_fg3a"]))
        st["tov_pct"] = 100 * st["tov"] / used
        st["ts"] = st["pts"] / (2 * (st["fga"] + 0.44 * st["fta"]))
        st["efg"] = (st["fgm"] + 0.5 * st["fg3m"]) / st["fga"]
        st["ftr"] = st["fta"] / st["fga"]
        st["tpar"] = st["fg3a"] / st["fga"]
    st["poss_used"] = used
    for c in RATE_COLS + ["ts", "efg", "ftr", "tpar"]:
        st[c] = st[c].replace([np.inf, -np.inf], np.nan)
    return st


def _fantasy_rates(df: pd.DataFrame, w: dict) -> pd.DataFrame:
    df = add_fantasy_columns(df, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["mpg"] = df["min"] / df["gp"]
        df["fppm"] = df["fp"] / df["min"]
        df["fppg"] = df["fp"] / df["gp"]
        df["fp_per_fga"] = df["fp"] / df["fga"]
        df["fp_per_poss"] = df["fp"] / df["poss_used"]
        df["fp_per_usg_pt"] = df["fppm"] / df["usg"]
        for s in ("fga", "fta", "fg3a", "ast", "reb", "stl", "blk", "tov", "pts"):
            df[f"{s}_pm"] = df[s] / df["min"]
    return df.replace([np.inf, -np.inf], np.nan)


def build_stints(pg: pd.DataFrame, tg: pd.DataFrame, w: dict) -> pd.DataFrame:
    sums = {c: (c, "sum") for c in STAT_COLS}
    st = pg.groupby(["player_id", "year", "team"], sort=False).agg(
        player_name=("player_name", "last"), gp=("game_id", "nunique"), first_date=("game_date", "min"),
        last_date=("game_date", "max"), first_tg=("team_game_no", "min"), last_tg=("team_game_no", "max"),
        **sums).reset_index()
    wc = ["min", "fgm", "fga", "fta", "tov", "reb", "oreb", "dreb", "fg3a", "poss",
          "opp_reb", "opp_oreb", "opp_dreb", "opp_fga", "opp_fg3a"]
    t = tg[["year", "team", "game_date"] + wc].rename(columns={c: f"tm_{c}" for c in wc})
    m = st[["player_id", "year", "team", "first_date", "last_date"]].merge(t, on=["year", "team"])
    m = m[(m["game_date"] >= m["first_date"]) & (m["game_date"] <= m["last_date"])]
    win = m.groupby(["player_id", "year", "team"])[[f"tm_{c}" for c in wc]].sum(min_count=1).reset_index()
    st = st.merge(win, on=["player_id", "year", "team"], how="left")
    st = _advanced(st)
    return _fantasy_rates(st, w)


def late_features(pg: pd.DataFrame, w: dict, n: int = 20) -> pd.DataFrame:
    d = pg.sort_values(["player_id", "year", "game_date"])
    tail = d.groupby(["player_id", "year"]).tail(n)
    tail = tail.assign(fp=fantasy_points(tail, w))
    g = tail.groupby(["player_id", "year"]).agg(late_n=("game_id", "nunique"), late_min=("min", "sum"),
                                                late_fp=("fp", "sum")).reset_index()
    g["late_mpg"] = g["late_min"] / g["late_n"]
    g["late_fppm"] = g["late_fp"] / g["late_min"]
    return g[["player_id", "year", "late_n", "late_mpg", "late_fppm"]]


def build_player_seasons(st: pd.DataFrame, pg: pd.DataFrame, starters: pd.DataFrame | None, w: dict) -> pd.DataFrame:
    keys = ["player_id", "year"]
    sums = {c: (c, "sum") for c in STAT_COLS}
    ps = st.groupby(keys, sort=False).agg(
        player_name=("player_name", "last"), gp=("gp", "sum"), n_teams=("team", "nunique"),
        first_date=("first_date", "min"), last_date=("last_date", "max"), **sums).reset_index()
    ps["traded"] = ps["n_teams"] > 1
    by_first = st.sort_values("first_date").groupby(keys)["team"].first().rename("first_team")
    by_last = st.sort_values("last_date").groupby(keys)["team"].last().rename("last_team")
    by_min = st.sort_values("min").groupby(keys)["team"].last().rename("primary_team")
    last = st.sort_values("last_date").groupby(keys).tail(1).set_index(keys)
    extra = pd.DataFrame({"last_stint_min": last["min"], "last_stint_gp": last["gp"], "last_stint_fppm": last["fppm"],
                          "last_stint_mpg": last["mpg"]})
    ps = ps.merge(by_first, on=keys).merge(by_last, on=keys).merge(by_min, on=keys).merge(extra, on=keys, how="left")
    for c in RATE_COLS + ["tm_poss"]:                      # minutes-weighted combination across stints
        if c in st.columns:
            ok = st[c].notna()
            num = (st[c] * st["min"]).where(ok).groupby([st["player_id"], st["year"]]).sum(min_count=1)
            den = st["min"].where(ok).groupby([st["player_id"], st["year"]]).sum(min_count=1)
            ps = ps.merge((num / den).rename(c).reset_index(), on=keys, how="left")
    # TS/eFG/FTr/3PAr/poss_used are recomputed from season totals
    ps["poss_used"] = ps["fga"] + 0.44 * ps["fta"] + ps["tov"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ps["ts"] = ps["pts"] / (2 * (ps["fga"] + 0.44 * ps["fta"]))
        ps["efg"] = (ps["fgm"] + 0.5 * ps["fg3m"]) / ps["fga"]
        ps["ftr"] = ps["fta"] / ps["fga"]
        ps["tpar"] = ps["fg3a"] / ps["fga"]
    # availability denominator: games the player's team(s) played that season
    tgc = pg.groupby(["team", "year"])["game_id"].nunique().rename("tg").reset_index()
    den = st.merge(tgc, on=["team", "year"], how="left").groupby(keys)["tg"].max().rename("team_games").reset_index()
    ps = ps.merge(den, on=keys, how="left")
    ps["avail"] = (ps["gp"] / ps["team_games"]).clip(0, 1)
    if starters is not None and len(starters):
        s = starters.groupby(keys, as_index=False)["gs"].sum()
        ps = ps.merge(s, on=keys, how="left")
        ps["gs"] = ps["gs"].fillna(0.0).clip(upper=ps["gp"])
        ps["gs_rate"] = ps["gs"] / ps["gp"]
    else:
        ps["gs"] = np.nan
        ps["gs_rate"] = np.nan
    ps = ps.merge(late_features(pg, w), on=keys, how="left")
    return _fantasy_rates(ps, w)


# ====================================================================== people / rosters
def _coach_key(v) -> str:
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def head_coaches(coaches: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["year", "team", "coach_id", "coach_name"]
    if coaches is None or len(coaches) == 0:
        return pd.DataFrame(columns=cols)
    c = coaches.copy()
    c = c[c["is_head"].fillna(False).astype(bool)].copy()
    c["coach_id"] = c["coach_id"].map(_coach_key)
    c = c.drop_duplicates(["year", "team"], keep="first")
    return c[cols].reset_index(drop=True)


def estimate_position(df: pd.DataFrame) -> pd.Series:
    reb = df.get("reb_pct", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    ast = df.get("ast_pct", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    return pd.Series(np.select([reb >= 14, ast >= 18], ["C", "G"], "F"), index=df.index)


def build_player_info(player_index, rosters, current_roster, ps) -> pd.DataFrame:
    cols = ["player_id", "player_name", "position", "height_in", "weight", "college", "country", "draft_year",
            "draft_round", "draft_pick", "from_year", "to_year"]
    info = player_index.copy() if player_index is not None and len(player_index) else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in info.columns:
            info[c] = np.nan
    info = info[cols].drop_duplicates("player_id", keep="last")
    frames = [r for r in (rosters, current_roster) if r is not None and len(r)]
    if frames:
        rb = pd.concat(frames, ignore_index=True)
        rb = rb[rb["player_id"].notna() & (rb["player_id"] > 0)]
        bd = rb.dropna(subset=["birth_date"]).sort_values("birth_date").drop_duplicates("player_id")[["player_id", "birth_date"]]
        info = info.merge(bd, on="player_id", how="outer")
        if "exp" in rb.columns:
            deb = (rb["year"] - rb["exp"]).groupby(rb["player_id"]).min().rename("debut_roster").reset_index()
            info = info.merge(deb, on="player_id", how="left")
    if "birth_date" not in info.columns:
        info["birth_date"] = pd.NaT
    if "debut_roster" not in info.columns:
        info["debut_roster"] = np.nan
    ids = ps[["player_id"]].drop_duplicates()
    info = info.merge(ids, on="player_id", how="outer")
    nm = ps.sort_values("year").drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
    info["player_name"] = info["player_name"].fillna(info["player_id"].map(nm))
    first_seen = ps.groupby("player_id")["year"].min()
    info["first_seen"] = info["player_id"].map(first_seen)
    info["debut_year"] = info["from_year"].fillna(info["debut_roster"]).fillna(info["first_seen"])
    info["drafted"] = info["draft_pick"].notna()
    return info.drop_duplicates("player_id").reset_index(drop=True)


def attach_demographics(ps: pd.DataFrame, info: pd.DataFrame, rosters: pd.DataFrame | None) -> pd.DataFrame:
    ps = ps.copy()
    pos = pd.Series(np.nan, index=ps.index, dtype=object)
    if rosters is not None and len(rosters):
        by_year = rosters.dropna(subset=["position"]).drop_duplicates(["player_id", "year"], keep="last")
        key = pd.MultiIndex.from_frame(ps[["player_id", "year"]])
        pos = pd.Series(by_year.set_index(["player_id", "year"])["position"].reindex(key).to_numpy(), index=ps.index)
        latest = by_year.sort_values("year").drop_duplicates("player_id", keep="last").set_index("player_id")["position"]
        pos = pos.fillna(ps["player_id"].map(latest))
    pos = pos.fillna(ps["player_id"].map(info.set_index("player_id")["position"]))
    pos = pos.fillna(estimate_position(ps))
    ps["position"] = pos.astype(str)
    ps["pos_group"] = ps["position"].map(position_group).replace("", "F")
    bd = pd.to_datetime(ps["player_id"].map(info.set_index("player_id")["birth_date"]))
    ref = pd.to_datetime(ps["year"].map(lambda y: pd.Timestamp(season_ref_date(int(y)))))
    ps["age"] = (ref - bd).dt.days / 365.25
    debut = ps["player_id"].map(info.set_index("player_id")["debut_year"])
    ps["exp"] = (ps["year"] - debut).clip(lower=0)
    return ps


def build_opening_rosters(stints: pd.DataFrame, rosters: pd.DataFrame | None, ps: pd.DataFrame, k: int) -> pd.DataFrame:
    """Preseason roster proxy for historical seasons (documented in README/feature policy).

    A: players who appeared for a team within its first `k` games (earliest stint).
    B: "returning" players - on the team's end-of-season roster or played for it, AND their previous-season
       final team was the same team (captures players injured at the start / who never played).
    Mid-season arrivals from other teams or the free-agent pool are NOT part of the opening roster.
    """
    first = stints.sort_values(["player_id", "year", "first_date"]).groupby(["player_id", "year"], as_index=False).first()
    a = first[first["first_tg"] <= k][["player_id", "year", "team"]].assign(src="opening")
    prev = ps[["player_id", "year", "last_team"]].assign(year=lambda d: d["year"] + 1).rename(columns={"last_team": "prev_team"})
    cand = [first[["player_id", "year", "team"]]]
    if rosters is not None and len(rosters):
        cand.append(rosters[["player_id", "year", "team"]])
    cand = pd.concat(cand).drop_duplicates(["player_id", "year", "team"]).merge(prev, on=["player_id", "year"], how="left")
    b = cand[cand["team"] == cand["prev_team"]][["player_id", "year", "team"]].assign(src="returning")
    out = pd.concat([a, b], ignore_index=True).drop_duplicates(["player_id", "year"], keep="first")
    return out.reset_index(drop=True)


def resolve_current_roster(cur: pd.DataFrame | None, info: pd.DataFrame, ps: pd.DataFrame, dq: DataQuality,
                           aliases: dict | None = None) -> pd.DataFrame | None:
    """Make sure every current-roster row has a player_id (ESPN fallback rosters carry names only)."""
    if cur is None or len(cur) == 0:
        return cur
    cur = cur.copy()
    if cur["player_id"].isna().any():
        last_year = ps.groupby("player_id")["year"].max()
        cand = pd.concat([info[["player_id", "player_name"]], ps[["player_id", "player_name"]]]).dropna()
        cand = cand.drop_duplicates()
        cand["key"] = cand["player_name"].map(lambda n: normalize_name(n, aliases))
        cand["ly"] = cand["player_id"].map(last_year).fillna(0)
        best = cand.sort_values("ly").drop_duplicates("key", keep="last").set_index("key")["player_id"]
        keys = cur["player_name"].map(lambda n: normalize_name(n, aliases))
        resolved = keys.map(best)
        miss = cur["player_id"].isna()
        cur.loc[miss, "player_id"] = resolved[miss]
        still = cur["player_id"].isna()
        if still.any():
            dq.issue("current roster names not found in history (assigned provisional ids)", "info", int(still.sum()),
                     "treated as new players", cur.loc[still, "player_name"].head(5).tolist())
            cur.loc[still, "player_id"] = -(np.arange(still.sum()) + 1)
    cur["player_id"] = cur["player_id"].astype("int64")
    dupe = cur.duplicated("player_id", keep="first")
    dq.drop("current_roster", "duplicate player on current rosters", cur[dupe], ["player_id", "player_name", "team"])
    return cur[~dupe].reset_index(drop=True)


# ====================================================================== orchestration
def _fingerprint(cfg, raw) -> str:
    return stable_hash({"v": PREPROCESS_VERSION, "scoring": cfg.scoring, "first": cfg.first_year,
                        "target": cfg.target_year, "k": cfg.get("eligibility.opening_roster_games", 10),
                        "files": {k: v.get("sha256") for k, v in raw.manifest.items()
                                  if k.split("/")[0] in ("game_logs", "team_game_logs", "starters", "rosters", "coaches", "player_index")}})


def build_history(raw, cfg, dq: DataQuality) -> dict:
    w = cfg.scoring
    pg = clean_player_games(raw.player_games, dq)
    if raw.team_games is not None and len(raw.team_games):
        tg_raw = clean_team_games(raw.team_games, dq)
    else:
        dq.issue("team game logs unavailable", "warning", 1, "derived by summing player rows (minutes assumed 240)")
        tg_raw = aggregate_team_games(pg)
    tg = enrich_team_games(tg_raw)
    pg = pg.merge(tg[["game_id", "team", "team_game_no", "home", "opp"]], on=["game_id", "team"], how="left")
    orphan = pg["team_game_no"].isna()
    dq.drop("player_games", "game not present in team logs", pg[orphan], PG_IDS)
    pg = pg[~orphan].reset_index(drop=True)
    pg["fp"] = fantasy_points(pg, w)
    stints = build_stints(pg, tg, w)
    ps = build_player_seasons(stints, pg, raw.starters, w)
    ts = build_team_seasons(tg)
    league = build_league(ts)
    rosters = raw.rosters.copy() if raw.rosters is not None else None
    cur = raw.current_rosters
    info = build_player_info(raw.player_index, rosters, cur[cur["player_id"].notna()] if cur is not None else None, ps)
    ps = attach_demographics(ps, info, rosters)
    stints = stints.merge(ps[["player_id", "year", "pos_group", "position"]], on=["player_id", "year"], how="left")
    opening = build_opening_rosters(stints, rosters, ps, int(cfg.get("eligibility.opening_roster_games", 10)))
    coaches = head_coaches(raw.coaches)
    # ---- data-quality checks that need the processed tables
    check_identity(pg, dq, cfg.aliases)
    check_coverage(pg, tg, cfg, dq)
    check_minutes_consistency(pg, dq)
    check_player_seasons(ps, stints, dq, cfg.season_length)
    return {"player_games": pg, "team_games": tg, "stints": stints, "player_seasons": ps, "team_seasons": ts,
            "league": league, "player_info": info, "rosters": rosters if rosters is not None else pd.DataFrame(),
            "opening_rosters": opening, "head_coaches": coaches}


def load_or_build(cfg, raw, dq: DataQuality, force: bool = False) -> ProcessedData:
    fp = _fingerprint(cfg, raw)
    fp_path = cfg.processed_dir / "_fingerprint.json"
    cached = None
    if not force and fp_path.exists() and json.loads(fp_path.read_text()).get("fingerprint") == fp \
            and all(table_exists(cfg.processed_dir / t) for t in HIST_TABLES) \
            and table_exists(cfg.processed_dir / "_dq_issues"):
        cached = {t: load_table(cfg.processed_dir / t) for t in HIST_TABLES}
        if any(v is None for v in cached.values()):
            cached = None
    if cached is not None:
        log.info("processed tables are up to date (fingerprint %s) - loaded from cache", fp)
        issues = load_table(cfg.processed_dir / "_dq_issues")
        dq.issues = issues.to_dict("records")
        for name, bucket in (("_dq_drops", dq._drops), ("_dq_xforms", dq._xforms)):
            t = load_table(cfg.processed_dir / name)
            if t is not None and len(t):
                bucket.append(t)
        hist = cached
    else:
        log.info("building processed tables (raw data changed or --retrain/--update-data)")
        hist = build_history(raw, cfg, dq)
        for name, df in hist.items():
            save_table(df, cfg.processed_dir / name)
        iss, drops, xf = dq.frames()
        save_table(iss, cfg.processed_dir / "_dq_issues")
        save_table(drops, cfg.processed_dir / "_dq_drops")
        save_table(xf, cfg.processed_dir / "_dq_xforms")
        fp_path.write_text(json.dumps({"fingerprint": fp}))
    cur = resolve_current_roster(raw.current_rosters, hist["player_info"], hist["player_seasons"], dq, cfg.aliases)
    check_current(cur, hist["player_seasons"], cfg.target_year, dq)
    cur_coaches = head_coaches(raw.current_coaches)
    return ProcessedData(**hist, current_roster=cur, current_coaches=cur_coaches, injuries=raw.injuries,
                         consensus=raw.consensus,
                         meta={"fingerprint": fp, "sources": {k: v.get("source") for k, v in raw.manifest.items()}})

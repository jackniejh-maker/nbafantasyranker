"""Live projection for the target season.

Pipeline:  current roster (+ manual overrides)  ->  leak-free features  ->  selected FPPM / minutes /
availability models (main pathway) or the rookie/low-history models (newcomer pathway)  ->  team-minutes
reconciliation (only if validation showed it helps)  ->  manual overrides  ->  known-injury cap  ->
(optional) empirical-Bayes in-season update.  The Monte Carlo step (simulation.py) then turns these point
estimates into distributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .consensus import match_names
from .utils import log
from .validation import reconcile_minutes

OVERRIDE_COLS = ["player_name", "player_id", "expected_minutes_override", "expected_games_override",
                 "team_override", "exclude", "notes"]
OUT_SEASON = r"out for the season|out for season|season-ending|season ending|out for year|done for the season"


@dataclass
class ProjectionSet:
    proj: pd.DataFrame        # point projections + simulation inputs, indexed by player_id
    X: pd.DataFrame           # live feature matrix (for explanations)
    meta: pd.DataFrame        # descriptive columns (ref_* = last-season stats, baselines, ...)
    inseason: bool
    overrides: pd.DataFrame
    injuries: pd.DataFrame


# ====================================================================== manual inputs
def load_overrides(cfg) -> pd.DataFrame:
    p = cfg.project_path(cfg.get("paths.overrides_csv", "config/player_overrides.csv"))
    if not p.exists():
        return pd.DataFrame(columns=OVERRIDE_COLS)
    df = pd.read_csv(p, comment="#", skip_blank_lines=True)
    for c in OVERRIDE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[OVERRIDE_COLS].dropna(how="all")


def resolve_overrides(ov: pd.DataFrame, roster: pd.DataFrame, cfg) -> pd.DataFrame:
    """Return the overrides indexed by player_id (matching on id first, then on normalised name)."""
    if ov is None or len(ov) == 0:
        return pd.DataFrame(columns=OVERRIDE_COLS).set_index("player_id")
    ov = ov.copy()
    r = roster.reset_index()[["player_id", "player_name", "team"]]
    byname = match_names(ov["player_name"].fillna(""), None, r, cfg.aliases)
    ov["player_id"] = pd.to_numeric(ov["player_id"], errors="coerce").fillna(byname)
    bad = ov["player_id"].isna()
    if bad.any():
        log.warning("overrides: could not match %s - ignored", ov.loc[bad, "player_name"].tolist())
    ov = ov.dropna(subset=["player_id"])
    ov["player_id"] = ov["player_id"].astype("int64")
    ov["exclude"] = ov["exclude"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return ov.drop_duplicates("player_id", keep="last").set_index("player_id")


def prospect_adjustments(cfg, roster: pd.DataFrame) -> pd.DataFrame:
    p = cfg.project_path(cfg.get("paths.external_prospects_csv", "data/external/prospects.csv"))
    if not p.exists():
        return pd.DataFrame(columns=["fppm_adjustment", "mpg_adjustment"])
    d = pd.read_csv(p, comment="#")
    d["player_id"] = match_names(d["player_name"], None, roster.reset_index()[["player_id", "player_name", "team"]], cfg.aliases)
    d = d.dropna(subset=["player_id"]).drop_duplicates("player_id", keep="last")
    d["player_id"] = d["player_id"].astype("int64")
    for c in ("fppm_adjustment", "mpg_adjustment"):
        if c not in d.columns:
            d[c] = 0.0
    return d.set_index("player_id")[["fppm_adjustment", "mpg_adjustment"]].fillna(0.0)


def injury_effects(inj: pd.DataFrame | None, roster: pd.DataFrame, cfg, games_rem: pd.Series) -> pd.DataFrame:
    """Translate the current injury report into 'games still to be missed' (known information, not a forecast)."""
    cols = ["inj_status", "inj_note", "games_out"]
    ic = cfg.get("injuries", {})
    if inj is None or len(inj) == 0 or not ic.get("enabled", True):
        return pd.DataFrame(columns=cols)
    d = inj.copy()
    r = roster.reset_index()[["player_id", "player_name", "team"]]
    use_ids = cfg.get("source_profile") == "espn" and "espn_id" in d.columns
    d["player_id"] = match_names(d["player_name"], d["team"] if "team" in d.columns else None, r, cfg.aliases,
                                 espn_ids=d["espn_id"] if use_ids else None)
    unm = d["player_id"].isna()
    if unm.any():
        log.info("injury report: %d names not on a current roster (ignored), e.g. %s", unm.sum(), d.loc[unm, "player_name"].head(3).tolist())
    d = d.dropna(subset=["player_id"]).drop_duplicates("player_id")
    d["player_id"] = d["player_id"].astype("int64")
    text = (d["status"].fillna("") + " " + d["injury_type"].fillna("") + " " + d["detail"].fillna("") + " "
            + d["comment"].fillna("")).str.lower()
    is_out = d["status"].fillna("").str.lower().str.contains("out")
    season_open = max(cfg.today, date(cfg.target_year, 10, 22))
    ret = pd.to_datetime(d["return_date"], errors="coerce")
    days = (ret - pd.Timestamp(season_open)).dt.days.clip(lower=0)
    gpd = float(ic.get("games_per_day", 0.5))
    games_out = np.where(text.str.contains(OUT_SEASON), np.inf,
                         np.where(is_out & ret.notna(), days * gpd, np.where(is_out, float(ic.get("unknown_duration_games", 0)), 0.0)))
    note = d["status"].fillna("") + np.where(d["injury_type"].notna(), " - " + d["injury_type"].fillna(""), "")
    note = note + np.where(ret.notna(), " (return " + ret.dt.strftime("%b %d").fillna("") + ")", "")
    return pd.DataFrame({"inj_status": d["status"].to_numpy(), "inj_note": np.asarray(note), "games_out": games_out},
                        index=d["player_id"].to_numpy())


# ====================================================================== in-season Bayesian update
def inseason_update(proj: pd.DataFrame, pdx, cfg, eb: dict) -> bool:
    """Blend the preseason projection with games already played (empirical-Bayes weights); remaining games only."""
    T = cfg.target_year
    ic = cfg.get("in_season", {})
    cur = pdx.player_seasons[pdx.player_seasons["year"] == T].drop_duplicates("player_id").set_index("player_id")
    if not ic.get("enabled", True) or len(cur) == 0 or cur["gp"].sum() == 0:
        return False
    played = proj["team"].map(pdx.team_seasons[pdx.team_seasons["year"] == T].set_index("team")["games"]).fillna(0.0)
    proj["games_rem"] = np.clip(cfg.season_length - played, 0, None)
    c = cur.reindex(proj.index)
    m, gp = c["min"].fillna(0.0), c["gp"].fillna(0.0)
    k_min, k_g, k_a = float(eb["k_minutes"]), float(eb["k_games"]), float(ic.get("avail_k_games", 20))
    proj["fppm"] = (k_min * proj["fppm"] + m * c["fppm"].fillna(proj["fppm"])) / (k_min + m)
    proj["mpg"] = (k_g * proj["mpg"] + gp * c["mpg"].fillna(proj["mpg"])) / (k_g + gp)
    obs = (gp / played.replace(0, np.nan)).clip(0, 1).fillna(proj["avail"])
    proj["avail"] = (k_a * proj["avail"] + played * obs) / (k_a + played)
    proj["mult"] = np.sqrt(k_min / (k_min + m))
    proj["fp_to_date"] = c["fp"].fillna(0.0)
    proj["gp_to_date"] = gp
    log.info("in-season update applied: %d players have games played this season (K_minutes=%.0f, K_games=%.1f)",
             int((gp > 0).sum()), k_min, k_g)
    return True


# ====================================================================== main entry point
def project(pdx, cfg, models, builder) -> ProjectionSet:
    T = cfg.target_year
    roster = builder.roster_for(T)
    ov = resolve_overrides(load_overrides(cfg), roster, cfg)
    if len(ov):
        t = ov["team_override"].dropna()
        t = t[t.index.isin(roster.index)]
        if len(t):
            roster.loc[t.index, "team"] = t.astype(str)
        roster = roster.drop(index=ov.index[ov["exclude"]], errors="ignore")
    min_over = {int(k): float(v) for k, v in ov["expected_minutes_override"].dropna().items()} if len(ov) else {}
    ff = builder.build(T, roster=roster, minutes_override=min_over)
    X, meta = ff.X, ff.meta.copy()
    main = (meta["pathway"] == "main").to_numpy()
    raw = {}
    for t, base in (("fppm", "base_fppm"), ("mpg", "base_mpg"), ("avail", "base_avail")):
        pm = models.targets[t].final.predict(X, meta[base])
        pn = models.newcomer[t].final.predict(X, meta[base])
        raw[t] = np.where(main, pm, pn)
    pa = prospect_adjustments(cfg, roster)
    max_mpg = float(cfg.get("minutes_reconciliation.max_mpg", 41.0))
    fppm = pd.Series(raw["fppm"], index=meta.index)
    mpg = pd.Series(raw["mpg"], index=meta.index)
    if len(pa):
        fppm = fppm.add(pa["fppm_adjustment"].reindex(meta.index).fillna(0.0))
        mpg = mpg.add(pa["mpg_adjustment"].reindex(meta.index).fillna(0.0))
    fppm = fppm.clip(lower=0.25)
    mpg = mpg.clip(2.0, max_mpg)
    avail = pd.Series(raw["avail"], index=meta.index).clip(0.02, 1.0)
    mpg_rec = reconcile_minutes(meta["team"], pd.Series(T, index=meta.index), mpg, avail, models.recon_mode, cfg)
    proj = pd.DataFrame({
        "player_name": meta["player_name"], "team": meta["team"], "position": meta["position"], "age": meta["age"],
        "pathway": meta["pathway"], "is_rookie": meta["is_rookie"], "fppm": fppm, "mpg": mpg_rec, "avail": avail,
        "mpg_model": mpg, "games_rem": float(cfg.season_length), "fp_to_date": 0.0, "gp_to_date": 0.0, "mult": 1.0,
        "prior_min": meta["prior_min"], "n_prior_seasons": meta["n_prior_seasons"],
        "prior_mpg": meta["ref_mpg"].fillna(meta["mpg_budget"]), "ref_total": meta["ref_fp"],
        "override_applied": ""}, index=meta.index)
    inseason = inseason_update(proj, pdx, cfg, models.eb)
    og = pd.Series(dtype=float)
    if len(ov):                                                    # manual overrides win over every model
        om = ov["expected_minutes_override"].dropna()
        om = om[om.index.isin(proj.index)]
        proj.loc[om.index, "mpg"] = om.clip(1.0, max_mpg)
        proj.loc[om.index, "override_applied"] += "minutes;"
        og = ov["expected_games_override"].dropna()
        og = og[og.index.isin(proj.index)]
        proj.loc[og.index, "avail"] = (og / proj.loc[og.index, "games_rem"].clip(lower=1)).clip(0, 1)
        proj.loc[og.index, "override_applied"] += "games;"
        tm = ov["team_override"].dropna()
        proj.loc[tm.index.intersection(proj.index), "override_applied"] += "team;"
    inj = injury_effects(pdx.injuries, roster, cfg, proj["games_rem"])
    proj["inj_status"], proj["inj_note"], proj["games_out"] = "", "", 0.0
    if len(inj):
        j = inj.reindex(proj.index)
        has = j["games_out"].notna()
        proj.loc[has, "inj_status"] = j.loc[has, "inj_status"].fillna("")
        proj.loc[has, "inj_note"] = j.loc[has, "inj_note"].fillna("")
        proj.loc[has, "games_out"] = j.loc[has, "games_out"]
    if len(og):                                                    # a manual games override is final (no injury double count)
        proj.loc[og.index, "games_out"] = 0.0
    gr = proj["games_rem"].clip(lower=1)
    proj["avail_cap"] = np.clip(1.0 - proj["games_out"] / gr, 0.0, 1.0)
    proj["avail_exp"] = np.minimum(proj["avail"], proj["avail_cap"])
    return ProjectionSet(proj=proj, X=X, meta=meta, inseason=inseason, overrides=ov, injuries=inj)

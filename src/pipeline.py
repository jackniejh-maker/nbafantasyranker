"""End-to-end orchestration used by project.py / train.py.

 1 check cached data            2 download what is new/stale        3 update rosters
 4 update injuries/context      5 process historical data           6 generate features
 7 train / update models        8 run projections                   9 Monte Carlo simulation
10 rankings                    11 player reports                   12 plots
"""
from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from .archetypes import classify, value_analysis
from .consensus import load_consensus
from .data_quality import DataQuality
from .explanations import group_effects, linear_contributions, player_report
from .feature_engineering import FeatureBuilder
from .ingest import Ingestor
from .preprocessing import load_or_build
from .projection import project
from .ranking import assemble, finalize_columns, rank_and_score
from .simulation import simulate
from .training import Models, train_all
from .utils import load_table, log, normalize_name, slugify, utc_now_iso

VERSION = "1.0.0"


@dataclass
class Result:
    rank: pd.DataFrame
    models: Models
    pdx: object
    X: pd.DataFrame
    meta: pd.DataFrame
    cfg: object
    inseason: bool


def _arg(args, name, default=None):
    return getattr(args, name, default) if args is not None else default


# ====================================================================== steps 1-7
def prepare(cfg, args=None, train: bool = True):
    t0 = time.time()
    raw = Ingestor(cfg, offline=bool(_arg(args, "offline", False)), force_current=bool(_arg(args, "update_data", False)),
                   force_history=bool(_arg(args, "refresh_history", False))).run()
    dq = DataQuality()
    pdx = load_or_build(cfg, raw, dq, force=bool(_arg(args, "refresh_history", False)))
    if _arg(args, "crosscheck", False):
        _crosscheck(cfg, pdx, dq)
    dq.write(cfg.outputs_dir / "data_quality")
    log.info("data ready in %.1fs", time.time() - t0)
    models = train_all(pdx, cfg, force=bool(_arg(args, "retrain", False)), tune=bool(_arg(args, "tune_weights", False))) if train else None
    return pdx, models, dq


def _crosscheck(cfg, pdx, dq: DataQuality) -> None:
    """Compare games/minutes with Basketball-Reference (secondary source) for the last completed seasons."""
    from .data_sources import build_source
    from .data_sources.basketball_reference import crosscheck
    try:
        src = build_source("basketball_reference", cfg)
        for y in cfg.history_years()[-2:]:
            bad = crosscheck(pdx.player_seasons, src.season_totals(y), y)
            dq.issue(f"cross-check vs Basketball-Reference {y}", "warning" if len(bad) else "info", len(bad),
                     "games/minutes differ by more than the tolerance", bad["player_name"].head(5).tolist())
    except Exception as exc:
        dq.issue("cross-check vs Basketball-Reference", "warning", 1, f"could not run: {exc}")


# ====================================================================== steps 8-10
def run_projection(cfg, args, pdx, models) -> Result:
    builder = FeatureBuilder(pdx, cfg)
    ps = project(pdx, cfg, models, builder)
    effects = group_effects(models, ps.X, ps.meta)
    n_sims = _arg(args, "n_sims", None)
    if n_sims:
        cfg.set("simulation.n_sims", int(n_sims))
    log.info("Monte Carlo: %d simulated seasons for each of %d players", int(cfg.get("simulation.n_sims")), len(ps.proj))
    ps.proj["mult"] = ps.proj["mult"] * float(models.sim_mult)      # calibrated to ~80% out-of-sample coverage
    sim = simulate(models.pools, ps.proj, cfg, seed=_arg(args, "seed", None))
    table = assemble(ps, sim, effects, cfg)
    table = table.join(classify(pdx.player_seasons, cfg.target_year, table.index))
    cons = load_consensus(pdx, cfg, table[["player_id", "player_name", "team"]])
    table = rank_and_score(table, cfg, cons)
    return Result(rank=table, models=models, pdx=pdx, X=ps.X, meta=ps.meta, cfg=cfg, inseason=ps.inseason)


# ====================================================================== reports
def _report_text(res: Result, r: pd.Series) -> str:
    pid, pdx, models = int(r["player_id"]), res.pdx, res.models
    ps_all = pdx.player_seasons
    hist = ps_all[ps_all["player_id"] == pid].sort_values("year").tail(4)
    stints = None
    if len(hist) and bool(hist.iloc[-1].get("traded", False)):
        st = pdx.stints[(pdx.stints["player_id"] == pid) & (pdx.stints["year"] == hist.iloc[-1]["year"])].copy()
        st["mpg"] = st["min"] / st["gp"]
        stints = st
    main = r["pathway"] == "main"
    tf = models.targets["fppm"].explain if main else models.newcomer["fppm"].final
    tm = models.targets["mpg"].explain if main else models.newcomer["mpg"].final
    row = res.X.loc[[pid]]
    contribs = {"fppm": linear_contributions(tf, row), "mpg": linear_contributions(tm, row)}
    return player_report(r, hist, stints, contribs, res.cfg, len(res.rank))


def write_reports(res: Result, n: int) -> int:
    d = res.cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.txt"):
        old.unlink()
    for _, r in res.rank.head(n).iterrows():
        (d / f"{int(r['rank_total']):03d}_{slugify(r['player_name'])}.txt").write_text(_report_text(res, r), encoding="utf-8")
    return min(n, len(res.rank))


def find_player(rank: pd.DataFrame, query: str):
    """Exact normalised match, else unique substring match, else suggestions."""
    q = normalize_name(query)
    names = rank["player_name"].map(normalize_name)
    exact = rank[names == q]
    if len(exact) == 1:
        return exact.iloc[0], []
    sub = rank[names.str.contains(q, regex=False)] if q else rank.iloc[0:0]
    if len(sub) == 1:
        return sub.iloc[0], []
    cand = list(sub["player_name"]) if len(sub) else difflib.get_close_matches(query, rank["player_name"].tolist(), n=5, cutoff=0.5)
    return None, cand[:8]


# ====================================================================== outputs
def write_outputs(res: Result, args=None, models_only: bool = False) -> None:
    cfg, rank, models, pdx = res.cfg, res.rank, res.models, res.pdx
    out = cfg.outputs_dir
    out.mkdir(parents=True, exist_ok=True)
    final = finalize_columns(rank).copy()
    num = final.select_dtypes("number").columns
    final[num] = final[num].round(4)
    final.to_csv(out / "rankings.csv", index=False)
    models.performance.to_csv(out / "model_performance.csv", index=False)
    models.importance.to_csv(out / "feature_importance.csv", index=False)
    models.feature_policy.to_csv(out / "feature_policy.csv", index=False)
    models.recon_table.to_csv(out / "minutes_reconciliation.csv")
    models.pools.diagnostics.to_csv(out / "residual_pools.csv", index=False)
    models.calibration.to_csv(out / "uncertainty_calibration.csv", index=False)
    if models.game is not None:
        models.game["context_effects"].to_csv(out / "game_context_effects.csv", index=False)
    va, md = value_analysis(pdx.player_seasons, cfg, cfg.target_year)
    va.to_csv(out / "fantasy_value_analysis.csv", index=False)
    (out / "fantasy_value_analysis.md").write_text(md, encoding="utf-8")
    joblib.dump({"rank": rank, "X": res.X, "meta": res.meta, "created": utc_now_iso(), "fingerprint": models.fingerprint,
                 "target_year": cfg.target_year}, out / "projection_cache.joblib")
    summary = {
        "version": VERSION, "created": utc_now_iso(), "target_season": cfg.target_year, "demo": cfg.demo,
        "in_season_update": res.inseason, "n_players": int(len(rank)), "n_sims": int(cfg.get("simulation.n_sims")),
        "simulation_seed": cfg.get("simulation.seed"), "scoring": cfg.scoring,
        "selected_models": {k: r.selected for k, r in models.targets.items()},
        "selected_newcomer_models": {k: r.selected for k, r in models.newcomer.items()},
        "minutes_reconciliation": models.recon_mode, "test_seasons": models.test_years,
        "training_seasons": [models.train_years[0], models.train_years[-1]], "model_fingerprint": models.fingerprint,
        "empirical_bayes": models.eb, "uncertainty_multiplier": models.sim_mult, "data_sources": pdx.meta.get("sources", {}),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if cfg.demo:
        _demo_check(cfg, rank, out)


def _demo_check(cfg, rank: pd.DataFrame, out) -> None:
    """Demo only: compare projections with the hidden simulated season (never an input to anything)."""
    from scipy import stats
    from .data_sources.demo import get_league
    h = get_league(cfg).holdout
    m = rank.merge(h[["player_id", "fp", "fppg", "gp"]], on="player_id", how="inner")
    m = m[m["projected_minutes"] >= 12]
    res = {"n_players": int(len(m)),
           "spearman_total": float(stats.spearmanr(m["total_p50"], m["fp"]).statistic),
           "spearman_fppg": float(stats.spearmanr(m["fppg_p50"], m["fppg"]).statistic),
           "coverage_p10_p90_total": float(((m["fp"] >= m["total_p10"]) & (m["fp"] <= m["total_p90"])).mean()),
           "coverage_p25_p75_total": float(((m["fp"] >= m["total_p25"]) & (m["fp"] <= m["total_p75"])).mean()),
           "top50_overlap": len(set(m.nlargest(50, "total_p50")["player_id"]) & set(m.nlargest(50, "fp")["player_id"])) / 50}
    (out / "demo_holdout_check.json").write_text(json.dumps(res, indent=1))
    log.info("DEMO CHECK vs hidden simulated season: Spearman(total)=%.3f, 80%% interval coverage=%.1f%%, top-50 overlap=%.0f%%",
             res["spearman_total"], 100 * res["coverage_p10_p90_total"], 100 * res["top50_overlap"])


def print_top(rank: pd.DataFrame, n: int) -> None:
    cols = ["rank_total", "player_name", "team", "position", "age", "projected_minutes", "projected_fppm",
            "projected_fppg", "projected_games", "total_p10", "total_p50", "total_p90", "rank_fppg",
            "rank_risk_adjusted", "archetype"]
    t = rank.head(n)[cols].rename(columns={"rank_total": "rank", "projected_minutes": "min", "projected_fppm": "fppm",
                                           "projected_fppg": "fppg", "projected_games": "gms", "player_name": "player",
                                           "rank_fppg": "rk_ppg", "rank_risk_adjusted": "rk_risk"})
    fm = {"age": "{:.1f}".format, "min": "{:.1f}".format, "fppm": "{:.2f}".format, "fppg": "{:.1f}".format,
          "gms": "{:.0f}".format, "total_p10": "{:.0f}".format, "total_p50": "{:.0f}".format, "total_p90": "{:.0f}".format}
    print("\n" + t.to_string(index=False, formatters=fm) + "\n")


# ====================================================================== top-level entry points
def run_pipeline(cfg, args=None) -> Result:
    cfg.make_dirs()
    t0 = time.time()
    pdx, models, _ = prepare(cfg, args, train=True)
    res = run_projection(cfg, args, pdx, models)
    write_outputs(res, args)
    if not _arg(args, "no_reports", False):
        n = write_reports(res, max(int(_arg(args, "top", 200) or 200), 200))
        log.info("wrote %d player reports -> %s", n, cfg.reports_dir)
    if not _arg(args, "no_plots", False):
        try:
            from .visualization import make_all
            make_all(res.rank, models, pdx.player_seasons, cfg, cfg.plots_dir)
        except Exception as exc:  # plotting must never kill a run
            log.warning("plotting failed: %s", exc)
    log.info("done in %.1fs  ->  %s", time.time() - t0, cfg.outputs_dir / "rankings.csv")
    return res


def cached_result(cfg) -> Result | None:
    """Rebuild a Result from the last run (no network, no retraining) - used by `--player`."""
    cache_p = cfg.outputs_dir / "projection_cache.joblib"
    models_p = cfg.models_dir / "bundle.joblib"
    if not (cache_p.exists() and models_p.exists()):
        return None
    try:
        cache, models = joblib.load(cache_p), joblib.load(models_p)
        if cache["fingerprint"] != models.fingerprint or cache["target_year"] != cfg.target_year:
            return None
        pdx = SimpleNamespace(player_seasons=load_table(cfg.processed_dir / "player_seasons"),
                              stints=load_table(cfg.processed_dir / "stints"))
        log.info("using the projections saved at %s (run `python project.py` to refresh)", cache["created"])
        return Result(rank=cache["rank"], models=models, pdx=pdx, X=cache["X"], meta=cache["meta"], cfg=cfg, inseason=False)
    except Exception as exc:
        log.warning("could not use the cached projections (%s)", exc)
        return None

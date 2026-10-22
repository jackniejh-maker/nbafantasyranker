"""Training orchestration: features -> chronological validation -> selection -> final fit -> uncertainty pools.

The whole bundle is persisted in data/processed/models/bundle.joblib together with a fingerprint of
(config, processed data, code version); `python project.py` retrains only when something changed
(`--retrain` forces it).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import pandas as pd

from . import availability_model, fppm_model, minutes_model, rookie_model
from .feature_engineering import FeatureBuilder, build_training_table, feature_registry, tune_weighting
from .game_model import estimate_eb, run_game_model
from .simulation import ResidualPools, build_pools, calibrate_uncertainty
from .utils import log, stable_hash, utc_now_iso
from .validation import (MODEL_LABELS, assemble_oof, choose_reconciliation, composite_metrics, reconcile_minutes,
                         test_years_for)

TRAIN_VERSION = "1.0"


@dataclass
class Models:
    targets: dict
    newcomer: dict
    recon_mode: str
    recon_table: pd.DataFrame
    pools: ResidualPools
    performance: pd.DataFrame
    importance: pd.DataFrame
    feature_policy: pd.DataFrame
    oof: pd.DataFrame
    test_years: list
    train_years: list
    eb: dict
    game: dict | None
    fingerprint: str
    trained_at: str = field(default_factory=utc_now_iso)
    sim_mult: float = 1.0
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)


def _performance_table(results: dict, newc: dict, comp: pd.DataFrame, game: dict | None) -> pd.DataFrame:
    frames = []
    for grp in (results, newc):
        for r in grp.values():
            labels = {**MODEL_LABELS, "fixed": r.spec.alt_label}
            fm = r.fold_metrics.copy()
            summ = r.summary.reset_index().assign(target=r.spec.name, test_year="mean")
            both = pd.concat([fm, summ], ignore_index=True)
            both["label"] = both["model"].map(labels)
            both["selected"] = both["model"] == r.selected
            frames.append(both)
    c = comp.copy()
    c["label"] = c["model"]
    c["selected"] = c["model"] == "pipeline_selected"
    frames.append(c)
    if game and len(game.get("metrics", [])):
        g = game["metrics"].copy()
        gm = g.groupby(["target", "model"], as_index=False).mean(numeric_only=True).assign(test_year="mean")
        g = pd.concat([g, gm], ignore_index=True)
        g["label"] = g["model"]
        g["selected"] = False
        frames.append(g)
    perf = pd.concat(frames, ignore_index=True)
    cols = ["target", "model", "label", "test_year", "n", "rmse", "mae", "r2", "pearson", "spearman", "mape",
            "top_n_mae", "top_n_rmse", "top_n_overlap", "selected"]
    for c_ in cols:
        if c_ not in perf.columns:
            perf[c_] = pd.NA
    return perf[cols]


def train_all(pdx, cfg, builder: FeatureBuilder | None = None, force: bool = False, with_game_model: bool = True,
              tune: bool = False) -> Models:
    years = list(range(cfg.first_year + 1, cfg.last_completed_year + 1))
    test_years = test_years_for(years, cfg)
    if tune:                                         # tune BEFORE fingerprinting so the saved models carry the tuned values
        tune_weighting(builder or FeatureBuilder(pdx, cfg), cfg, years, test_years[0])
        builder = FeatureBuilder(pdx, cfg)           # re-read the tuned parameters
    tuned_file = cfg.processed_dir / "tuned_weighting.json"
    fp = stable_hash({"cfg": cfg.model_fingerprint(), "pre": pdx.meta.get("fingerprint"), "v": TRAIN_VERSION,
                      "game": with_game_model, "tuned": tuned_file.read_text() if tuned_file.exists() else None})
    path = cfg.models_dir / "bundle.joblib"
    if not force and not tune and path.exists():
        try:
            bundle = joblib.load(path)
            if bundle.fingerprint == fp:
                log.info("models are up to date (fingerprint %s, trained %s) - loaded from disk", fp, bundle.trained_at)
                return bundle
            log.info("data/config changed since the last training run - retraining")
        except Exception as exc:
            log.warning("could not load saved models (%s) - retraining", exc)
    builder = builder or FeatureBuilder(pdx, cfg)
    log.info("building the player-season training table for target seasons %s..%s", years[0], years[-1])
    tab = build_training_table(builder, years)
    log.info("chronological test seasons: %s  (each is predicted by models trained only on earlier seasons)", test_years)
    results = {"fppm": fppm_model.run(tab, cfg, test_years), "mpg": minutes_model.run(tab, cfg, test_years),
               "avail": availability_model.run(tab, cfg, test_years)}
    log.info("rookie / low-history pathway")
    newc = rookie_model.run(tab, cfg, test_years)
    oof = assemble_oof(results, tab)
    recon_mode, recon_tab = choose_reconciliation(oof, cfg)
    log.info("minutes reconciliation to the 240-minute team budget: mode=%s  (MAE %s)", recon_mode,
             recon_tab["mpg_mae"].round(3).to_dict())
    oof["p_mpg_rec"] = reconcile_minutes(oof["team"], oof["year"], oof["p_mpg"], oof["p_avail"], recon_mode, cfg)
    games_by_year = pdx.league.set_index("year")["games"].to_dict()
    comp = composite_metrics(oof, games_by_year, "p_mpg_rec", cfg)
    allrow = comp[(comp["test_year"] == "all")]
    for _, r in allrow.iterrows():
        log.info("   %-9s %-28s spearman=%.3f rmse=%.1f top%d_mae=%.1f", r["target"], r["model"], r["spearman"], r["rmse"],
                 int(cfg.get("validation.top_n", 100)), r["top_n_mae"])
    pools = build_pools(oof, newc, tab.meta, cfg)
    sim_mult, calib = calibrate_uncertainty(oof, games_by_year, cfg)
    game, eb = None, {"k_minutes": float(cfg.get("in_season.k_minutes_default")),
                      "k_games": float(cfg.get("in_season.k_games_default"))}
    if with_game_model and cfg.get("game_model.enabled", True):
        game = run_game_model(pdx, cfg, test_years)
        eb = estimate_eb(game, oof, cfg)
    importance = pd.concat([r.importance for r in list(results.values()) + list(newc.values()) if len(r.importance)],
                           ignore_index=True)
    bundle = Models(targets=results, newcomer=newc, recon_mode=recon_mode, recon_table=recon_tab, pools=pools,
                    performance=_performance_table(results, newc, comp, game), importance=importance,
                    feature_policy=feature_registry(tab.X.columns), oof=oof, test_years=test_years,
                    train_years=years, eb=eb, game=game, fingerprint=fp,
                    sim_mult=sim_mult, calibration=calib)
    for r in list(results.values()) + list(newc.values()):
        r.fold_models = {}          # only needed to compute permutation importance (already done)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    log.info("models saved -> %s", path)
    return bundle

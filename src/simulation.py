"""Monte Carlo projection built from REAL out-of-sample model errors (no arbitrary +/- percentages).

1.  Residuals e = actual - projected come from the chronological test folds (the model never saw them).
2.  FPPM and minutes residuals are heteroscedastic (thin histories / old players are less predictable), so a
    log-linear scale model s(x) = exp(b'x) is fitted to log|e| and residuals are standardised z = e / s(x).
3.  FPPM and minutes residuals are resampled *jointly* (same historical player-season), preserving their
    correlation.  Availability residuals are resampled from the bucket of players with a similar projected
    availability, so the skewed, zero-inflated shape (full-season absences) is kept.  Availability is treated as
    independent of per-minute performance (documented simplification).
4.  Every simulated season: FPPM x minutes x (availability x games) -> total fantasy points; percentiles,
    mean, downside tail (CVaR), probability of a role expansion, etc. are taken over >= 10,000 draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils import log

PCTS = (10, 25, 50, 75, 90)


# ====================================================================== scale models
def scale_features(prior_min, age, n_seasons) -> np.ndarray:
    prior_min = np.nan_to_num(np.asarray(prior_min, float), nan=0.0)
    age = np.nan_to_num(np.asarray(age, float), nan=27.0)
    n = np.nan_to_num(np.asarray(n_seasons, float), nan=0.0)
    return np.column_stack([np.ones(len(prior_min)), np.log(prior_min + 100.0), (age - 27.0) ** 2 / 50.0, 1.0 / (n + 1.0)])


def fit_scale(F: np.ndarray, resid: np.ndarray) -> np.ndarray | None:
    if len(resid) < 60:
        return None
    t = np.log(np.abs(resid) + 1e-3)
    b, *_ = np.linalg.lstsq(F, t, rcond=None)
    return b


def eval_scale(F: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    return np.ones(len(F)) if b is None else np.exp(F @ b)


# ====================================================================== pools
@dataclass
class Pool:
    zf: np.ndarray
    zm: np.ndarray
    avail_edges: np.ndarray
    avail_res: list
    b_f: np.ndarray | None
    b_m: np.ndarray | None
    n_pairs: int
    n_avail: int
    corr_fm: float = np.nan


@dataclass
class ResidualPools:
    main: Pool
    newcomer: Pool
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


def _avail_pool(p_avail: np.ndarray, resid: np.ndarray, n_bins: int = 3):
    if len(resid) < 3 * 40:
        return np.array([]), [resid.copy()]
    edges = np.quantile(p_avail, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, p_avail)
    return edges, [resid[b == i] for i in range(n_bins)]


def fit_main_pool(df: pd.DataFrame) -> Pool:
    d = df[df["pathway"] == "main"]
    F = scale_features(d["prior_min"], d["age"], d["n_prior_seasons"])
    ef = (d["y_fppm"] - d["p_fppm"]).to_numpy()
    em = (d["y_mpg"] - d["p_mpg_rec"]).to_numpy()
    mf, mm = d["elig_fppm"].to_numpy(bool), d["elig_mpg"].to_numpy(bool)
    b_f, b_m = fit_scale(F[mf], ef[mf]), fit_scale(F[mm], em[mm])
    both = mf & mm
    zf = ef[both] / eval_scale(F[both], b_f)
    zm = em[both] / eval_scale(F[both], b_m)
    ma = d["elig_avail"].to_numpy(bool)
    ea = (d["y_avail"] - d["p_avail"]).to_numpy()[ma]
    edges, res = _avail_pool(d["p_avail"].to_numpy()[ma], ea)
    corr = float(np.corrcoef(zf, zm)[0, 1]) if len(zf) > 10 else np.nan
    return Pool(zf, zm, edges, res, b_f, b_m, int(both.sum()), int(ma.sum()), corr)


def fit_newcomer_pool(newc: dict, tab_meta: pd.DataFrame, main_pool: Pool, cfg) -> Pool:
    """Pool from the newcomer model's own out-of-sample residuals (falls back to the main pool, inflated)."""
    sel = {k: r.selected for k, r in newc.items()}
    o = {k: r.oof[r.oof["elig"]][["year", "player_id", "y", f"pred_{sel[k]}"]].rename(columns={"y": f"y_{k}", f"pred_{sel[k]}": f"p_{k}"})
         for k, r in newc.items()}
    pair = o["fppm"].merge(o["mpg"], on=["year", "player_id"])
    ef, em = (pair["y_fppm"] - pair["p_fppm"]).to_numpy(), (pair["y_mpg"] - pair["p_mpg"]).to_numpy()
    ea = (o["avail"]["y_avail"] - o["avail"]["p_avail"]).to_numpy()
    min_pool = int(cfg.get("simulation.min_pool_size", 80))
    mult = float(cfg.get("simulation.newcomer_uncertainty_mult", 1.5))
    if len(pair) < min_pool:
        log.warning("only %d newcomer residual pairs - using the main-pathway pool inflated x%.1f", len(pair), mult)
        sf = np.nanstd(main_pool.zf) or 1.0
        return Pool(main_pool.zf * mult, main_pool.zm * mult, main_pool.avail_edges, main_pool.avail_res,
                    main_pool.b_f, main_pool.b_m, main_pool.n_pairs, main_pool.n_avail, main_pool.corr_fm)
    edges, res = _avail_pool(o["avail"]["p_avail"].to_numpy(), ea, n_bins=2)
    corr = float(np.corrcoef(ef, em)[0, 1]) if len(ef) > 10 else np.nan
    return Pool(ef, em, edges, res, None, None, len(pair), len(ea), corr)


def build_pools(oof: pd.DataFrame, newc: dict, tab_meta: pd.DataFrame, cfg) -> ResidualPools:
    main = fit_main_pool(oof)
    newcomer = fit_newcomer_pool(newc, tab_meta, main, cfg)
    diag = pd.DataFrame([
        {"pool": "main", "paired_fppm_minutes_residuals": main.n_pairs, "availability_residuals": main.n_avail,
         "corr_fppm_minutes_resid": main.corr_fm, "sd_z_fppm": float(np.std(main.zf)), "sd_z_minutes": float(np.std(main.zm))},
        {"pool": "newcomer", "paired_fppm_minutes_residuals": newcomer.n_pairs,
         "availability_residuals": newcomer.n_avail, "corr_fppm_minutes_resid": newcomer.corr_fm,
         "sd_z_fppm": float(np.std(newcomer.zf)), "sd_z_minutes": float(np.std(newcomer.zm))}])
    return ResidualPools(main, newcomer, diag)


# ====================================================================== Monte Carlo
def simulate(pools: ResidualPools, proj: pd.DataFrame, cfg, seed: int | None = None) -> pd.DataFrame:
    """proj columns: pathway, fppm, mpg, avail, games_rem, fp_to_date, prior_min, age, n_prior_seasons,
    prior_mpg, ref_total (optional), mult (uncertainty multiplier), avail_cap, p_floor_games (optional)."""
    sc = cfg.get("simulation")
    n_sims = int(sc["n_sims"])
    rng = np.random.default_rng(sc["seed"] if seed is None else seed)
    chunk = int(sc.get("chunk_players", 100))
    q = float(cfg.get("ranking.downside_quantile", 0.10))
    delta = float(cfg.get("breakout.minutes_expansion_delta", 3.0))
    max_mpg = float(cfg.get("minutes_reconciliation.max_mpg", 41.0))
    out = {}
    n = len(proj)
    cols = {f"{v}_p{p}": np.full(n, np.nan) for v in ("fppm", "minutes", "fppg", "total", "games") for p in PCTS}
    extra = {k: np.full(n, np.nan) for k in ("total_mean", "total_cvar", "p_role_expansion", "p_beat_ref", "avail_p10",
                                              "avail_p90", "fppg_mean")}
    ar = lambda c, default=0.0: proj[c].to_numpy(float) if c in proj else np.full(n, default)
    fppm0, mpg0, av0 = ar("fppm"), ar("mpg"), ar("avail")
    grem, fp0 = ar("games_rem", cfg.season_length), ar("fp_to_date")
    mult, cap, ref_total = ar("mult", 1.0), ar("avail_cap", 1.0), ar("ref_total", np.nan)
    prior_mpg = ar("prior_mpg", np.nan)
    path = proj["pathway"].to_numpy()
    F_all = scale_features(proj["prior_min"], proj["age"], proj["n_prior_seasons"])
    for name, pool in (("main", pools.main), ("newcomer", pools.newcomer)):
        idx = np.flatnonzero(path == name)
        if len(idx) == 0:
            continue
        sf, sm = eval_scale(F_all[idx], pool.b_f), eval_scale(F_all[idx], pool.b_m)
        bins = np.searchsorted(pool.avail_edges, av0[idx]) if len(pool.avail_edges) else np.zeros(len(idx), int)
        for s in range(0, len(idx), chunk):
            sl = np.arange(s, min(s + chunk, len(idx)))
            ii = idx[sl]
            k = len(sl)
            j = rng.integers(0, len(pool.zf), size=(k, n_sims))
            f = np.clip(fppm0[ii, None] + pool.zf[j] * (sf[sl] * mult[ii])[:, None], 0.15, None)
            m = np.clip(mpg0[ii, None] + pool.zm[j] * (sm[sl] * mult[ii])[:, None], 3.0, max_mpg)
            a = np.empty((k, n_sims))
            for b in np.unique(bins[sl]):
                rows = np.flatnonzero(bins[sl] == b)
                res = pool.avail_res[min(b, len(pool.avail_res) - 1)]
                if len(res) == 0:
                    res = np.concatenate(pool.avail_res) if sum(len(r) for r in pool.avail_res) else np.zeros(1)
                a[rows] = av0[ii[rows], None] + res[rng.integers(0, len(res), size=(len(rows), n_sims))]
            a = np.minimum(np.clip(a, 0.0, 1.0), cap[ii, None])
            games = a * grem[ii, None]
            fppg = f * m
            total = fp0[ii, None] + fppg * games
            for key, arr in (("fppm", f), ("minutes", m), ("fppg", fppg), ("total", total), ("games", games)):
                pc = np.percentile(arr, PCTS, axis=1)
                for r, p in enumerate(PCTS):
                    cols[f"{key}_p{p}"][ii] = pc[r]
            kt = max(1, int(q * n_sims))
            extra["total_mean"][ii] = total.mean(1)
            extra["fppg_mean"][ii] = fppg.mean(1)
            extra["total_cvar"][ii] = np.partition(total, kt - 1, axis=1)[:, :kt].mean(1)
            extra["p_role_expansion"][ii] = (m >= (prior_mpg[ii, None] + delta)).mean(1)
            extra["p_beat_ref"][ii] = (total > ref_total[ii, None]).mean(1)
            extra["avail_p10"][ii] = np.percentile(a, 10, axis=1)
            extra["avail_p90"][ii] = np.percentile(a, 90, axis=1)
    res = pd.DataFrame({**cols, **extra}, index=proj.index)
    return res


# ====================================================================== empirical Bayes (in-season updating)
def eb_prior_strength(resid_var: float, noise_var_per_unit: float, mean_units: float, default: float) -> float:
    """K (in minutes or games) such that posterior = (K*prior + n*obs) / (K + n).

    resid_var  : variance of out-of-sample season-level errors (true-talent drift + sampling noise)
    noise_var_per_unit : sampling variance of one unit (a minute / a game)  -> sampling var of a season = nv / units
    tau^2 = resid_var - noise_var / mean_units ;  K = noise_var / tau^2
    """
    tau2 = resid_var - noise_var_per_unit / max(mean_units, 1.0)
    if not np.isfinite(tau2) or tau2 <= 1e-9 or not np.isfinite(noise_var_per_unit) or noise_var_per_unit <= 0:
        return default
    return float(np.clip(noise_var_per_unit / tau2, 50.0 if default > 100 else 3.0, 6000.0 if default > 100 else 150.0))


# ====================================================================== calibration of the simulated intervals
def calibrate_uncertainty(oof: pd.DataFrame, games_by_year: dict, cfg, target: float = 0.80):
    """Leave-one-test-season-out check of the simulated season-total intervals.

    For each test season S: residual pools are fitted on the OTHER test seasons only, the main-pathway players of S are
    simulated, and we record how often the realised total fantasy points fall in the simulated p10-p90 / p25-p75 range.
    A single multiplier on the FPPM/minutes residual scale is then chosen so that p10-p90 covers `target` (80%)."""
    c2 = cfg.copy()
    c2.set("simulation.n_sims", 800)
    folds = []
    for T in sorted(oof["year"].unique()):
        tr = oof[(oof["year"] != T) & (oof["pathway"] == "main")]
        te = oof[(oof["year"] == T) & (oof["pathway"] == "main")]
        if len(tr) < 150 or len(te) < 30:
            continue
        pool = fit_main_pool(tr)
        proj = pd.DataFrame({"pathway": "main", "fppm": te["p_fppm"], "mpg": te["p_mpg_rec"], "avail": te["p_avail"],
                             "games_rem": float(games_by_year.get(T, cfg.season_length)), "fp_to_date": 0.0,
                             "prior_min": te["prior_min"], "age": te["age"], "n_prior_seasons": te["n_prior_seasons"],
                             "mult": 1.0, "avail_cap": 1.0, "prior_mpg": np.nan, "ref_total": np.nan}, index=te.index)
        folds.append((T, pool, proj, te["y_fp"].to_numpy()))
    if not folds:
        return 1.0, pd.DataFrame()

    def coverage(m: float):
        out = []
        for T, pool, proj, y in folds:
            sim = simulate(ResidualPools(pool, pool), proj.assign(mult=m), c2, seed=11)
            out.append((T, float(((y >= sim["total_p10"]) & (y <= sim["total_p90"])).mean()),
                        float(((y >= sim["total_p25"]) & (y <= sim["total_p75"])).mean()), len(y)))
        return out

    def pooled(res):
        n = np.array([r[3] for r in res], float)
        return float(np.average([r[1] for r in res], weights=n)), float(np.average([r[2] for r in res], weights=n))

    before = coverage(1.0)
    c80, _ = pooled(before)
    m = 1.0
    if abs(c80 - target) > 0.02:
        lo, hi = 0.6, 3.0
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            if pooled(coverage(mid))[0] < target:
                lo = mid
            else:
                hi = mid
        m = round(0.5 * (lo + hi), 3)
    after = coverage(m)
    tab = pd.DataFrame([{"test_year": a[0], "n": a[3], "coverage_p10_p90_before": a[1], "coverage_p25_p75_before": a[2],
                         "coverage_p10_p90_after": b[1], "coverage_p25_p75_after": b[2], "multiplier": m}
                        for a, b in zip(before, after)])
    pb, pa = pooled(before), pooled(after)
    log.info("simulation calibration (leave-one-season-out): 80%% interval covers %.1f%% / 50%% interval covers %.1f%% "
             "-> multiplier %.2f -> %.1f%% / %.1f%%", 100 * pb[0], 100 * pb[1], m, 100 * pa[0], 100 * pa[1])
    return m, tab

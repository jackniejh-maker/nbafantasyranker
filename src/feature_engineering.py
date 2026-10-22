"""Leak-free player-season feature builder.

`FeatureBuilder.build(T, roster)` returns features for every player on the season-T roster using ONLY

  (a) player/team/coach statistics from seasons strictly BEFORE T  (hard-filtered at the top of build()), and
  (b) information that is genuinely known before the season starts: the roster (team), position,
      birth date / draft slot, head coach, and the injury-free baseline of the teammates.

No season-T box score, team rating, injury, minutes or coaching result ever enters X.
`FEATURE_REGISTRY` (exported to outputs/feature_policy.csv) lists every feature with its group and the exact
information it may use; tests/test_no_leakage.py perturbs season-T data and asserts X is unchanged.

Feature groups (prefix): hist, late, trade, age, role, pos, team, mate, coach, newc.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from .coach_features import coach_deviations, coach_features, team_style
from .preprocessing import ProcessedData
from .team_features import team_context
from .teammate_features import teammate_features
from .utils import log, position_group, season_ref_date

LAG_COLS = ["min", "gp", "mpg", "fppm", "fppg", "gs_rate", "avail", "usg", "ts", "ast_pct", "reb_pct", "stl_pct",
            "blk_pct", "tov_pct", "ftr", "tpar", "fga_pm", "ast_pm", "reb_pm", "tov_pm", "late_mpg", "late_fppm",
            "late_n", "last_stint_fppm", "last_stint_min", "fp_per_fga", "fp", "last_team"]
MPG_EDGES = [12, 18, 24, 30, 36]
ROLE_EDGES = [20, 25, 30, 35]
GROUPS = ("hist", "late", "trade", "age", "role", "pos", "team", "mate", "coach", "newc")


def pick_bucket(pick) -> np.ndarray:
    """Draft-slot buckets: 0: 1-5, 1: 6-14, 2: 15-30, 3: 31-60, 4: undrafted/unknown."""
    p = np.asarray(pd.to_numeric(pd.Series(pick), errors="coerce"), dtype=float)
    return np.select([p <= 5, p <= 14, p <= 30, p <= 60], [0, 1, 2, 3], 4).astype(int)


@dataclass
class FeatureFrame:
    X: pd.DataFrame
    meta: pd.DataFrame
    year: int


class FeatureBuilder:
    def __init__(self, pdx: ProcessedData, cfg):
        self.pdx, self.cfg = pdx, cfg
        self.ps = pdx.player_seasons
        self.L = cfg.history_window
        w0 = np.array(cfg.get("weighting.recency_weights")[: self.L], float)
        self.a = w0 / w0.sum() * self.L
        self.wc = dict(cfg.get("weighting"))
        tuned = cfg.processed_dir / "tuned_weighting.json"
        if self.wc.get("use_tuned", True) and tuned.exists():
            try:
                self.wc.update(json.loads(tuned.read_text())["params"])
                log.info("using tuned weighting parameters: %s", json.loads(tuned.read_text())["params"])
            except Exception as exc:
                log.warning("could not read %s (%s)", tuned, exc)
        info = pdx.player_info.drop_duplicates("player_id").set_index("player_id")
        for c in ("player_name", "position", "birth_date", "height_in", "weight", "draft_pick", "country",
                  "debut_year"):
            if c not in info.columns:
                info[c] = np.nan
        self.info = info
        self.style = team_style(pdx.stints, self.ps, pdx.team_seasons, pdx.player_games)
        self.coach_dev = coach_deviations(self.style, pdx.head_coaches)
        hc = pdx.head_coaches
        self.coach_by_year = {y: dict(zip(g["team"], g["coach_id"])) for y, g in hc.groupby("year")}

    # ------------------------------------------------------------------ rosters
    def roster_for(self, T: int) -> pd.DataFrame:
        """Preseason roster: current roster for the live season, opening-roster proxy for history."""
        if T >= self.cfg.target_year:
            cur = self.pdx.current_roster
            if cur is None or len(cur) == 0:
                raise RuntimeError("no current roster available for the target season")
            keep = [c for c in ("player_id", "team", "player_name", "position", "birth_date", "height_in", "weight",
                                "exp") if c in cur.columns]
            base = cur[keep].copy()
        else:
            op = self.pdx.opening_rosters
            base = op[op["year"] == T][["player_id", "team"]].copy()
            r = self.pdx.rosters
            if r is not None and len(r):
                rr = r[r["year"] == T].drop_duplicates("player_id")[["player_id", "position", "birth_date", "player_name"]]
                base = base.merge(rr, on="player_id", how="left")
        return self._decorate(base, T)

    def _decorate(self, base: pd.DataFrame, T: int) -> pd.DataFrame:
        r = base.copy()
        info, ps = self.info, self.ps
        hist = ps[ps["year"] < T].sort_values("year").drop_duplicates("player_id", keep="last").set_index("player_id")

        def col(name):
            return r[name] if name in r.columns else pd.Series(np.nan, index=r.index, dtype=object)

        pos = col("position").astype(object)
        pos = pos.where(pos.notna() & ~pos.astype(str).isin(["", "nan", "None"]))
        r["position"] = pos.fillna(r["player_id"].map(hist["position"])).fillna(r["player_id"].map(info["position"])).fillna("F")
        r["pos_group"] = r["position"].map(position_group).replace("", "F")
        r["player_name"] = col("player_name").fillna(r["player_id"].map(info["player_name"]))
        bd = pd.to_datetime(col("birth_date")).fillna(pd.to_datetime(r["player_id"].map(info["birth_date"])))
        r["age"] = (pd.Timestamp(season_ref_date(T)) - bd).dt.days / 365.25
        fb = r["player_id"].map(hist["age"]) + (T - r["player_id"].map(hist["year"]))
        r["age"] = r["age"].fillna(fb)
        debut = pd.to_numeric(r["player_id"].map(info["debut_year"]), errors="coerce")
        if "exp" in r.columns:
            debut = debut.fillna(T - pd.to_numeric(r["exp"], errors="coerce"))
        r["debut_year"] = debut.fillna(T)
        r["exp"] = (T - r["debut_year"]).clip(lower=0)
        for c in ("draft_pick", "height_in", "weight", "country"):
            s = col(c)
            r[c] = s.fillna(r["player_id"].map(info[c]))
        r["draft_pick"] = pd.to_numeric(r["draft_pick"], errors="coerce")
        return r.drop_duplicates("player_id").set_index("player_id")

    # ------------------------------------------------------------------ helpers
    def _wide(self, T: int, ids) -> pd.DataFrame:
        parts = []
        for lag in range(1, self.L + 1):
            s = self.ps[self.ps["year"] == T - lag].drop_duplicates("player_id").set_index("player_id")
            cols = [c for c in LAG_COLS if c in s.columns]
            d = s[cols].reindex(ids)
            d.columns = [f"{c}_l{lag}" for c in cols]
            parts.append(d)
        return pd.concat(parts, axis=1)

    def _arr(self, wide: pd.DataFrame, name: str) -> np.ndarray:
        return np.column_stack([pd.to_numeric(wide[f"{name}_l{l}"], errors="coerce").to_numpy(float)
                                for l in range(1, self.L + 1)])

    def priors(self, T: int) -> dict:
        """League-level priors computed ONLY from seasons < T."""
        wc, ps, info = self.wc, self.ps, self.info
        hist = ps[ps["year"] < T]
        lb = hist[hist["year"] >= T - int(wc["prior_lookback_seasons"])]
        q = lb[lb["min"] >= 200]
        if len(q) < 30:
            q = hist[hist["min"] >= 200]
        mu_all = float(q["fp"].sum() / q["min"].sum()) if len(q) else 0.9
        bins = np.digitize(q["mpg"], MPG_EDGES)
        mu_bin = (q["fp"].groupby(bins).sum() / q["min"].groupby(bins).sum()).reindex(range(len(MPG_EDGES) + 1))
        mu_bin = mu_bin.fillna(mu_all).to_numpy()
        qa = q[q["min"] >= 400]
        mu_avail = float(qa["avail"].mean()) if len(qa) else 0.8
        pos_rates = q.groupby("pos_group")[["usg", "ast_pct", "reb_pct", "fga_pm", "tpar", "fppm"]].mean()
        overall = q[["usg", "ast_pct", "reb_pct", "fga_pm", "tpar", "fppm"]].mean()
        for g in ("G", "F", "C"):
            if g not in pos_rates.index:
                pos_rates.loc[g] = overall
        pos_rates = pos_rates.fillna(overall)
        h = hist.copy()
        h["debut"] = h["player_id"].map(info["debut_year"])
        rk = h[(h["year"] == h["debut"]) & (h["gp"] >= 10)].copy()
        rk["bucket"] = pick_bucket(rk["player_id"].map(info["draft_pick"]))
        gb = rk.groupby("bucket")
        rp = pd.DataFrame({"mpg": gb["min"].sum() / gb["gp"].sum(), "fppm": gb["fp"].sum() / gb["min"].sum(),
                           "avail": gb["avail"].mean(), "n": gb.size()}).reindex(range(5))
        dflt = {"mpg": 12.0, "fppm": 0.92 * mu_all, "avail": 0.70}
        for c, v in dflt.items():
            tot = (rk[{"mpg": "min", "fppm": "fp", "avail": "avail"}[c]].sum() / (rk["gp"].sum() if c == "mpg" else
                   rk["min"].sum() if c == "fppm" else max(len(rk), 1))) if len(rk) else v
            rp[c] = rp[c].fillna(tot if np.isfinite(tot) else v)
        fr = lb[(lb["min"] >= 100) & (lb["min"] < 600)]
        return {"mu_all": mu_all, "mu_bin": mu_bin, "mu_avail": mu_avail, "pos_rates": pos_rates,
                "rookie_prior": rp, "fringe_mpg": float(fr["mpg"].mean()) if len(fr) > 20 else 10.0}

    # ------------------------------------------------------------------ main entry point
    def build(self, T: int, roster: pd.DataFrame | None = None, minutes_override: dict | None = None,
              weighting: dict | None = None, capture: dict | None = None) -> FeatureFrame:
        pdx, wc = self.pdx, {**self.wc, **(weighting or {})}
        assert (self.ps[self.ps["year"] < T]["year"].max() < T) if (self.ps["year"] < T).any() else True
        roster = self.roster_for(T) if roster is None else roster
        ids = roster.index.to_numpy()
        n, L, a = len(ids), self.L, self.a
        pri = self.priors(T)
        wide = self._wide(T, ids)
        M, G, MPG = (np.nan_to_num(self._arr(wide, k)) for k in ("min", "gp", "mpg"))
        FPPM, AV = self._arr(wide, "fppm"), self._arr(wide, "avail")
        pos_g = roster["pos_group"].to_numpy()
        pr = pri["pos_rates"]

        # ---- stage 1: baseline minutes / availability (no fitted parameters) -----------------------
        Wg = a[None, :] * G
        mpg_w0 = np.where(Wg.sum(1) > 0, (Wg * MPG).sum(1) / np.maximum(Wg.sum(1), 1e-9), np.nan)
        g_eff = G.sum(1)
        bucket = pick_bucket(roster["draft_pick"])
        rp = pri["rookie_prior"]
        prior_mpg = np.where(roster["exp"].to_numpy() <= 3, rp["mpg"].to_numpy()[bucket], pri["fringe_mpg"])
        cred_g = g_eff / (g_eff + float(wc["baseline_games_k"]))
        mpg_ref0 = np.where(np.isnan(mpg_w0), prior_mpg, cred_g * np.nan_to_num(mpg_w0) + (1 - cred_g) * prior_mpg)
        if minutes_override:
            for pid, v in minutes_override.items():
                if pid in roster.index:
                    mpg_ref0[roster.index.get_loc(pid)] = float(v)
        av_has = ~np.isnan(AV)
        av_num = (a[None, :] * np.nan_to_num(AV) * av_has).sum(1)
        av_den = (a[None, :] * av_has).sum(1)
        ka = float(wc["avail_prior_weight"])
        avail_shr = (av_num + ka * pri["mu_avail"]) / (av_den + ka)
        avail_ref = np.where(np.isnan(avail_shr), pri["mu_avail"], avail_shr)

        def wrate(name: str, prior: np.ndarray, k: float = 600.0) -> np.ndarray:
            R = self._arr(wide, name)
            W = a[None, :] * M * ~np.isnan(R)
            den = W.sum(1)
            val = np.where(den > 0, (W * np.nan_to_num(R)).sum(1) / np.maximum(den, 1e-9), np.nan)
            c = den / (den + k)
            return np.where(np.isnan(val), prior, c * np.nan_to_num(val) + (1 - c) * prior)

        pri_of = lambda col: pr[col].reindex(pos_g).to_numpy(float)
        usg_w = wrate("usg", pri_of("usg"))
        ast_w = wrate("ast_pct", pri_of("ast_pct"))
        reb_w = wrate("reb_pct", pri_of("reb_pct"))
        fga_w = wrate("fga_pm", pri_of("fga_pm"))
        tpar_w = wrate("tpar", pri_of("tpar"))
        mu0 = pri["mu_bin"][np.digitize(mpg_ref0, MPG_EDGES)]
        fppm_w = wrate("fppm", mu0)
        base = pd.DataFrame({"team": roster["team"].to_numpy(), "pos_group": pos_g, "mpg_ref0": mpg_ref0,
                             "avail_ref": avail_ref, "usg_w": usg_w, "ast_w": ast_w, "reb_w": reb_w,
                             "fga_pm_w": fga_w, "tpar_w": tpar_w, "fppm_w": fppm_w}, index=ids)

        # ---- context: team, teammates, coach ----------------------------------------------------------
        tctx = team_context(T, roster.reset_index(), base, pdx.stints, self.ps, pdx.team_seasons, pdx.league)
        team_f = tctx.reindex(roster["team"].to_numpy())
        team_f.index = ids
        mate_f = teammate_features(base)
        coach_f = self._coach_features(T, roster).reindex(roster["team"].to_numpy())
        coach_f.index = ids

        # ---- budget-scaled minutes (injury-free redistribution of 240 minutes) --------------------------
        exp_min = base["mpg_ref0"] * base["avail_ref"]
        tsum = exp_min.groupby(base["team"]).transform("sum")
        scale = np.clip(240.0 / tsum, 0.8, 1.25)
        mpg_budget = np.minimum(base["mpg_ref0"] * scale, float(self.cfg.get("minutes_reconciliation.max_mpg", 41.0)))

        # ---- stage 2: role/team/age-adjusted recency weights ------------------------------------------
        fppm_shr, fppm_raw, fppm_fixed, eff_min = self._weighted_fppm(
            wide, roster, mpg_budget.to_numpy(), pri, wc, M, MPG, FPPM)
        if capture is not None:      # stage-1 intermediates, so weighting parameters can be tuned cheaply
            capture.update(dict(wide=wide, roster=roster, mpg_budget=mpg_budget.to_numpy(), pri=pri, M=M, MPG=MPG, FPPM=FPPM))
        X: dict = {}                       # columns are collected in a dict and turned into one frame at the end
        X["hist__fppm_shrunk"] = fppm_shr
        X["hist__fppm_fixed"] = fppm_fixed
        X["hist__fppm_raw_w"] = fppm_raw
        X["hist__mpg_w"] = mpg_ref0
        X["hist__avail_shrunk"] = avail_ref
        X["hist__usg_w"], X["hist__ast_pct_w"], X["hist__reb_pct_w"] = usg_w, ast_w, reb_w
        X["hist__eff_minutes_log"] = np.log1p(eff_min)
        has_l = [(M[:, l] > 0) for l in range(L)]
        X["hist__n_seasons"] = np.sum(has_l, axis=0)
        for l in range(1, L + 1):
            f_l, m_l, v_l = FPPM[:, l - 1], MPG[:, l - 1], AV[:, l - 1]
            X[f"hist__fppm_l{l}"] = np.where(np.isnan(f_l), fppm_shr, f_l)
            X[f"hist__mpg_l{l}"] = np.where(M[:, l - 1] > 0, m_l, mpg_ref0)
            X[f"hist__avail_l{l}"] = np.where(np.isnan(v_l), avail_ref, v_l)
            if l <= 2:
                X[f"hist__log_min_l{l}"] = np.log1p(M[:, l - 1])
            if l >= 2:
                X[f"hist__has_l{l}"] = has_l[l - 1].astype(float)
        X["hist__gs_rate_l1"] = self._arr(wide, "gs_rate")[:, 0]
        X["hist__gs_rate_l2"] = self._arr(wide, "gs_rate")[:, 1] if L > 1 else np.nan
        for nm in ("usg", "ts", "ast_pct", "reb_pct", "stl_pct", "blk_pct", "tov_pct", "ftr", "tpar", "fp_per_fga"):
            X[f"hist__{nm}_l1"] = self._arr(wide, nm)[:, 0]
        X["hist__missing_l1"] = (M[:, 0] <= 0).astype(float)
        d = lambda nm: self._arr(wide, nm)
        X["hist__d_mpg"] = np.nan_to_num(d("mpg")[:, 0] - d("mpg")[:, 1]) if L > 1 else 0.0
        X["hist__d_usg"] = np.nan_to_num(d("usg")[:, 0] - d("usg")[:, 1]) if L > 1 else 0.0
        X["hist__d_ts"] = np.nan_to_num(d("ts")[:, 0] - d("ts")[:, 1]) if L > 1 else 0.0
        X["hist__d_fppm"] = np.nan_to_num(d("fppm")[:, 0] - d("fppm")[:, 1]) if L > 1 else 0.0
        # late-season form (last 20 games of season T-1)
        lm, lf, ln = d("late_mpg")[:, 0], d("late_fppm")[:, 0], d("late_n")[:, 0]
        ok = ln >= 10
        X["late__mpg_l1"] = np.where(ok, lm, X["hist__mpg_l1"])
        X["late__fppm_l1"] = np.where(ok, lf, X["hist__fppm_l1"])
        X["late__mpg_gap"] = X["late__mpg_l1"] - X["hist__mpg_l1"]
        X["late__fppm_gap"] = X["late__fppm_l1"] - X["hist__fppm_l1"]
        # traded players: performance after the trade vs. for the whole season
        lsm, tm1 = d("last_stint_min")[:, 0], M[:, 0]
        traded = (tm1 > 0) & (lsm < tm1 - 1)
        X["trade__traded_l1"] = traded.astype(float)
        X["trade__last_stint_share"] = np.where(traded, lsm / np.maximum(tm1, 1), 1.0)
        gap = d("last_stint_fppm")[:, 0] - FPPM[:, 0]
        X["trade__last_stint_gap"] = np.where(traded & ~np.isnan(gap), gap * np.minimum(lsm / 600.0, 1.0), 0.0)
        # age / development (piecewise-linear spline + quadratic; shape is learned, not assumed)
        age = roster["age"].fillna(roster["age"].median()).to_numpy(float)
        X["age__age"] = age
        X["age__c2"] = (age - 27.0) ** 2
        X["age__hinge_young"] = np.maximum(0, 25.0 - age)
        X["age__hinge_old"] = np.maximum(0, age - 30.0)
        X["age__exp"] = roster["exp"].to_numpy(float)
        # role
        X["role__mpg_ref"] = mpg_ref0
        X["role__mpg_budget"] = mpg_budget.to_numpy()
        X["role__budget_ratio"] = mpg_budget.to_numpy() / np.maximum(mpg_ref0, 1.0)
        X["role__gap_hist"] = mpg_budget.to_numpy() - np.where(has_l[0], MPG[:, 0], mpg_ref0)
        for lo, hi, nm in ((20, 25, "20_25"), (25, 30, "25_30"), (30, 35, "30_35"), (35, 99, "35p")):
            X[f"role__bin_{nm}"] = ((mpg_budget.to_numpy() >= lo) & (mpg_budget.to_numpy() < hi)).astype(float)
        X["role__usg_low"] = (usg_w < 17).astype(float)
        X["role__usg_high"] = (usg_w > 24).astype(float)
        X["role__is_top5"] = (mate_f["role__rank_in_team"].to_numpy() <= 5).astype(float)
        for p, nm in (("G", "g"), ("F", "f"), ("C", "c")):
            X[f"pos__{nm}"] = roster["position"].astype(str).str.contains(p, regex=False).astype(float).to_numpy()
        for c in team_f.columns:
            X[c] = team_f[c].to_numpy()
        for c in mate_f.columns:
            X[c] = mate_f[c].to_numpy()
        for c in coach_f.columns:
            X[c] = coach_f[c].to_numpy()
        # newcomer block (used by the rookie / low-history model)
        pick = roster["draft_pick"].to_numpy(float)
        cred_m = (M * a[None, :]).sum(1) / ((M * a[None, :]).sum(1) + float(wc["shrink_minutes"]))
        X["newc__log_pick"] = np.log(np.where(np.isnan(pick), 70.0, pick))
        X["newc__undrafted"] = np.isnan(pick).astype(float)
        X["newc__lottery"] = (pick <= 14).astype(float)
        X["newc__height_in"] = roster["height_in"].astype(float).to_numpy()
        X["newc__intl"] = (~roster["country"].fillna("USA").astype(str).isin(["USA", "United States", "nan"])).astype(float).to_numpy()
        X["newc__cred"] = cred_m
        X["newc__obs_dev"] = np.where(np.isnan(fppm_raw), 0.0, (fppm_raw - mu0) * cred_m)
        X["newc__is_rookie"] = (roster["exp"].to_numpy() <= 0).astype(float)
        X = pd.DataFrame(X, index=ids).astype(float)

        # ---- descriptive / baseline columns (NOT model inputs) ----------------------------------------
        prior_min = M.sum(1)
        ref_lag = np.argmax(M > 0, axis=1)
        any_hist = (M > 0).any(1)
        pick_row = lambda arr: np.where(any_hist, arr[np.arange(n), ref_lag], np.nan)
        meta = pd.DataFrame({
            "player_name": roster["player_name"].to_numpy(), "team": roster["team"].to_numpy(),
            "position": roster["position"].to_numpy(), "pos_group": pos_g, "age": roster["age"].to_numpy(),
            "exp": roster["exp"].to_numpy(), "debut_year": roster["debut_year"].to_numpy(),
            "draft_pick": roster["draft_pick"].to_numpy(), "prior_min": prior_min, "prior_gp": G.sum(1),
            "n_prior_seasons": np.asarray(X["hist__n_seasons"]),
            "is_rookie": (roster["exp"].to_numpy() <= 0) & (prior_min <= 0),
            "pathway": np.where(prior_min >= float(self.cfg.get("eligibility.min_history_minutes", 400)), "main", "newcomer"),
            "ref_year": np.where(any_hist, T - 1 - ref_lag, np.nan),
            "ref_mpg": pick_row(MPG), "ref_fppm": pick_row(np.where(np.isnan(FPPM), 0, FPPM)),
            "ref_fppg": pick_row(np.nan_to_num(self._arr(wide, "fppg"))), "ref_gp": pick_row(G),
            "ref_min": pick_row(M), "ref_fp": pick_row(np.nan_to_num(self._arr(wide, "fp"))),
            "base_fppm": fppm_shr, "base_fppm_fixed": fppm_fixed, "base_mpg": mpg_ref0, "base_avail": avail_ref,
            "mpg_budget": mpg_budget.to_numpy(), "usg_w": usg_w, "ast_w": ast_w, "reb_w": reb_w,
        }, index=ids)
        meta.loc[~any_hist, ["ref_mpg", "ref_fppm", "ref_fppg", "ref_gp", "ref_min", "ref_fp"]] = np.nan
        return FeatureFrame(X=X, meta=meta, year=T)

    # ------------------------------------------------------------------ pieces
    def _coach_features(self, T: int, roster: pd.DataFrame) -> pd.DataFrame:
        if T >= self.cfg.target_year:
            cc = self.pdx.current_coaches
            cur = dict(zip(cc["team"], cc["coach_id"])) if cc is not None and len(cc) else {}
        else:
            cur = self.coach_by_year.get(T, {})
        prev = self.coach_by_year.get(T - 1, {})
        teams = sorted(roster["team"].dropna().unique())
        return coach_features(T, self.coach_dev, {t: cur.get(t) for t in teams}, prev)

    def _weighted_fppm(self, wide, roster, mpg_ref, pri, wc, M, MPG, FPPM):
        """Recency weights x minutes, adjusted for role change, team change and age; then shrunk to a role prior."""
        L, a = self.L, self.a
        n = len(roster)
        tau = float(wc["role_tau_minutes"])
        lt = np.column_stack([wide[f"last_team_l{l}"].astype(object).to_numpy() for l in range(1, L + 1)])
        same = lt == roster["team"].to_numpy()[:, None]
        team_f = np.where(same, 1.0, float(wc["team_change_discount"]))
        mpg_f = np.where(M > 0, MPG, mpg_ref[:, None])
        role_f = np.exp(-np.abs(mpg_f - mpg_ref[:, None]) / max(tau, 1e-6))
        age = roster["age"].fillna(27.0).to_numpy(float)
        old = np.exp(-float(wc["old_age_decay"]) * np.maximum(0, age - 31.0))[:, None]
        age_f = np.ones((n, L))
        age_f[:, 1:] = np.repeat(old, L - 1, axis=1)
        A = a[None, :] * role_f * team_f * age_f
        K = float(wc["shrink_minutes"])
        mu = pri["mu_bin"][np.digitize(mpg_ref, MPG_EDGES)]
        F0 = np.nan_to_num(FPPM)
        W, W0 = A * M, a[None, :] * M
        num, den = (W * F0).sum(1), W.sum(1)
        num0, den0 = (W0 * F0).sum(1), W0.sum(1)
        shr = (num + K * mu) / (den + K)
        fixed = (num0 + K * mu) / (den0 + K)
        raw = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        return shr, raw, fixed, den

    @staticmethod
    def feature_group(name: str) -> str:
        return name.split("__")[0]


# ====================================================================== training table
@dataclass
class TrainingTable:
    X: pd.DataFrame
    meta: pd.DataFrame      # includes year, player_id, pathway, baselines and ref_* columns
    y: pd.DataFrame         # realised targets of season T (NEVER fed to features)


def build_training_table(builder: FeatureBuilder, years: list[int]) -> TrainingTable:
    cfg, ps = builder.cfg, builder.ps
    Xs, Ms, Ys = [], [], []
    elig = cfg.get("eligibility")
    for T in years:
        ff = builder.build(T)
        t = ps[ps["year"] == T].drop_duplicates("player_id").set_index("player_id")
        y = pd.DataFrame(index=ff.X.index)
        y["y_fppm"] = t["fppm"].reindex(y.index)
        y["y_mpg"] = t["mpg"].reindex(y.index)
        y["y_avail"] = t["avail"].reindex(y.index).fillna(0.0)
        y["y_gp"] = t["gp"].reindex(y.index).fillna(0.0)
        y["y_min"] = t["min"].reindex(y.index).fillna(0.0)
        y["y_fp"] = t["fp"].reindex(y.index).fillna(0.0)
        y["y_fppg"] = t["fppg"].reindex(y.index)
        m = ff.meta.copy()
        m["year"] = T
        m["player_id"] = m.index
        main = (m["pathway"] == "main").to_numpy()
        new = ~main
        y["elig_fppm"] = main & (y["y_min"] >= elig["min_target_minutes"]).to_numpy()
        y["elig_mpg"] = main & (y["y_gp"] >= elig["min_target_games"]).to_numpy()
        y["elig_avail"] = main
        y["elig_newc_fppm"] = new & (y["y_min"] >= elig["newcomer_min_target_minutes"]).to_numpy()
        y["elig_newc_mpg"] = new & (y["y_gp"] >= elig["min_target_games"]).to_numpy()
        y["elig_newc_avail"] = new
        Xs.append(ff.X)
        Ms.append(m)
        Ys.append(y)
        log.info("features %s: %d roster rows (%d main, %d newcomer)", T, len(ff.X), main.sum(), new.sum())
    X = pd.concat(Xs, ignore_index=True)
    meta = pd.concat(Ms, ignore_index=True)
    y = pd.concat(Ys, ignore_index=True)
    return TrainingTable(X=X, meta=meta, y=y)


# ====================================================================== feature policy (documentation)
_DESCR = {
    "hist": "player's own track record from seasons < T (lags l1..l3, recency/credibility weights, rates)",
    "late": "form over the last 20 games of season T-1",
    "trade": "last-season trade split (stint with final team vs whole season) - season T-1 only",
    "age": "age on Feb 1 of season T from birth date (+ spline/quadratic terms), years in league",
    "role": "baseline minutes from seasons < T, redistributed over the preseason roster (240-minute budget); starter proxies",
    "pos": "listed position (known before the season)",
    "team": "team ratings / pace / possessions from seasons < T; roster turnover vs last season; projected roster structure",
    "mate": "preseason teammates' prior-season usage/assist/rebound rates and baseline minutes; own-x-teammate products",
    "coach": "head coach's tendencies from seasons < T (shrunk to league mean); coach new to team flag",
    "newc": "draft slot, height, nationality, credibility of the limited NBA history (rookie / low-history model)",
}


def feature_registry(columns) -> pd.DataFrame:
    rows = []
    for c in columns:
        g = c.split("__")[0]
        rows.append({"feature": c, "group": g, "allowed_information": _DESCR.get(g, ""),
                     "uses_target_season_stats": False,
                     "uses_preseason_roster_info": g in ("team", "mate", "coach", "role", "age", "pos", "newc")})
    return pd.DataFrame(rows)


# ====================================================================== weighting-parameter tuning
TUNE_GRID = {"shrink_minutes": [750.0, 1500.0, 3000.0], "role_tau_minutes": [6.0, 10.0, 20.0, 1e6],
             "team_change_discount": [0.85, 0.92, 1.0], "old_age_decay": [0.0, 0.05]}


def tune_weighting(builder: FeatureBuilder, cfg, years: list[int], first_test_year: int) -> dict:
    """Grid-search the recency-weight adjustments on seasons strictly BEFORE the first test season.

    Objective: minutes-weighted RMSE of the weighted-average FPPM benchmark (Model A) on the realised season FPPM.
    Test seasons are never touched, so the later chronological validation stays honest."""
    import itertools
    ps = builder.ps
    cache = []
    for T in [y for y in years if y < first_test_year]:
        cap: dict = {}
        ff = builder.build(T, capture=cap)
        t = ps[ps["year"] == T].drop_duplicates("player_id").set_index("player_id")
        y = t["fppm"].reindex(ff.X.index)
        ym = t["min"].reindex(ff.X.index)
        ok = (ff.meta["pathway"] == "main").to_numpy() & (ym.fillna(0) >= cfg.get("eligibility.min_target_minutes", 300)).to_numpy()
        cache.append((cap, y.to_numpy()[ok], ym.to_numpy()[ok], ok))
    if not cache:
        return {}
    keys = list(TUNE_GRID)
    rows = []
    for combo in itertools.product(*[TUNE_GRID[k] for k in keys]):
        wc = {**builder.wc, **dict(zip(keys, combo))}
        se, w = 0.0, 0.0
        for cap, y, ym, ok in cache:
            shr, *_ = builder._weighted_fppm(cap["wide"], cap["roster"], cap["mpg_budget"], cap["pri"], wc, cap["M"], cap["MPG"], cap["FPPM"])
            w_ = ym / (ym + 600.0)
            se += float((w_ * (shr[ok] - y) ** 2).sum())
            w += float(w_.sum())
        rows.append({**dict(zip(keys, combo)), "rmse": float(np.sqrt(se / w))})
    res = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    dflt = {k: builder.wc[k] for k in keys}
    cur = res[(res[keys] == pd.Series(dflt)).all(axis=1)]
    best = res.iloc[0]
    out = {"params": {k: float(best[k]) for k in keys}, "rmse_best": float(best["rmse"]),
           "rmse_config_defaults": float(cur["rmse"].iloc[0]) if len(cur) else None,
           "tuning_years": [int(y) for y in years if y < first_test_year], "n_combinations": int(len(res))}
    path = cfg.processed_dir / "tuned_weighting.json"
    path.write_text(json.dumps(out, indent=1))
    log.info("weighting tuned on seasons %s: %s  (rmse %.4f vs %s with config defaults)", out["tuning_years"],
             out["params"], out["rmse_best"], out["rmse_config_defaults"])
    return out

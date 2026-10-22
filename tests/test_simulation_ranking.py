import unittest

import numpy as np
import pandas as pd

from tests._common import ROOT  # noqa: F401
from src.config import Config
from src.consensus import compare, match_names
from src.ranking import SPEC_COLUMNS, rank_and_score
from src.simulation import Pool, ResidualPools, eb_prior_strength, simulate


def pool(sd_f=0.0, sd_m=0.0, sd_a=0.0, n=400, seed=0):
    r = np.random.default_rng(seed)
    return Pool(zf=r.normal(0, sd_f, n), zm=r.normal(0, sd_m, n), avail_edges=np.array([]),
                avail_res=[r.normal(0, sd_a, n)], b_f=None, b_m=None, n_pairs=n, n_avail=n)


def proj(**kw):
    d = pd.DataFrame({"pathway": "main", "fppm": [1.0, 1.4, 0.8], "mpg": [30.0, 34.0, 20.0], "avail": [0.9, 0.8, 1.0],
                      "games_rem": 82.0, "fp_to_date": 0.0, "prior_min": 2000.0, "age": 27.0, "n_prior_seasons": 3,
                      "prior_mpg": 28.0, "ref_total": 2000.0, "mult": 1.0, "avail_cap": 1.0}, index=[10, 11, 12])
    for k, v in kw.items():
        d[k] = v
    return d


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.load()
        self.cfg.set("simulation.n_sims", 3000)

    def test_zero_noise_limit_equals_the_product_formula(self):
        p = proj()
        s = simulate(ResidualPools(pool(), pool()), p, self.cfg, seed=1)
        np.testing.assert_allclose(s["total_p50"], p["fppm"] * p["mpg"] * p["avail"] * 82.0, rtol=1e-9)
        np.testing.assert_allclose(s["fppg_p50"], p["fppm"] * p["mpg"], rtol=1e-9)

    def test_percentiles_are_ordered_and_reproducible(self):
        pools = ResidualPools(pool(0.1, 3.0, 0.1), pool(0.1, 3.0, 0.1))
        a = simulate(pools, proj(), self.cfg, seed=7)
        b = simulate(pools, proj(), self.cfg, seed=7)
        pd.testing.assert_frame_equal(a, b)
        for v in ("fppm", "minutes", "fppg", "total", "games"):
            cols = [f"{v}_p{q}" for q in (10, 25, 50, 75, 90)]
            self.assertTrue((a[cols].diff(axis=1).iloc[:, 1:] >= -1e-9).all().all(), v)
        self.assertTrue((a["total_cvar"] <= a["total_p10"] + 1e-9).all())

    def test_more_uncertainty_widens_the_interval_and_injury_cap_binds(self):
        pools = ResidualPools(pool(0.1, 3.0, 0.1), pool(0.1, 3.0, 0.1))
        w1 = simulate(pools, proj(), self.cfg, seed=3)
        w2 = simulate(pools, proj(mult=2.0), self.cfg, seed=3)
        self.assertTrue(((w2["total_p90"] - w2["total_p10"]) > (w1["total_p90"] - w1["total_p10"])).all())
        capped = simulate(pools, proj(avail_cap=0.0), self.cfg, seed=3)
        self.assertTrue((capped["total_p90"] == 0).all())            # "out for the season"
        banked = simulate(pools, proj(fp_to_date=500.0, games_rem=0.0), self.cfg, seed=3)
        np.testing.assert_allclose(banked["total_p50"], 500.0)        # nothing left to play: only banked points remain

    def test_empirical_bayes_strength(self):
        noise, tau2, mean_units = 4.0, 0.01, 2000.0
        k = eb_prior_strength(tau2 + noise / mean_units, noise, mean_units, default=900)
        self.assertAlmostEqual(k, noise / tau2, delta=1e-6 * noise / tau2 + 1.0)
        self.assertEqual(eb_prior_strength(-1.0, noise, mean_units, default=900), 900)      # impossible variance -> default


def table(n=60, seed=0):
    r = np.random.default_rng(seed)
    t50 = r.uniform(500, 5000, n)
    return pd.DataFrame({
        "player_id": np.arange(n), "player_name": [f"P{i}" for i in range(n)], "team": "AAA", "pathway": "main",
        "total_p10": t50 * 0.6, "total_p25": t50 * 0.8, "total_p50": t50, "total_p75": t50 * 1.1, "total_p90": t50 * 1.2,
        "total_cvar": t50 * 0.45, "fppg_p50": t50 / 70 + r.normal(0, 2, n), "ref_total": t50 * r.uniform(0.8, 1.2, n),
        "projected_minutes": 25.0, "projected_games": 70.0, "projected_fppm": 1.0, "ref_fppm": r.uniform(0.8, 1.1, n),
        "minutes_change": r.normal(0, 3, n), "p_role_expansion": r.uniform(0, 1, n), "p_beat_last_season": r.uniform(0, 1, n)})


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.load()

    def test_three_rankings_and_risk_adjustment(self):
        t = rank_and_score(table(), self.cfg, None)
        n = len(t)
        for c in ("rank_total", "rank_fppg", "rank_risk_adjusted"):
            self.assertEqual(sorted(t[c]), list(range(1, n + 1)))
        self.assertTrue((t["total_p50"].diff().dropna() <= 1e-9).all())          # default order = median total
        lam = self.cfg.get("ranking.risk_lambda")
        np.testing.assert_allclose(t["risk_adjusted_value"], t["total_p50"] - lam * (t["total_p50"] - t["total_cvar"]))
        self.assertTrue(t["breakout_score"].between(0, 100).all())
        self.assertTrue(t["confidence_score"].between(0, 100).all())

    def test_consensus_is_comparison_only(self):
        t = rank_and_score(table(), self.cfg, None)
        cons = pd.DataFrame({"player_id": t["player_id"], "consensus_rank": t["rank_total"]})
        c = compare(t, cons, "total_p50", "rank_total")
        np.testing.assert_allclose(c["rank_diff_vs_consensus"], 0.0)
        np.testing.assert_allclose(c["value_diff_vs_consensus"], 0.0, atol=1e-9)
        cons2 = pd.DataFrame({"player_id": t["player_id"], "consensus_rank": t["rank_total"].max() + 1 - t["rank_total"]})
        c2 = compare(t, cons2, "total_p50", "rank_total")
        top = c2.iloc[0]                                                          # model's #1 is the market's last pick
        self.assertLess(top["rank_diff_vs_consensus"], 0)                         # model likes him more
        self.assertGreater(top["value_diff_vs_consensus"], 0)

    def test_name_matching(self):
        roster = pd.DataFrame({"player_id": [1, 2, 3, 4], "team": ["MEM", "SAC", "BOS", "DEN"],
                               "player_name": ["Jaren Jackson Jr.", "Dennis Schröder", "Marcus Smith", "Marcus Smith"]})
        names = pd.Series(["Jaren Jackson", "Dennis Schroder", "Marcus Smith", "Marcus Smith"])
        teams = pd.Series(["MEM", "SAC", "BOS", "DEN"])
        self.assertEqual(match_names(names, teams, roster).tolist(), [1, 2, 3, 4])
        out = match_names(names, None, roster)
        self.assertTrue(out.iloc[2:].isna().all() and out.iloc[:2].tolist() == [1, 2])   # ambiguous without a team


class OutputSchemaTests(unittest.TestCase):
    def test_spec_columns_have_the_required_names_and_order(self):
        required = ("rank_total rank_fppg rank_risk_adjusted player_id player_name team position age projected_minutes "
                    "projected_fppm projected_fppg projected_games projected_total_fantasy_points fppm_p10 fppm_p25 fppm_p50 "
                    "fppm_p75 fppm_p90 fppg_p10 fppg_p25 fppg_p50 fppg_p75 fppg_p90 total_p10 total_p25 total_p50 total_p75 "
                    "total_p90 upside_score risk_score confidence_score historical_fppm historical_fppg minutes_change "
                    "role_change team_effect teammate_effect coach_effect age_effect").split()
        self.assertEqual(SPEC_COLUMNS, required)


if __name__ == "__main__":
    unittest.main()

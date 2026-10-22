"""End-to-end run on a tiny offline synthetic league (10 teams, 5 short seasons)."""
import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests._common import ROOT, small_cfg  # noqa: F401
import project
from src import pipeline
from src.ranking import SPEC_COLUMNS

ARGS = SimpleNamespace(top=20, no_plots=False, no_reports=False, n_sims=None, seed=None)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()
        cls.res = pipeline.run_pipeline(cls.cfg, ARGS)
        cls.out = cls.cfg.outputs_dir
        cls.csv = pd.read_csv(cls.out / "rankings.csv")

    def test_rankings_csv_schema_and_integrity(self):
        t = self.csv
        self.assertEqual(list(t.columns[:len(SPEC_COLUMNS)]), SPEC_COLUMNS)
        n = len(t)
        for c in ("rank_total", "rank_fppg", "rank_risk_adjusted"):
            self.assertEqual(sorted(t[c]), list(range(1, n + 1)))
        self.assertTrue((t["rank_total"].diff().dropna() > 0).all())                  # default order = total value
        for v in ("fppm", "fppg", "total"):
            q = t[[f"{v}_p{p}" for p in (10, 25, 50, 75, 90)]]
            self.assertTrue((q.diff(axis=1).iloc[:, 1:] >= -1e-6).all().all(), v)
        # decomposition: FPPG = FPPM x minutes, total = FPPG x games (+ banked points in-season; none preseason)
        np.testing.assert_allclose(t["projected_fppg"], t["projected_fppm"] * t["projected_minutes"], rtol=2e-4)
        np.testing.assert_allclose(t["projected_total_fantasy_points"], t["projected_fppg"] * t["projected_games"], rtol=2e-4)
        self.assertEqual(t["player_id"].duplicated().sum(), 0)
        self.assertTrue((t["projected_games"] <= self.cfg.season_length + 1e-9).all())

    def test_no_ai_api_dependency_and_all_artifacts_exist(self):
        for f in ("rankings.csv", "model_performance.csv", "feature_importance.csv", "feature_policy.csv", "run_summary.json",
                  "fantasy_value_analysis.md", "uncertainty_calibration.csv", "data_quality/data_quality_report.md",
                  "data_quality/dropped_rows.csv", "data_quality/transformed_rows.csv"):
            self.assertTrue((self.out / f).exists(), f)
        self.assertEqual(len(list((self.out / "plots").glob("*.png"))), 10)
        self.assertGreaterEqual(len(list((self.out / "player_reports").glob("*.txt"))), 20)
        src = "".join(p.read_text() for p in (ROOT / "src").rglob("*.py")) + (ROOT / "project.py").read_text()
        for banned in ("anthropic", "openai", "import langchain"):
            self.assertNotIn(banned, src.lower().replace("anthropic.com", ""))

    def test_model_performance_and_feature_importance_contents(self):
        perf = pd.read_csv(self.out / "model_performance.csv")
        self.assertTrue({"fppm", "mpg", "avail", "total_fp", "fppg"} <= set(perf["target"]))
        for tgt in ("fppm", "mpg"):
            labels = set(perf[(perf["target"] == tgt)]["label"])
            self.assertTrue({"A_weighted_average", "B_ridge", "C_elastic_net", "D_gradient_boosting"} <= labels)
        self.assertEqual(int(perf[(perf["target"] == "fppm") & (perf["test_year"] == "mean")]["selected"].sum()), 1)
        imp = pd.read_csv(self.out / "feature_importance.csv")
        lin = imp[(imp["model"] == "ridge") & imp["coef_std"].notna()]
        self.assertGreater(len(lin), 10)
        self.assertGreater(imp["perm_importance_mean"].notna().sum(), 10)

    def test_data_quality_report_documents_the_injected_defects(self):
        md = (self.out / "data_quality" / "data_quality_report.md").read_text()
        self.assertIn("duplicate (game_id, player_id) row", md)
        drops = pd.read_csv(self.out / "data_quality" / "dropped_rows.csv")
        self.assertGreater(len(drops), 5)

    def test_player_report_content(self):
        r = self.res.rank.iloc[0]
        txt = pipeline._report_text(self.res, r)
        for needle in ("PLAYER:", "Projected Rank", "Projected FPPM", "Projected Minutes", "Projected FPPG", "Projected Games",
                       "Projected Total", "10th percentile total", "90th percentile total", "Historical",
                       "Main positive drivers", "Main negative drivers", "Confidence:"):
            self.assertIn(needle, txt)
        row, sugg = pipeline.find_player(self.res.rank, self.res.rank.iloc[3]["player_name"].lower())
        self.assertEqual(int(row["player_id"]), int(self.res.rank.iloc[3]["player_id"]))
        none, sugg = pipeline.find_player(self.res.rank, "zzzz nobody")
        self.assertIsNone(none)

    def test_cached_player_lookup_needs_no_pipeline_rerun(self):
        cached = pipeline.cached_result(self.cfg)
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached.rank), len(self.res.rank))

    def test_manual_minutes_override_and_determinism(self):
        top = self.res.rank.iloc[0]
        ovr = self.cfg.outputs_dir / "overrides_test.csv"
        ovr.write_text("player_name,player_id,expected_minutes_override,expected_games_override,team_override,exclude,notes\n"
                       f"{top['player_name']},,20.0,40,,,test\n")
        cfg = self.cfg.copy()
        cfg.set("paths.overrides_csv", str(ovr))
        a = pipeline.run_projection(cfg, SimpleNamespace(seed=5, n_sims=None), self.res.pdx, self.res.models).rank
        b = pipeline.run_projection(cfg, SimpleNamespace(seed=5, n_sims=None), self.res.pdx, self.res.models).rank
        r = a[a["player_id"] == top["player_id"]].iloc[0]
        self.assertAlmostEqual(r["projected_minutes"], 20.0)
        self.assertAlmostEqual(r["projected_games"], 40.0, delta=0.01)
        self.assertIn("minutes", r["override_applied"])
        self.assertLess(r["rank_total"], len(a) + 1)
        self.assertLess(r["total_p50"], top["total_p50"])                         # fewer minutes + games -> lower value
        pd.testing.assert_series_equal(a["total_p50"], b["total_p50"])           # same seed -> identical simulation

    def test_demo_check_against_the_hidden_season(self):
        chk = json.loads((self.out / "demo_holdout_check.json").read_text())
        self.assertGreater(chk["spearman_total"], 0.5)

    def test_cli_parser(self):
        a = project.build_parser().parse_args(["--top", "200", "--player", "A B", "--update-data", "--retrain", "--demo"])
        self.assertEqual((a.top, a.player, a.update_data, a.retrain, a.demo), (200, "A B", True, True, True))


class InSeasonTests(unittest.TestCase):
    def test_in_season_update_uses_games_already_played(self):
        cfg = small_cfg(partial_rounds=15, seed=9)
        res = pipeline.run_pipeline(cfg, SimpleNamespace(top=10, no_plots=True, no_reports=True, n_sims=800, seed=1))
        self.assertTrue(res.inseason)
        t = res.rank
        played = t[t["gp_to_date"] > 0]
        self.assertGreater(len(played), 30)
        rem = cfg.season_length - 15
        self.assertTrue((t["projected_games"] <= rem + 1e-9).all())               # only the remaining games are projected
        np.testing.assert_allclose(played["projected_total_fantasy_points"],
                                   played["fp_to_date"] + played["projected_fppg"] * played["projected_games"], rtol=2e-4)
        self.assertTrue((played["total_p10"] >= played["fp_to_date"] - 1e-6).all())   # banked points are a floor


if __name__ == "__main__":
    unittest.main()

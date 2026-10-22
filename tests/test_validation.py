import unittest
from unittest import mock

import numpy as np
import pandas as pd

from tests._common import ROOT  # noqa: F401
from src.config import Config
from src.feature_engineering import TrainingTable
from src import validation as V
from src.models import season_splits


class SplitTests(unittest.TestCase):
    def test_season_splits_are_strictly_chronological(self):
        years = np.repeat(np.arange(2016, 2024), 7)
        splits = season_splits(years, min_train=3)
        self.assertEqual(len(splits), 5)
        for tr, va in splits:
            self.assertLess(years[tr].max(), years[va].min())
            self.assertEqual(len(set(years[va])), 1)

    def test_test_years_follow_the_config(self):
        cfg = Config.load()
        ys = list(range(2016, 2026))
        self.assertEqual(V.test_years_for(ys, cfg), [2021, 2022, 2023, 2024, 2025])
        cfg.set("validation.n_test_seasons", 9)
        self.assertGreaterEqual(min(V.test_years_for(ys, cfg)), 2020)      # keeps >= min_train_seasons before the first test year
        with self.assertRaises(ValueError):
            V.test_years_for([2024, 2025], cfg)


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        y = np.arange(1.0, 21.0)
        m = V.regression_metrics(y, y, top_n=5)
        self.assertAlmostEqual(m["rmse"], 0.0)
        self.assertAlmostEqual(m["spearman"], 1.0)
        m2 = V.regression_metrics(y, y + 2.0, top_n=5)
        self.assertAlmostEqual(m2["rmse"], 2.0)
        self.assertAlmostEqual(m2["top_n_mae"], 2.0)
        self.assertAlmostEqual(m2["r2"], 1 - 4 * 20 / ((y - y.mean()) ** 2).sum())


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.load()          # selection metric top_n_mae, tolerance 1%

    def _sel(self, **vals):
        return V.select_model(pd.DataFrame({"top_n_mae": vals}), self.cfg)

    def test_prefers_the_simpler_model_when_performance_is_nearly_identical(self):
        self.assertEqual(self._sel(baseline=1.5, ridge=1.000, elastic_net=1.004, gbm=0.995), "ridge")

    def test_uses_the_complex_model_only_for_a_meaningful_gain(self):
        self.assertEqual(self._sel(baseline=1.5, ridge=1.000, elastic_net=0.999, gbm=0.950), "gbm")

    def test_benchmark_wins_when_nothing_beats_it(self):
        self.assertEqual(self._sel(baseline=0.990, ridge=1.000, elastic_net=1.001, gbm=1.02), "baseline")


class WalkForwardTests(unittest.TestCase):
    def test_no_training_row_is_from_the_test_season_or_later(self):
        n = 6 * 40
        years = np.repeat(np.arange(2018, 2024), 40)
        rng = np.random.default_rng(0)
        tab = TrainingTable(
            X=pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}),
            meta=pd.DataFrame({"year": years, "player_id": np.arange(n), "base": rng.normal(size=n)}),
            y=pd.DataFrame({"y": rng.normal(size=n), "y_min": 1000.0, "elig": True}))
        spec = V.TargetSpec("t", "y", "base", "elig", lambda Y, cfg: np.ones(len(Y)), min_rows=5)
        seen = []

        class Stub:
            def __init__(self, kind, cfg): pass
            def fit(self, X, y, base, w, years_):
                seen.append(np.unique(years_))
                return self
            def predict(self, X, base): return np.asarray(base)

        with mock.patch("src.validation.TargetModel", Stub):
            oof, _ = V.walk_forward(tab, spec, ["ridge"], [2021, 2022, 2023], Config.load())
        self.assertEqual(len(seen), 3)
        for T, ys in zip([2021, 2022, 2023], seen):
            self.assertLess(ys.max(), T)
        self.assertEqual(sorted(oof["year"].unique()), [2021, 2022, 2023])


class ReconcileTests(unittest.TestCase):
    def test_team_minutes_budget(self):
        cfg = Config.load()
        team = pd.Series(["A"] * 5 + ["B"] * 5)
        year = pd.Series([2024] * 10)
        mpg = pd.Series([40.0] * 5 + [30.0] * 5)                 # A: 200*... = 200 min; B: 150
        avail = pd.Series([1.5] * 10).clip(upper=1.0)
        mpg = pd.Series([48.0] * 5 + [30.0] * 5)                 # A sums to 240, make it 300 with avail=1.25? use direct numbers
        mpg = pd.Series([60.0] * 5 + [30.0] * 5)                 # team A: 300 expected minutes, team B: 150
        down = V.reconcile_minutes(team, year, mpg, avail, "down", cfg)
        self.assertAlmostEqual(float((down[:5] * avail[:5]).sum()), 240.0 + 0.0, delta=45.0)   # scaled towards 240 (capped by max_mpg / cap_down)
        self.assertTrue((down[5:] == mpg[5:]).all())              # 'down' never scales a short team up
        both = V.reconcile_minutes(team, year, mpg, avail, "both", cfg)
        self.assertGreater(float(both[5:].iloc[0]), 30.0)
        self.assertTrue((V.reconcile_minutes(team, year, mpg, avail, "none", cfg) == mpg).all())


if __name__ == "__main__":
    unittest.main()

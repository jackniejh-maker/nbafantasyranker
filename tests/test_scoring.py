import unittest

import pandas as pd

from tests._common import ROOT  # noqa: F401  (adds project root to sys.path)
from src.config import Config
from src.fantasy_scoring import (DEFAULT_ESPN, add_fantasy_columns, fantasy_points, get_weights, scoring_breakevens)


def line(**kw):
    base = dict(pts=0, fgm=0, fga=0, ftm=0, fta=0, fg3m=0, reb=0, ast=0, stl=0, blk=0, tov=0)
    base.update(kw)
    return pd.DataFrame([base])


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.w = Config.load().scoring

    def test_default_config_is_exactly_the_espn_table(self):
        self.assertEqual(self.w, {k: float(v) for k, v in DEFAULT_ESPN.items()})
        self.assertEqual(self.w["fg3m"], 1.0)          # 3PM is IN ADDITION to the FGM value

    def test_shot_values(self):
        fp = lambda **kw: float(fantasy_points(line(**kw), self.w).iloc[0])
        self.assertEqual(fp(pts=3, fgm=1, fga=1, fg3m=1), 5.0)    # made three: 3 + 2 - 1 + 1
        self.assertEqual(fp(fga=1), -1.0)                          # any missed shot
        self.assertEqual(fp(pts=2, fgm=1, fga=1), 3.0)            # made two
        self.assertEqual(fp(pts=1, ftm=1, fta=1), 1.0)            # made free throw
        self.assertEqual(fp(fta=1), -1.0)                          # missed free throw

    def test_full_formula(self):
        l = line(pts=25, fgm=10, fga=19, ftm=5, fta=6, fg3m=1, reb=12, ast=9, stl=2, blk=1, tov=3)
        expected = 25 + 2 * 10 - 19 + 5 - 6 + 1 + 12 + 2 * 9 + 4 * 2 + 4 * 1 - 2 * 3
        self.assertEqual(float(fantasy_points(l, self.w).iloc[0]), expected)

    def test_components_add_up(self):
        l = add_fantasy_columns(line(pts=25, fgm=10, fga=19, ftm=5, fta=6, fg3m=1, reb=12, ast=9, stl=2, blk=1, tov=3), self.w)
        self.assertAlmostEqual(float(l["fp_scoring"].iloc[0] + l["fp_peripheral"].iloc[0]), float(l["fp"].iloc[0]))
        self.assertEqual(float(l["fp_stocks"].iloc[0]), 12.0)

    def test_scoring_is_configurable(self):
        w = dict(self.w, stl=3.0)
        self.assertEqual(float(fantasy_points(line(stl=2), get_weights(w)).iloc[0]), 6.0)
        with self.assertRaises(ValueError):
            get_weights({"pts": 1})

    def test_breakevens(self):
        be = scoring_breakevens(self.w)
        self.assertAlmostEqual(be["breakeven_2pt_pct"], 0.25)
        self.assertAlmostEqual(be["breakeven_3pt_pct"], 1 / 6)
        self.assertAlmostEqual(be["breakeven_ft_pct"], 0.5)


if __name__ == "__main__":
    unittest.main()

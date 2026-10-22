import unittest

import numpy as np
import pandas as pd

from tests._common import ROOT, small_cfg  # noqa: F401
from src import preprocessing as pp
from src.data_quality import DataQuality
from src.fantasy_scoring import fantasy_points
from src.ingest import Ingestor


def _pg(**over):
    base = dict(year=2024, game_id="g1", game_date="2024-11-01", player_id=1, player_name="A B", team="AAA",
                matchup="AAA vs. BBB", min=30, fgm=5, fga=10, ftm=2, fta=3, fg3m=1, fg3a=3, oreb=1, dreb=4, reb=5,
                ast=3, stl=1, blk=0, tov=2, pf=2, pts=13, plus_minus=3)
    base.update(over)
    return base


class CleaningTests(unittest.TestCase):
    def test_every_dropped_or_repaired_row_is_logged(self):
        rows = [_pg(), _pg(),                                   # exact duplicate
                _pg(player_id=2, fga=np.nan),                   # missing statistic
                _pg(player_id=3, min=-3),                       # invalid minutes
                _pg(player_id=4, fgm=12, fga=10),               # makes > attempts -> repaired
                _pg(player_id=5, min=0),                        # zero minutes
                _pg(player_id=np.nan)]                          # missing id
        dq = DataQuality()
        out = pp.clean_player_games(pd.DataFrame(rows), dq)
        iss, drops, xf = dq.frames()
        self.assertEqual(len(out), 2)                           # player 1 (once) and the repaired player 4
        self.assertEqual(len(drops), 5)
        self.assertEqual(len(xf), 1)
        self.assertEqual(float(out.loc[out["player_id"] == 4, "fga"].iloc[0]), 12.0)
        self.assertIn("duplicate (game_id, player_id) row", set(drops["reason"]))
        self.assertTrue(all(drops["reason"].str.len() > 0))


class FormulaTests(unittest.TestCase):
    def test_advanced_rates_match_hand_calculation(self):
        r = dict(min=36, fga=20, fta=8, tov=4, fgm=9, fg3m=3, fg3a=8, ast=6, reb=10, oreb=2, dreb=8, stl=2, blk=1, pts=27,
                 tm_min=240, tm_fga=80, tm_fta=20, tm_tov=12, tm_fgm=36, tm_reb=44, tm_oreb=10, tm_dreb=34, tm_fg3a=24,
                 tm_poss=100, tm_opp_reb=40, tm_opp_oreb=12, tm_opp_dreb=28, tm_opp_fga=84, tm_opp_fg3a=24)
        a = pp._advanced(pd.DataFrame([r])).iloc[0]
        self.assertAlmostEqual(a["usg"], 100 * (20 + 0.44 * 8 + 4) * 48 / (36 * (80 + 0.44 * 20 + 12)), places=6)
        self.assertAlmostEqual(a["ast_pct"], 100 * 6 / ((36 / 48) * 36 - 9), places=6)          # = 33.33
        self.assertAlmostEqual(a["reb_pct"], 100 * 10 * 48 / (36 * 84), places=6)
        self.assertAlmostEqual(a["stl_pct"], 100 * 2 * 48 / (36 * 100), places=6)
        self.assertAlmostEqual(a["blk_pct"], 100 * 1 * 48 / (36 * 60), places=6)
        self.assertAlmostEqual(a["ts"], 27 / (2 * (20 + 0.44 * 8)), places=6)
        self.assertAlmostEqual(a["efg"], (9 + 1.5) / 20, places=6)
        self.assertAlmostEqual(a["tov_pct"], 100 * 4 / (20 + 0.44 * 8 + 4), places=6)

    def test_possession_formula(self):
        base = dict(year=2024, game_date=pd.Timestamp("2024-11-01"), min=240.0, ftm=0, fg3m=0, fg3a=0, stl=0, blk=0, ast=0, reb=0)
        a = dict(base, game_id="g1", team="AAA", matchup="AAA vs. BBB", fga=80, fta=20, oreb=10, dreb=36, fgm=36, tov=12, pts=100)
        b = dict(base, game_id="g1", team="BBB", matchup="BBB @ AAA", fga=84, fta=24, oreb=12, dreb=30, fgm=38, tov=14, pts=98)
        tg = pp.enrich_team_games(pd.DataFrame([a, b]))
        poss_a = 80 + 8 - 1.07 * (10 / (10 + 30)) * 44 + 12
        poss_b = 84 + 9.6 - 1.07 * (12 / (12 + 36)) * 46 + 14
        self.assertAlmostEqual(tg["poss"].iloc[0], 0.5 * (poss_a + poss_b), places=6)
        self.assertEqual(tg.set_index("team").loc["AAA", "home"], True)
        self.assertEqual(tg.set_index("team").loc["AAA", "opp"], "BBB")


class StructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = small_cfg()
        cls.cfg = cfg
        cls.dq = DataQuality()
        cls.pdx = pp.load_or_build(cfg, Ingestor(cfg).run(), cls.dq, force=True)

    def test_fantasy_points_come_from_box_scores(self):
        ps = self.pdx.player_seasons
        pg = self.pdx.player_games
        tot = pg.groupby(["player_id", "year"])[["pts", "fgm", "fga", "ftm", "fta", "fg3m", "reb", "ast", "stl", "blk", "tov"]].sum()
        fp = fantasy_points(tot, self.cfg.scoring).rename("fp")
        m = ps.set_index(["player_id", "year"])["fp"].to_frame().join(fp, rsuffix="_check")
        np.testing.assert_allclose(m["fp"], m["fp_check"], rtol=1e-9)

    def test_trades_keep_every_stint(self):
        ps, st = self.pdx.player_seasons, self.pdx.stints
        traded = ps[ps["traded"]]
        self.assertGreater(len(traded), 0)
        for _, r in traded.head(10).iterrows():
            s = st[(st["player_id"] == r["player_id"]) & (st["year"] == r["year"])]
            self.assertGreaterEqual(len(s), 2)
            self.assertAlmostEqual(s["min"].sum(), r["min"])                  # combined season = sum of stints
            self.assertEqual(s["gp"].sum(), r["gp"])
            self.assertLess(r["last_stint_gp"], r["gp"])                       # post-trade split is distinct
        self.assertTrue((ps["avail"] <= 1.0).all() and (ps["gp"] <= ps["team_games"] + 1e-9).all())

    def test_data_quality_report_flags_injected_defects(self):
        iss, drops, xf = self.dq.frames()
        reasons = set(drops["reason"])
        self.assertIn("duplicate (game_id, player_id) row", reasons)
        self.assertIn("missing box-score statistic", reasons)
        self.assertIn("invalid minutes (<=0 or >70)", reasons)
        self.assertGreater(len(xf), 0)

    def test_opening_roster_logic(self):
        stints = pd.DataFrame({"player_id": [1, 2, 3], "year": 2024, "team": "AAA",
                               "first_date": pd.to_datetime(["2024-10-25", "2024-12-20", "2024-12-20"]), "first_tg": [1, 30, 30]})
        ps = pd.DataFrame({"player_id": [2, 3, 4], "year": 2023, "last_team": ["AAA", "BBB", "AAA"]})
        ros = pd.DataFrame({"player_id": [4], "year": 2024, "team": "AAA"})        # never played, on the roster
        op = pp.build_opening_rosters(stints, ros, ps, k=10).set_index("player_id")
        self.assertEqual(set(op.index), {1, 2, 4})                    # 3 arrived mid-season from another team -> excluded
        self.assertEqual(op.loc[1, "src"], "opening")
        self.assertEqual(op.loc[2, "src"], "returning")               # injured at the start, same team as last season


if __name__ == "__main__":
    unittest.main()

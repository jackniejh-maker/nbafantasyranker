"""Hard requirement: nothing from season T (stats, ratings, minutes, injuries, coaching results) may enter T's features."""
import dataclasses
import unittest

import numpy as np
import pandas as pd

from tests._common import ROOT, small_cfg  # noqa: F401
from src.data_quality import DataQuality
from src.feature_engineering import FeatureBuilder, GROUPS
from src.game_model import GAME_FEATURES, build_game_dataset
from src.ingest import Ingestor
from src.preprocessing import load_or_build


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()
        cls.pdx = load_or_build(cls.cfg, Ingestor(cls.cfg).run(), DataQuality(), force=True)

    def _jolt(self, T):
        """Replace every numeric value belonging to seasons >= T by garbage."""
        rng = np.random.default_rng(0)
        out = {}
        for name in ("player_seasons", "stints", "team_seasons", "team_games", "player_games", "league"):
            d = getattr(self.pdx, name).copy()
            m = (d["year"] >= T).to_numpy()
            for c in d.select_dtypes("number").columns:
                if c not in ("year", "player_id", "team_game_no"):
                    d[c] = d[c].astype(float)
                    d.loc[m, c] = d.loc[m, c] * rng.uniform(0.2, 4.0, m.sum()) + rng.normal(0, 7, m.sum())
            out[name] = d
        return dataclasses.replace(self.pdx, **out)

    def test_features_are_unchanged_when_target_season_data_is_destroyed(self):
        for T in (self.cfg.last_completed_year, self.cfg.last_completed_year - 1):
            b1 = FeatureBuilder(self.pdx, self.cfg)
            roster = b1.roster_for(T)                       # roster membership is preseason information
            x1 = b1.build(T, roster=roster).X
            x2 = FeatureBuilder(self._jolt(T), self.cfg).build(T, roster=roster).X
            pd.testing.assert_frame_equal(x1, x2, check_exact=False, rtol=1e-9, atol=1e-9)

    def test_feature_names_are_registered_groups_and_no_targets(self):
        x = FeatureBuilder(self.pdx, self.cfg).build(self.cfg.last_completed_year).X
        self.assertFalse([c for c in x.columns if c.startswith("y_")])
        self.assertTrue(all(c.split("__")[0] in GROUPS for c in x.columns))

    def test_game_level_features_exclude_the_game_itself(self):
        pdx = self.pdx
        g1 = build_game_dataset(pdx, self.cfg).set_index(["player_id", "game_id"])
        pg = pdx.player_games
        pid = pg.groupby("player_id").size().idxmax()
        rows = pg[pg["player_id"] == pid].sort_values("game_date")
        target = rows.iloc[25]
        pg2 = pg.copy()
        hit = (pg2["player_id"] == pid) & (pg2["game_id"] == target["game_id"])
        pg2.loc[hit, ["fp", "min"]] = [999.0, 47.0]
        g2 = build_game_dataset(dataclasses.replace(pdx, player_games=pg2), self.cfg).set_index(["player_id", "game_id"])
        key = (pid, target["game_id"])
        same = g1.loc[key, GAME_FEATURES].astype(float).to_numpy()
        np.testing.assert_allclose(same, g2.loc[key, GAME_FEATURES].astype(float).to_numpy(), equal_nan=True)   # own game: unchanged
        nxt = (pid, rows.iloc[26]["game_id"])
        self.assertNotEqual(g1.loc[nxt, "r5_fppm"], g2.loc[nxt, "r5_fppm"])                                       # later games: changed


if __name__ == "__main__":
    unittest.main()

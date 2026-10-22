import unittest
from unittest import mock

import pandas as pd

from tests._common import ROOT, small_cfg  # noqa: F401
from src.data_sources.base import SourceUnavailable
from src.data_sources.demo import DemoSource
from src.ingest import IngestError, Ingestor


class IngestTests(unittest.TestCase):
    def test_completed_seasons_are_downloaded_once(self):
        cfg = small_cfg()
        Ingestor(cfg).run()
        calls = []
        orig = DemoSource.player_game_logs

        def spy(self, year, date_from=None):
            calls.append(year)
            return orig(self, year, date_from)

        with mock.patch.object(DemoSource, "player_game_logs", spy):
            raw = Ingestor(cfg).run()                                  # second run: history comes from data/raw
        self.assertEqual([y for y in calls if cfg.is_completed(y)], [])          # nothing historical re-downloaded
        self.assertGreater(len(raw.player_games), 1000)
        for y in cfg.history_years():
            self.assertEqual(raw.manifest[f"game_logs/{y}"]["source"], "demo")       # provenance recorded per dataset

    def test_missing_latest_season_is_fatal(self):
        cfg = small_cfg()
        orig = DemoSource.player_game_logs

        def flaky(self, year, date_from=None):
            if year == cfg.last_completed_year:
                raise SourceUnavailable("simulated outage")
            return orig(self, year, date_from)

        with mock.patch.object(DemoSource, "player_game_logs", flaky):
            with self.assertRaises(IngestError) as cm:
                Ingestor(cfg).run()
        self.assertIn("stale", str(cm.exception))

    def test_current_season_games_are_fetched_incrementally(self):
        cfg = small_cfg(partial_rounds=10)
        raw1 = Ingestor(cfg).run()
        n1 = int((raw1.player_games["year"] == cfg.target_year).sum())
        self.assertGreater(n1, 0)
        calls = []
        orig = DemoSource.player_game_logs

        def spy(self, year, date_from=None):
            calls.append((year, date_from))
            return orig(self, year, date_from)

        with mock.patch.object(DemoSource, "player_game_logs", spy):
            raw2 = Ingestor(cfg, force_current=True).run()
        self.assertEqual(calls, [(cfg.target_year, calls[0][1])])         # one call, only for the in-progress season
        self.assertIsNotNone(calls[0][1])                                  # ... starting from the last stored date (DateFrom)
        self.assertEqual(int((raw2.player_games["year"] == cfg.target_year).sum()), n1)    # de-duplicated merge

    def test_user_supplied_csv_replaces_an_unavailable_season(self):
        cfg = small_cfg()
        Ingestor(cfg).run()
        y = cfg.history_years()[0]
        df = pd.read_pickle(cfg.raw_dir / "game_logs" / f"{y}.pkl")
        import shutil
        shutil.rmtree(cfg.raw_dir / "game_logs")
        (cfg.manifest_path if hasattr(cfg, "manifest_path") else cfg.raw_dir / "manifest.json").unlink()
        (cfg.external_dir / "game_logs").mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.external_dir / "game_logs" / f"{y}.csv", index=False)
        with mock.patch.object(DemoSource, "player_game_logs", side_effect=SourceUnavailable("down")):
            ing = Ingestor(cfg)
            got = ing._ensure_year("game_logs", "player_game_logs", y, required=True)
        self.assertEqual(len(got), len(df))
        self.assertEqual(ing.manifest[f"game_logs/{y}"]["source"], "external_csv")


if __name__ == "__main__":
    unittest.main()

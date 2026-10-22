"""ESPN source: parsers (both column orders, DNPs, schema errors), and an offline end-to-end run against a fake ESPN API."""
import functools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from tests._common import ROOT  # noqa: F401
from tests.fake_espn import FakeESPN, FMT, KEYS_A, KEYS_B, LABELS_A, LABELS_B
from src import pipeline
from src.config import Config
from src.consensus import match_names
from src.data_sources import build_source
from src.data_sources import espn_stats as es
from src.data_sources.base import SourceUnavailable
from src.demo_data import generate_demo_league
from src.ingest import IngestError, Ingestor
from src.scraping.cache import DiskCache
from src.scraping.scraper import FetchError, HttpClient
from src.scraping.validators import ValidationError

ROW = dict(min=34, fgm=10, fga=19, ftm=6, fta=6, fg3m=2, fg3a=6, oreb=1, dreb=8, reb=9, ast=8, stl=2, blk=0, tov=3, pf=2,
           plus_minus=11, pts=28)


@functools.lru_cache(maxsize=None)
def league():
    return generate_demo_league(first_year=2021, target_year=2026, n_teams=10, n_rounds=30, seed=3, inject_issues=False)


def espn_cfg(root, rounds=30):
    cfg = Config.load(data_dir=Path(root) / "data", output_dir=Path(root) / "out").use_espn_stats()
    for k, v in {"seasons.first_season": 2021, "seasons.target_season": 2026, "seasons.season_length": rounds,
                 "espn_stats.workers": 3, "espn_stats.min_teams": 8, "espn_stats.min_players": 80,
                 "scraping.min_interval_seconds.espn_stats": 0.0, "scraping.max_retries": 2, "scraping.backoff_seconds": 0.0,
                 "sources.injuries": [], "sources.consensus": [],
                 "validation.n_test_seasons": 2, "validation.min_train_seasons": 2, "validation.inner_cv_min_seasons": 2,
                 "models.gbm_grid": [{"max_depth": 2, "n_estimators": 60, "learning_rate": 0.08}],
                 "models.ridge_alphas": [10, 100, 1000], "models.enet_alphas": [0.03, 0.1], "models.enet_l1_ratios": [0.5],
                 "simulation.n_sims": 800, "game_model.enabled": False, "eligibility.min_history_minutes": 300}.items():
        cfg.set(k, v)
    return cfg


def network(fake):
    """Route every HttpClient request to the fake ESPN and pretend the connection works."""
    return [mock.patch.object(HttpClient, "session", new_callable=mock.PropertyMock, return_value=fake),
            mock.patch("src.ingest.network_reachable", return_value=True)]


class NetworkCase(unittest.TestCase):
    def use(self, fake):
        for p in network(fake):
            p.start()
            self.addCleanup(p.stop)
        return fake


# ====================================================================== parsers
class ParserTests(unittest.TestCase):
    def test_columns_are_resolved_by_name_in_either_season_layout(self):
        for labels, keys in ((LABELS_A, KEYS_A), (LABELS_B, KEYS_B)):
            idx = es.column_index({"labels": labels, "keys": keys, "names": labels})
            st = es.parse_stats([FMT[l](ROW) for l in labels], idx)
            self.assertEqual((st["min"], st["fgm"], st["fga"], st["fg3m"], st["fg3a"], st["ftm"], st["fta"], st["oreb"], st["dreb"],
                              st["reb"], st["ast"], st["stl"], st["blk"], st["tov"], st["pf"], st["plus_minus"], st["pts"]),
                             (34, 10, 19, 2, 6, 6, 6, 1, 8, 9, 8, 2, 0, 3, 2, 11, 28))

    def test_minutes_formats_dnp_and_derived_rebounds(self):
        idx = es.column_index({"labels": ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "AST", "STL", "BLK", "TO", "PF", "PTS"]})
        vals = ["34:30", "5-10", "1-3", "2-2", "2", "6", "4", "1", "0", "2", "3", "13"]
        st = es.parse_stats(vals, idx)
        self.assertAlmostEqual(st["min"], 34.5)
        self.assertEqual(st["reb"], 8.0)                                  # REB column absent -> OREB + DREB
        self.assertIsNone(es.parse_stats(["--"] + vals[1:], idx))        # did not play
        self.assertIsNone(es.parse_stats(["0"] + vals[1:], idx))

    def test_pts_is_derived_when_missing(self):
        labels = [l for l in LABELS_A if l != "PTS"]
        st = es.parse_stats([FMT[l](ROW) for l in labels], es.column_index({"labels": labels}))
        self.assertEqual(st["pts"], 2 * (10 - 2) + 3 * 2 + 6)

    def test_unrecognised_columns_fail_loudly_and_show_what_espn_returned(self):
        payload = {"boxscore": {"players": [{"team": {"abbreviation": "LAL"}, "statistics": [
            {"labels": ["FOO", "BAR"], "keys": ["foo", "bar"], "athletes": [{"athlete": {"id": "1"}, "stats": ["1", "2"]}]}]}]}}
        with self.assertRaises(ValidationError) as cm:
            es.compact_box(payload)
        self.assertIn("labels=['FOO', 'BAR']", str(cm.exception))

    def test_box_score_with_too_few_players_is_rejected(self):
        with self.assertRaises(ValidationError) as cm:
            es.compact_box({"boxscore": {"players": []}})
        self.assertIn("boxscore keys=['players']", str(cm.exception))
        self.assertIn("player groups=0 []", str(cm.exception))

    def test_box_score_error_reports_nested_player_group_shape(self):
        payload = {"header": {"id": "401"}, "boxscore": {"players": [
            {"team": {"abbreviation": "LAL"}, "statistics": [{"keys": KEYS_A, "labels": LABELS_A, "athletes": [
                {"athlete": {"id": "1"}, "stats": ["--"] + ["0"] * (len(KEYS_A) - 1)},
                {"athlete": {"id": "2"}, "didNotPlay": True, "stats": []},
                {"athlete": {"id": "3"}, "stats": []},
                {"athlete": {"id": "4"}, "stats": ["invalid"] + ["0"] * (len(KEYS_A) - 1)},
            ]}]}
        ]}}
        with self.assertRaises(ValidationError) as cm:
            es.compact_box(payload)
        message = str(cm.exception)
        self.assertIn("event=401", message)
        self.assertIn("'did_not_play': 1", message)
        self.assertIn("'without_stats': 1", message)
        self.assertIn("'unparsed_minutes': 2", message)
        self.assertIn("'raw_min': '--'", message)

    def test_box_score_with_all_minutes_scrubbed_is_marked_unavailable(self):
        stats = [FMT[label](ROW) for label in LABELS_A]
        stats[0] = "--"
        athletes = [{"athlete": {"id": str(i)}, "stats": stats.copy()} for i in range(10)]
        box = es.compact_box({"header": {"id": "401"}, "boxscore": {
            "players": [{"team": {"abbreviation": "LAL"}, "statistics": [
                {"keys": KEYS_A, "labels": LABELS_A, "athletes": athletes}
            ]}]
        }})
        self.assertEqual(box["rows"], [])
        self.assertIn("event=401", box["unavailable"])

    def test_events_dates_and_abbreviations(self):
        ev = {"id": 401, "date": "2024-04-15T02:30Z", "seasonType": {"type": 2}, "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"id": "9", "abbreviation": "GS"}, "score": {"value": 112.0}},
            {"homeAway": "away", "team": {"id": "30"}, "score": "99"}], "status": {"type": {"completed": True}}}]}
        e = es.parse_event(ev, {"30": "CHA"})
        self.assertEqual((e["home"], e["away"], e["home_score"], e["away_score"], e["completed"]), ("GSW", "CHA", 112.0, 99.0, True))
        self.assertEqual(es.to_et_date(ev["date"]), pd.Timestamp("2024-04-14"))      # 10:30pm ET on the 14th
        self.assertIsNone(es.parse_event({"id": 1, "competitions": [{"competitors": []}]}))

    def test_positions_and_athlete_parsing(self):
        self.assertEqual(es.season_position(["PG", "PG", "SG"]), "G")
        self.assertEqual(es.season_position(["SG", "SF", "SF", "SG"]), "G-F")
        self.assertEqual(es.season_position(["PF", "C", "C"]), "F-C")
        a = es.compact_athlete({"athlete": {"id": "5", "fullName": "A B", "dateOfBirth": "2003-05-17T07:00Z", "displayHeight": "6' 9\"",
                                            "draft": {"displayText": "2024 Round 1 Pick 7"}, "debutYear": 2024,
                                            "birthPlace": {"country": "France"}}})
        self.assertEqual((a["dob"], a["height_in"], a["draft_year"], a["draft_round"], a["draft_pick"], a["country"]),
                         ("2003-05-17", 81.0, 2024, 1, 7, "France"))
        r = es.compact_roster({"athletes": [{"position": "guards", "items": [{"id": "7", "fullName": "C D"}]}],
                               "coach": [{"id": "1", "firstName": "E", "lastName": "F"}]})
        self.assertEqual((r["athletes"][0]["id"], r["coach"][0]["name"]), (7, "E F"))


class SlimCacheTests(unittest.TestCase):
    def test_transform_is_cached_not_the_raw_document_and_bad_transform_is_retried(self):
        cache = DiskCache(tempfile.mkdtemp())
        calls = []

        class S:
            def get(self, url, params=None, headers=None, timeout=None):
                calls.append(url)
                return SimpleNamespace(status_code=200, content=json.dumps({"big": "x" * 5000, "keep": 1}).encode(), headers={})

        c = HttpClient("slim_test", cache, user_agent="u", min_interval=0, max_retries=2, backoff=0, session=S())
        out = c.get("http://x", transform=lambda p: {"keep": p["keep"]})
        self.assertEqual(out, {"keep": 1})
        self.assertEqual(c.get("http://x", transform=lambda p: {"keep": p["keep"]}), {"keep": 1})
        self.assertEqual(len(calls), 1)                                                   # second read came from the cache
        self.assertLess(len(cache.get("slim_test", cache.make_key("http://x")).content), 50)
        with self.assertRaises(FetchError):
            c.get("http://y", transform=lambda p: p["missing"])                          # schema problem -> retried, never cached
        self.assertEqual(sum(u == "http://y" for u in calls), 2)


class IdMatchingTests(unittest.TestCase):
    def test_espn_ids_take_priority_over_names(self):
        roster = pd.DataFrame({"player_id": [3032977, 4278129], "team": ["MIL", "ATL"], "player_name": ["Giannis A", "Trae Y"]})
        names = pd.Series(["G. Antetokounmpo", "Trae Young"])
        out = match_names(names, None, roster, espn_ids=pd.Series(["3032977", None]))
        self.assertEqual(int(out.iloc[0]), 3032977)
        self.assertTrue(pd.isna(out.iloc[1]))                  # no id and no usable name match


# ====================================================================== end-to-end against the fake ESPN API
class EspnIngestTests(NetworkCase):
    @classmethod
    def setUpClass(cls):
        cls.L = league()
        cls.fake = FakeESPN(cls.L)
        for p in network(cls.fake):
            p.start()
            cls.addClassCleanup(p.stop)
        cls.cfg = espn_cfg(tempfile.mkdtemp())
        cls.raw = Ingestor(cls.cfg).run()

    def test_player_logs_equal_the_source_data(self):
        a = self.raw.player_games[self.raw.player_games["year"] < 2026]
        b = self.L.player_games
        self.assertEqual(len(a), len(b))
        m = a.merge(b, on=["game_id", "player_id"], suffixes=("_e", "_d"))
        self.assertEqual(len(m), len(b))
        for c in ["min", "fgm", "fga", "ftm", "fta", "fg3m", "fg3a", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts",
                  "plus_minus", "team", "year", "matchup", "player_name"]:
            pd.testing.assert_series_equal(m[f"{c}_e"], m[f"{c}_d"], check_names=False, check_dtype=False, obj=c)
        self.assertTrue((pd.to_datetime(m["game_date_e"]).dt.normalize() == pd.to_datetime(m["game_date_d"]).dt.normalize()).all())

    def test_team_logs_and_starters(self):
        t, d = self.raw.team_games, self.L.team_games
        m = t.merge(d, on=["game_id", "team"], suffixes=("_e", "_d"))
        self.assertEqual(len(m), len(d))
        for c in ["pts", "fgm", "fga", "fta", "oreb", "dreb", "reb", "ast", "tov", "wl"]:
            pd.testing.assert_series_equal(m[f"{c}_e"], m[f"{c}_d"], check_names=False, check_dtype=False, obj=c)
        self.assertTrue((m["min_e"] == 240).all())
        gp = self.raw.player_games.groupby(["player_id", "year"]).size().rename("gp")
        s = self.raw.starters.set_index(["player_id", "year"]).join(gp)
        self.assertTrue(((s["gs"] >= 0) & (s["gs"] <= s["gp"])).all())
        self.assertTrue(s["gs"].sum() > 0)

    def test_rosters_coaches_current_roster_and_player_index(self):
        d, ros = self.L.rosters, self.raw.rosters
        m = d.merge(ros, on=["year", "team", "player_id"], how="left", indicator=True, suffixes=("_d", "_e"))
        self.assertTrue((m["_merge"] == "both").all())                                   # every real roster spot is present
        self.assertTrue((pd.to_datetime(m["birth_date_d"]).dt.normalize() == pd.to_datetime(m["birth_date_e"]).dt.normalize()).all())
        dc = self.L.coaches[self.L.coaches["is_head"]]
        cm = dc.merge(self.raw.coaches, on=["year", "team"], suffixes=("_d", "_e"))
        self.assertEqual(len(cm), len(dc))
        self.assertTrue((cm["coach_name_d"] == cm["coach_name_e"]).all())
        cur, dcur = self.raw.current_rosters, self.L.current_rosters
        self.assertEqual(set(zip(cur["team"], cur["player_id"])), set(zip(dcur["team"], dcur["player_id"])))
        self.assertEqual(set(self.raw.current_coaches["coach_name"]), set(self.L.current_coaches[self.L.current_coaches["is_head"]]["coach_name"]))
        pi = self.L.player_index[self.L.player_index["draft_pick"].notna()].merge(self.raw.player_index, on="player_id", suffixes=("_d", "_e"))
        self.assertGreater(len(pi), 100)
        self.assertTrue((pi["draft_pick_d"] == pi["draft_pick_e"]).all() and (pi["from_year_d"] == pi["from_year_e"]).all())

    def test_second_run_is_served_from_disk(self):
        before = self.fake.calls["summary"]
        Ingestor(self.cfg).run()
        self.assertEqual(self.fake.calls["summary"], before)

    def test_http_cache_keeps_only_the_compact_box_score(self):
        gid = sorted(self.L.team_games["game_id"].unique())[5]
        cache = DiskCache(self.cfg.cache_dir / "http")
        entry = cache.get("espn_stats", cache.make_key(f"{es.SITE}/summary", {"event": gid}))
        body = json.loads(entry.content)
        self.assertEqual(set(body), {"rows"})
        self.assertLess(len(entry.content), 25_000)

    def test_source_profiles_cannot_be_mixed(self):
        cfg = Config.load(data_dir=self.cfg.data_dir, output_dir=self.cfg.outputs_dir)        # default 'nba' profile
        with self.assertRaises(IngestError) as cm:
            Ingestor(cfg)
        self.assertIn("source profile", str(cm.exception))


class EspnResilienceTests(NetworkCase):
    def test_completed_game_with_scrubbed_minutes_is_skipped_and_logged_as_partial(self):
        L = league()
        gid = str(sorted(L.team_games[L.team_games["year"] == 2022]["game_id"].unique())[3])
        fake = self.use(FakeESPN(L))
        original_summary = fake.summary

        def scrubbed_summary(event_id):
            payload = original_summary(event_id)
            if str(event_id) == gid:
                for group in payload["boxscore"]["players"]:
                    for block in group["statistics"]:
                        for athlete in block["athletes"]:
                            if not athlete.get("didNotPlay") and athlete.get("stats"):
                                athlete["stats"][0] = "--"
            return payload

        fake.summary = scrubbed_summary
        df = build_source("espn_stats", espn_cfg(tempfile.mkdtemp())).player_game_logs(2022)
        self.assertGreater(len(df), 0)
        self.assertNotIn(gid, set(df["game_id"].astype(str)))

    def test_failed_box_scores_never_produce_a_partial_season_and_only_they_are_retried(self):
        L = league()
        gid = sorted(L.team_games[L.team_games["year"] == 2022]["game_id"].unique())[3]
        fake = self.use(FakeESPN(L, fail_events={gid: 500}))
        cfg = espn_cfg(tempfile.mkdtemp())
        with self.assertRaises(SourceUnavailable) as cm:
            build_source("espn_stats", cfg).player_game_logs(2022)
        self.assertIn(gid, str(cm.exception))
        done = fake.calls["summary"]
        fake.fail_events.clear()
        df = build_source("espn_stats", cfg).player_game_logs(2022)
        self.assertEqual(fake.calls["summary"] - done, 1)                                 # only the failed game is requested again
        self.assertEqual(len(df), int((L.player_games["year"] == 2022).sum()))

    def test_roster_endpoint_that_ignores_the_season_parameter_is_detected(self):
        L = league()
        self.use(FakeESPN(L, ignore_season=True))
        ros, coa = build_source("espn_stats", espn_cfg(tempfile.mkdtemp())).rosters(2021)
        debut = L.player_index.set_index("player_id")["from_year"]
        self.assertTrue((ros["player_id"].map(debut) <= 2021).all())                       # no future players leaked in
        want = set(L.coaches[(L.coaches["year"] == 2021) & L.coaches["is_head"]]["coach_name"])
        self.assertEqual(set(coa["coach_name"]), want)                                     # coaches come from the core API fallback


class EspnPipelineTest(NetworkCase):
    def test_full_pipeline_runs_on_espn_ids(self):
        L = league()
        self.use(FakeESPN(L))
        cfg = espn_cfg(tempfile.mkdtemp())
        res = pipeline.run_pipeline(cfg, SimpleNamespace(top=10, no_plots=True, no_reports=True, n_sims=800, seed=1))
        t = res.rank
        self.assertEqual(set(t["player_id"]), set(L.current_rosters["player_id"]))
        self.assertTrue(t["total_p50"].notna().all() and (t["rank_total"].sort_values().to_numpy() == np.arange(1, len(t) + 1)).all())
        self.assertTrue((cfg.outputs_dir / "rankings.csv").exists())
        self.assertEqual(json.loads((cfg.raw_dir / "_profile.json").read_text())["profile"], "espn")


if __name__ == "__main__":
    unittest.main()

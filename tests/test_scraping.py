import json
import tempfile
import unittest

import requests

from tests._common import ROOT  # noqa: F401
from src.data_sources import nba, espn, rotowire
from src.scraping import validators as V
from src.scraping.cache import DiskCache
from src.scraping.scraper import FetchError, HttpClient, OfflineError


class FakeResp:
    def __init__(self, status=200, body=b"{}", headers=None):
        self.status_code, self.content, self.headers = status, body, headers or {}


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def client(session, cache, offline=False, source=None):
    sleeps = []
    c = HttpClient(source or f"t{id(session)}", cache, user_agent="ua", min_interval=1.0, max_retries=3, backoff=0.01,
                   offline=offline, session=session, sleep=sleeps.append)
    c.sleeps = sleeps
    return c


class CacheTests(unittest.TestCase):
    def test_roundtrip_ttl_and_checksum(self):
        cache = DiskCache(tempfile.mkdtemp())
        key = cache.make_key("http://x", {"a": 1})
        cache.put("ns", key, b"hello", url="http://x")
        self.assertEqual(cache.get("ns", key).content, b"hello")
        self.assertIsNotNone(cache.get("ns", key, max_age_hours=1))
        self.assertIsNone(cache.get("ns", key, max_age_hours=0))                 # always stale
        meta_p = cache._paths("ns", key)[1]
        meta = json.loads(meta_p.read_text())
        meta["fetched_at"] = "2000-01-01T00:00:00Z"
        meta_p.write_text(json.dumps(meta))
        self.assertIsNone(cache.get("ns", key, max_age_hours=24))                # expired
        self.assertIsNotNone(cache.get("ns", key, max_age_hours=None))           # never-expiring history
        cache._paths("ns", key)[0].write_bytes(b"tampered")
        self.assertIsNone(cache.get("ns", key))                                   # checksum mismatch -> miss


class ClientTests(unittest.TestCase):
    def test_cache_hit_avoids_second_request(self):
        s = FakeSession([FakeResp(body=b'{"a": 1}')])
        c = client(s, DiskCache(tempfile.mkdtemp()))
        self.assertEqual(c.get("http://x"), {"a": 1})
        self.assertEqual(c.get("http://x"), {"a": 1})
        self.assertEqual(s.calls, 1)
        self.assertEqual([r["from_cache"] for r in c.requests_log], [False, True])

    def test_retries_on_503_then_succeeds(self):
        s = FakeSession([FakeResp(503), requests.Timeout("slow"), FakeResp(body=b'{"ok": true}')])
        c = client(s, DiskCache(tempfile.mkdtemp()))
        self.assertEqual(c.get("http://x"), {"ok": True})
        self.assertEqual(s.calls, 3)

    def test_invalid_payload_is_never_cached(self):
        cache = DiskCache(tempfile.mkdtemp())
        s = FakeSession([FakeResp(body=b"<html>blocked</html>")] * 3)
        c = client(s, cache)
        with self.assertRaises(FetchError):
            c.get("http://x")
        self.assertIsNone(cache.get(c.source, cache.make_key("http://x")))

    def test_validator_rejects_schema_change_and_403_is_not_retried(self):
        s = FakeSession([FakeResp(403)])
        c = client(s, DiskCache(tempfile.mkdtemp()))
        with self.assertRaises(FetchError):
            c.get("http://x")
        self.assertEqual(s.calls, 1)
        with self.assertRaises(V.ValidationError):
            V.nba_resultsets(1, ("PLAYER_ID",))({"resultSets": [{"headers": ["X"], "rowSet": [[1]]}]})

    def test_offline_mode(self):
        cache = DiskCache(tempfile.mkdtemp())
        c = client(FakeSession([]), cache, offline=True)
        with self.assertRaises(OfflineError):
            c.get("http://x")
        cache.put(c.source, cache.make_key("http://x"), b'{"z": 2}')
        self.assertEqual(c.get("http://x"), {"z": 2})           # served from (possibly stale) cache

    def test_rate_limit_sleeps_between_calls(self):
        s = FakeSession([FakeResp(body=b"{}"), FakeResp(body=b"{}")])
        c = HttpClient("ratelimit_test", DiskCache(tempfile.mkdtemp()), user_agent="u", min_interval=5.0, session=s,
                       sleep=lambda t: sleeps.append(t), clock=lambda: 100.0)
        sleeps = []
        c.get("http://a")
        c.get("http://b")
        self.assertTrue(sleeps and sleeps[-1] > 0)


class ParserTests(unittest.TestCase):
    def test_nba_game_log_parsing(self):
        headers = ["SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID",
                   "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
                   "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]
        row = ["22025", 203999, "Nikola Jokic", 1610612743, "DEN", "Nuggets", "0022500001", "2025-10-22", "DEN vs. LAL", "W",
               35, 10, 18, .55, 2, 5, .4, 5, 6, .83, 3, 9, 12, 9, 2, 1, 3, 2, 27, 11]
        payload = {"resultSets": [{"name": "LeagueGameLog", "headers": headers, "rowSet": [row]}]}
        df = nba.canonical_game_logs(nba.parse_result_sets(payload)["LeagueGameLog"], 2025)
        r = df.iloc[0]
        self.assertEqual((r["player_id"], r["team"], r["year"], r["min"], r["fg3m"]), (203999, "DEN", 2025, 35, 2))
        self.assertTrue(r["game_date"].year == 2025)

    def test_nba_roster_and_coach_parsing(self):
        ros = nba.NBAStatsSource._canon_roster(__import__("pandas").DataFrame({
            "team": ["DEN", "DEN"], "PLAYER_ID": [1, 2], "PLAYER": ["A B", "C D"], "POSITION": ["G-F", "C"],
            "HEIGHT": ["6-8", "7-0"], "WEIGHT": ["220", "250"], "BIRTH_DATE": ["MAR 10, 1994", "DEC 01, 2005"],
            "AGE": [31, 20], "EXP": ["R", "5"], "SCHOOL": ["x", "y"], "HOW_ACQUIRED": ["", ""]}), 2025)
        self.assertEqual(ros["exp"].tolist(), [0, 5])
        self.assertEqual(ros["height_in"].tolist(), [80, 84])
        self.assertEqual(ros["birth_date"].iloc[0].year, 1994)
        coa = nba.NBAStatsSource._canon_coaches(__import__("pandas").DataFrame({
            "team": ["DEN", "DEN"], "COACH_ID": [1, 2], "COACH_NAME": ["H", "A"], "COACH_TYPE": ["Head Coach", "Assistant Coach"],
            "IS_ASSISTANT": [0, 1]}), 2025)
        self.assertEqual(coa["is_head"].tolist(), [True, False])

    def test_espn_and_rotowire_parsers(self):
        inj = espn.parse_injuries({"injuries": [{"displayName": "Denver Nuggets", "injuries": [
            {"status": "Out", "shortComment": "knee", "athlete": {"displayName": "Some Guy", "id": "9"},
             "details": {"returnDate": "2026-11-05", "type": "Knee"}}]}]})
        self.assertEqual((inj.iloc[0]["team"], inj.iloc[0]["status"]), ("DEN", "Out"))
        fan = espn.parse_fantasy_players({"players": [{"id": 1, "player": {"fullName": "A B", "ownership": {
            "averageDraftPosition": 3.5, "percentOwned": 99}, "draftRanksByRankType": {"STANDARD": {"rank": 4}}}}]})
        self.assertEqual((fan.iloc[0]["adp"], fan.iloc[0]["espn_rank"]), (3.5, 4))
        rw = rotowire.parse_injury_table([{"player": "A B", "team": "DEN", "status": "Out", "injury": "Ankle", "est_return": "Nov 10"}])
        self.assertEqual(rw.iloc[0]["injury_type"], "Ankle")



class OfflineIngestTests(unittest.TestCase):
    def test_missing_network_and_no_cache_fails_fast_with_a_helpful_message(self):
        from unittest import mock
        from src.config import Config
        from src.ingest import IngestError, Ingestor
        cfg = Config.load(data_dir=tempfile.mkdtemp(), output_dir=tempfile.mkdtemp())
        with mock.patch("src.ingest.network_reachable", return_value=False):
            ing = Ingestor(cfg)
            self.assertTrue(ing.offline)
            with self.assertRaises(IngestError) as cm:
                ing.run()
        self.assertIn("--demo", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

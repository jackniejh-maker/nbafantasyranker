"""A fake ESPN API: serves ESPN-shaped JSON generated from a synthetic league, so the ESPN source can be tested offline.

The shapes follow ESPN's public endpoints (teams, team schedule, summary/box score, team roster, athlete, core coaches).
Box-score column ORDER alternates between seasons on purpose, as it does at ESPN, to prove columns are resolved by name."""
import collections
import json
import re
import threading
from urllib.parse import urlparse

import pandas as pd

from src.data_sources.espn_stats import FALLBACK_TEAM_IDS

ESPN_AB = {"GSW": "GS", "NYK": "NY", "NOP": "NO", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"}
LABELS_A = ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "+/-", "PTS"]
KEYS_A = ["minutes", "fieldGoalsMade-fieldGoalsAttempted", "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
          "freeThrowsMade-freeThrowsAttempted", "offensiveRebounds", "defensiveRebounds", "rebounds", "assists", "steals",
          "blocks", "turnovers", "fouls", "plusMinus", "points"]
LABELS_B = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
KEYS_B = ["minutes", "points", "fieldGoalsMade-fieldGoalsAttempted", "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
          "freeThrowsMade-freeThrowsAttempted", "rebounds", "assists", "turnovers", "steals", "blocks", "offensiveRebounds",
          "defensiveRebounds", "fouls", "plusMinus"]
FMT = {
    "MIN": lambda r: str(int(r["min"])), "FG": lambda r: f"{int(r['fgm'])}-{int(r['fga'])}",
    "3PT": lambda r: f"{int(r['fg3m'])}-{int(r['fg3a'])}", "FT": lambda r: f"{int(r['ftm'])}-{int(r['fta'])}",
    "OREB": lambda r: str(int(r["oreb"])), "DREB": lambda r: str(int(r["dreb"])), "REB": lambda r: str(int(r["reb"])),
    "AST": lambda r: str(int(r["ast"])), "STL": lambda r: str(int(r["stl"])), "BLK": lambda r: str(int(r["blk"])),
    "TO": lambda r: str(int(r["tov"])), "PF": lambda r: str(int(r["pf"])), "+/-": lambda r: f"{int(r['plus_minus']):+d}",
    "PTS": lambda r: str(int(r["pts"]))}
POS = {"G": "PG", "G-F": "SG", "F": "SF", "F-C": "PF", "C": "C"}


class Resp:
    def __init__(self, status=200, obj=None):
        self.status_code = status
        self.content = json.dumps(obj if obj is not None else {}).encode()
        self.headers = {}


class FakeESPN:
    def __init__(self, league, ignore_season=False, core_coaches=True, fail_events=None):
        self.L = league
        self.ignore_season, self.core_coaches = ignore_season, core_coaches
        self.fail_events = dict(fail_events or {})
        pg = league.player_games.drop_duplicates(["game_id", "player_id"])
        self.abbrs = sorted(pg["team"].unique())
        self.tid = {a: FALLBACK_TEAM_IDS[a] for a in self.abbrs}
        self.rev = {v: k for k, v in self.tid.items()}
        self.by_game = {g: d.to_dict("records") for g, d in pg.groupby("game_id")}
        self.games = {}
        for r in league.team_games.itertuples():
            d = self.games.setdefault(r.game_id, {"year": int(r.year), "date": r.game_date, "teams": {}})
            d["teams"][r.team] = {"pts": float(r.pts), "home": " vs. " in r.matchup}
        self.pos = league.player_index.set_index("player_id")["position"].to_dict()
        self.pidx = league.player_index.set_index("player_id")
        allr = pd.concat([league.rosters, league.current_rosters])
        self.dob = allr.dropna(subset=["birth_date"]).drop_duplicates("player_id").set_index("player_id")["birth_date"].to_dict()
        self.calls = collections.Counter()
        self.lock = threading.Lock()

    # ---- helpers ------------------------------------------------------------------------------------
    def _ab(self, a):
        return ESPN_AB.get(a, a)

    def _pos(self, pid):
        return POS.get(self.pos.get(pid, "F"), "SF")

    def event(self, gid):
        g = self.games[gid]
        comps = [{"homeAway": "home" if i["home"] else "away",
                  "team": {"id": str(self.tid[ab]), "abbreviation": self._ab(ab)},
                  "score": {"value": i["pts"], "displayValue": str(int(i["pts"]))}} for ab, i in g["teams"].items()]
        return {"id": gid, "date": f"{pd.Timestamp(g['date']):%Y-%m-%d}T17:00Z", "seasonType": {"type": 2},
                "competitions": [{"competitors": comps, "status": {"type": {"completed": True, "name": "STATUS_FINAL"}}}]}

    def summary(self, gid):
        year = self.games[gid]["year"]
        labels, keys = (LABELS_A, KEYS_A) if year % 2 == 0 else (LABELS_B, KEYS_B)
        groups = []
        for ab in self.games[gid]["teams"]:
            rows = sorted([r for r in self.by_game[gid] if r["team"] == ab], key=lambda r: -r["min"])
            ath = [{"athlete": {"id": str(r["player_id"]), "displayName": r["player_name"],
                                "position": {"abbreviation": self._pos(r["player_id"])}},
                    "starter": i < 5, "didNotPlay": False, "stats": [FMT[l](r) for l in labels]} for i, r in enumerate(rows)]
            ath.append({"athlete": {"id": "9999999", "displayName": "Bench Warmer"}, "starter": False, "didNotPlay": True,
                        "reason": "COACH'S DECISION", "stats": []})
            groups.append({"team": {"id": str(self.tid[ab]), "abbreviation": self._ab(ab)},
                           "statistics": [{"names": labels, "keys": keys, "labels": labels, "athletes": ath, "totals": []}]})
        return {"boxscore": {"players": groups, "teams": []}, "header": {"id": gid}, "plays": ["x" * 200] * 50}

    def roster(self, tid, season, honor_season=False):
        ab = self.rev[tid]
        L = self.L
        if season is None or (self.ignore_season and not honor_season):
            ros, coa = L.current_rosters[L.current_rosters["team"] == ab], L.current_coaches
            coa = coa[(coa["team"] == ab) & coa["is_head"]]
        else:
            ros = L.rosters[(L.rosters["year"] == season - 1) & (L.rosters["team"] == ab)]
            coa = L.coaches[(L.coaches["year"] == season - 1) & (L.coaches["team"] == ab) & L.coaches["is_head"]]
        ath = [{"id": str(r.player_id), "fullName": r.player_name, "displayName": r.player_name,
                "position": {"abbreviation": self._pos(r.player_id)}, "dateOfBirth": f"{pd.Timestamp(r.birth_date):%Y-%m-%d}T08:00Z",
                "height": float(r.height_in), "weight": float(r.weight), "experience": {"years": int(r.exp)},
                "college": {"name": "State U"}} for r in ros.itertuples()]
        coach = [{"id": str(c.coach_id), "firstName": c.coach_name.partition(" ")[0], "lastName": c.coach_name.partition(" ")[2]}
                 for c in coa.itertuples()]
        return {"team": {"id": str(tid)}, "athletes": ath, "coach": coach}

    def athlete(self, pid):
        if pid not in self.pidx.index:
            return None
        r = self.pidx.loc[pid]
        d = {"id": str(pid), "fullName": r["player_name"], "displayName": r["player_name"], "height": float(r["height_in"]),
             "weight": float(r["weight"]), "position": {"abbreviation": self._pos(pid)}, "debutYear": int(r["from_year"]),
             "experience": {"years": max(0, 2026 - int(r["from_year"]))}, "birthPlace": {"country": r["country"]}}
        if pid in self.dob:
            d["dateOfBirth"] = f"{pd.Timestamp(self.dob[pid]):%Y-%m-%d}T08:00Z"
        if pd.notna(r["draft_pick"]):
            d["draft"] = {"year": int(r["draft_year"]), "round": int(r["draft_round"]), "selection": int(r["draft_pick"])}
        return {"athlete": d}

    # ---- the "network" -------------------------------------------------------------------------------
    def get(self, url, params=None, headers=None, timeout=None):
        path, params = urlparse(url).path, params or {}
        with self.lock:
            kind = ("summary" if path.endswith("/summary") else "schedule" if path.endswith("/schedule")
                    else "roster" if path.endswith("/roster") else "athlete" if "/athletes/" in path
                    else "teams" if path.endswith("/nba/teams") else "core" if "/seasons/" in path else "other")
            self.calls[kind] += 1
        if kind == "teams":
            return Resp(200, {"sports": [{"leagues": [{"teams": [
                {"team": {"id": str(self.tid[a]), "abbreviation": self._ab(a)}} for a in self.abbrs]}]}]})
        m = re.search(r"/teams/(\d+)/schedule$", path)
        if m:
            year, tid = int(params["season"]) - 1, int(m.group(1))
            ab = self.rev.get(tid)
            evs = [self.event(g) for g, info in sorted(self.games.items()) if info["year"] == year and ab in info["teams"]]
            return Resp(200, {"events": evs})
        if kind == "summary":
            gid = str(params["event"])
            if gid in self.fail_events:
                return Resp(self.fail_events[gid])
            return Resp(200, self.summary(gid)) if gid in self.by_game else Resp(404)
        m = re.search(r"/teams/(\d+)/roster$", path)
        if m:
            tid = int(m.group(1))
            return Resp(200, self.roster(tid, int(params["season"]) if "season" in params else None)) if tid in self.rev \
                else Resp(200, {"athletes": [], "coach": []})
        m = re.search(r"/athletes/(\d+)$", path)
        if m:
            a = self.athlete(int(m.group(1)))
            return Resp(200, a) if a else Resp(404)
        m = re.search(r"/seasons/(\d+)/teams/(\d+)/coaches$", path)
        if m and self.core_coaches:
            season, tid = int(m.group(1)), int(m.group(2))
            c = self.roster(tid, season, honor_season=True)["coach"] if tid in self.rev else []
            return Resp(200, {"items": [{"$ref": f"http://sports.core.api.espn.com/v2/x/seasons/{season}/coaches/{c[0]['id']}?lang=en"}]}) if c else Resp(404)
        m = re.search(r"/seasons/(\d+)/coaches/(\d+)$", path)
        if m and self.core_coaches:
            season, cid = int(m.group(1)), m.group(2)
            for tid in self.rev:
                for c in self.roster(tid, int(season), honor_season=True)["coach"]:
                    if c["id"] == cid:
                        return Resp(200, c)
        return Resp(404)

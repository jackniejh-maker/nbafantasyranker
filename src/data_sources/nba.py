"""stats.nba.com (structured JSON endpoints - no HTML scraping).

Endpoints used
--------------
leaguegamelog          league-wide player (P) and team (T) box scores, one request per season
leaguedashplayerstats  games STARTED per player (StarterBench=Starters)
commonteamroster       roster (+birth dates, positions) and head/assistant coaches per team-season
playerindex            historical player directory with draft position, college, country

NOTE: stats.nba.com occasionally blocks cloud IP ranges and changes parameters without notice.
Responses are schema-validated before caching, and failures raise FetchError so the ingestion
layer can fall back to another source / the local cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..scraping import validators as V
from ..scraping.scraper import FetchError
from ..utils import log, parse_height_inches, season_label
from .base import DataSource, SourceUnavailable

BASE = "https://stats.nba.com/stats/"
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}
TEAM_IDS = {
    "ATL": 1610612737, "BOS": 1610612738, "CLE": 1610612739, "NOP": 1610612740, "CHI": 1610612741,
    "DAL": 1610612742, "DEN": 1610612743, "GSW": 1610612744, "HOU": 1610612745, "LAC": 1610612746,
    "LAL": 1610612747, "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750, "BKN": 1610612751,
    "NYK": 1610612752, "ORL": 1610612753, "IND": 1610612754, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "OKC": 1610612760, "TOR": 1610612761,
    "UTA": 1610612762, "MEM": 1610612763, "WAS": 1610612764, "DET": 1610612765, "CHA": 1610612766,
}
BOX_COLS = ["min", "fgm", "fga", "ftm", "fta", "fg3m", "fg3a", "oreb", "dreb", "reb",
            "ast", "stl", "blk", "tov", "pf", "pts", "plus_minus"]

_DASH_PARAMS = {
    "College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "", "Division": "",
    "DraftPick": "", "DraftYear": "", "GameScope": "", "GameSegment": "", "Height": "", "ISTRound": "",
    "LastNGames": 0, "LeagueID": "00", "Location": "", "MeasureType": "Base", "Month": 0,
    "OpponentTeamID": 0, "Outcome": "", "PORound": 0, "PaceAdjust": "N", "PerMode": "Totals",
    "Period": 0, "PlayerExperience": "", "PlayerPosition": "", "PlusMinus": "N", "Rank": "N",
    "SeasonSegment": "", "SeasonType": "Regular Season", "ShotClockRange": "", "StarterBench": "",
    "TeamID": 0, "TwoWay": 0, "VsConference": "", "VsDivision": "", "Weight": "",
}


def parse_result_sets(payload: dict) -> dict[str, pd.DataFrame]:
    """{'resultSets': [{'name','headers','rowSet'}]} -> {name: DataFrame} (also handles 'resultSet')."""
    sets = payload.get("resultSets", payload.get("resultSet")) or []
    if isinstance(sets, dict):
        sets = [sets]
    out = {}
    for i, s in enumerate(sets):
        out[s.get("name", f"set{i}")] = pd.DataFrame(s.get("rowSet") or [], columns=s.get("headers") or [])
    return out


def canonical_game_logs(df: pd.DataFrame, year: int, team_level: bool = False) -> pd.DataFrame:
    """Raw leaguegamelog frame -> canonical lower-case schema."""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"team_abbreviation": "team"})
    for c in BOX_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["game_id"] = df["game_id"].astype(str)
    df["year"] = int(year)
    cols = ["year", "game_id", "game_date", "team_id", "team", "matchup", "wl"] + BOX_COLS
    if not team_level:
        cols = cols[:3] + ["player_id", "player_name"] + cols[3:]
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


class NBAStatsSource(DataSource):
    name = "nba"
    capabilities = frozenset({"game_logs", "team_game_logs", "starters", "rosters",
                              "current_rosters", "player_index"})

    # ------------------------------------------------------------------ plumbing
    def _call(self, endpoint: str, params: dict, max_age, validator):
        if self.http is None:
            raise SourceUnavailable("nba: no HTTP client configured")
        self.http.headers.update(NBA_HEADERS)
        return self.http.get(BASE + endpoint, params, max_age_hours=max_age, validator=validator)

    def _game_log_params(self, year: int, team_level: bool, date_from=None) -> dict:
        return {"Counter": 0, "DateFrom": date_from.strftime("%m/%d/%Y") if date_from is not None else "",
                "DateTo": "", "Direction": "ASC", "LeagueID": "00",
                "PlayerOrTeam": "T" if team_level else "P", "Season": season_label(year),
                "SeasonType": "Regular Season", "Sorter": "DATE"}

    def _min_rows(self, year: int, date_from) -> int:
        """Completed seasons must return rows; the in-progress/new season may legitimately be empty."""
        return 1 if (self.cfg.is_completed(year) and date_from is None) else 0

    # ------------------------------------------------------------------ datasets
    def player_game_logs(self, year: int, date_from=None) -> pd.DataFrame:
        need = ("PLAYER_ID", "GAME_ID", "GAME_DATE", "MIN", "FGA", "PTS")
        payload = self._call("leaguegamelog", self._game_log_params(year, False, date_from),
                             self.ttl("game_logs", year), V.nba_resultsets(self._min_rows(year, date_from), need))
        sets = parse_result_sets(payload)
        df = next(iter(sets.values()))
        return canonical_game_logs(df, year, team_level=False)

    def team_game_logs(self, year: int, date_from=None) -> pd.DataFrame:
        need = ("TEAM_ID", "GAME_ID", "GAME_DATE", "MIN", "FGA", "PTS")
        payload = self._call("leaguegamelog", self._game_log_params(year, True, date_from),
                             self.ttl("team_game_logs", year), V.nba_resultsets(self._min_rows(year, date_from), need))
        df = next(iter(parse_result_sets(payload).values()))
        return canonical_game_logs(df, year, team_level=True)

    def starters(self, year: int) -> pd.DataFrame:
        params = {**_DASH_PARAMS, "Season": season_label(year), "StarterBench": "Starters"}
        payload = self._call("leaguedashplayerstats", params, self.ttl("starters", year),
                             V.nba_resultsets(1, ("PLAYER_ID", "GP")))
        df = next(iter(parse_result_sets(payload).values()))
        out = pd.DataFrame({"year": int(year), "player_id": pd.to_numeric(df["PLAYER_ID"]),
                            "gs": pd.to_numeric(df["GP"])})
        return out

    def rosters(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        label = season_label(year)
        ros, coa, failed = [], [], []
        for abbr, tid in TEAM_IDS.items():
            try:
                payload = self._call("commonteamroster", {"LeagueID": "00", "Season": label, "TeamID": tid},
                                     self.ttl("rosters", year),
                                     V.nba_resultsets(0, ("PLAYER_ID",), set_name="CommonTeamRoster"))
            except FetchError as exc:
                failed.append(abbr)
                log.warning("nba roster %s %s failed: %s", abbr, label, exc)
                continue
            sets = parse_result_sets(payload)
            r = sets.get("CommonTeamRoster", pd.DataFrame())
            if len(r):
                r = r.copy()
                r["team"] = abbr
                ros.append(r)
            c = sets.get("Coaches", pd.DataFrame())
            if len(c):
                c = c.copy()
                c["team"] = abbr
                coa.append(c)
        if not ros:
            raise SourceUnavailable(f"nba: no roster data for {label} (failed teams: {failed})")
        if failed:
            raise SourceUnavailable(f"nba: rosters {label} incomplete - teams {failed} failed. Successful teams are cached, "
                                    "so simply rerun to retry only the missing ones.")
        return self._canon_roster(pd.concat(ros, ignore_index=True), year), \
            self._canon_coaches(pd.concat(coa, ignore_index=True) if coa else pd.DataFrame(), year)

    def current_rosters(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        ros, coa = self.rosters(self.cfg.target_year)
        if ros["team"].nunique() < 25 or len(ros) < 300:
            raise SourceUnavailable(
                f"nba: roster endpoint for {season_label(self.cfg.target_year)} looks incomplete "
                f"({ros['team'].nunique()} teams, {len(ros)} players)")
        return ros, coa

    def player_index(self) -> pd.DataFrame:
        params = {"Active": "", "AllStar": "", "College": "", "Country": "", "DraftPick": "", "DraftRound": "",
                  "DraftYear": "", "Height": "", "Historical": 1, "LeagueID": "00",
                  "Season": season_label(self.cfg.target_year), "TeamID": 0, "Weight": ""}
        payload = self._call("playerindex", params, self.ttl("player_index"),
                             V.nba_resultsets(100, ("PERSON_ID", "PLAYER_LAST_NAME")))
        df = next(iter(parse_result_sets(payload).values()))
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        out = pd.DataFrame({
            "player_id": pd.to_numeric(df["person_id"], errors="coerce"),
            "player_name": (df["player_first_name"].fillna("") + " " + df["player_last_name"].fillna("")).str.strip(),
            "position": df.get("position"),
            "height_in": df.get("height", pd.Series(index=df.index, dtype=object)).map(parse_height_inches),
            "weight": pd.to_numeric(df.get("weight"), errors="coerce"),
            "college": df.get("college"),
            "country": df.get("country"),
            "draft_year": pd.to_numeric(df.get("draft_year"), errors="coerce"),
            "draft_round": pd.to_numeric(df.get("draft_round"), errors="coerce"),
            "draft_pick": pd.to_numeric(df.get("draft_number"), errors="coerce"),
            "from_year": pd.to_numeric(df.get("from_year"), errors="coerce"),
            "to_year": pd.to_numeric(df.get("to_year"), errors="coerce"),
            "team": df.get("team_abbreviation"),
        })
        return out.dropna(subset=["player_id"]).reset_index(drop=True)

    # ------------------------------------------------------------------ canonicalisation
    @staticmethod
    def _canon_roster(df: pd.DataFrame, year: int) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        bd = pd.to_datetime(df.get("birth_date"), errors="coerce", format="mixed")
        exp = df.get("exp", pd.Series(index=df.index, dtype=object)).astype(str).str.upper().replace({"R": "0"})
        return pd.DataFrame({
            "year": int(year), "team": df["team"], "player_id": pd.to_numeric(df["player_id"], errors="coerce"),
            "player_name": df.get("player"), "position": df.get("position"),
            "height_in": df.get("height", pd.Series(index=df.index, dtype=object)).map(parse_height_inches),
            "weight": pd.to_numeric(df.get("weight"), errors="coerce"), "birth_date": bd,
            "age": pd.to_numeric(df.get("age"), errors="coerce"), "exp": pd.to_numeric(exp, errors="coerce"),
            "school": df.get("school"), "how_acquired": df.get("how_acquired"),
        }).dropna(subset=["player_id"]).reset_index(drop=True)

    @staticmethod
    def _canon_coaches(df: pd.DataFrame, year: int) -> pd.DataFrame:
        cols = ["year", "team", "coach_id", "coach_name", "coach_type", "is_head"]
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=cols)
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        ctype = df.get("coach_type", pd.Series("", index=df.index)).fillna("").astype(str)
        is_asst = pd.to_numeric(df.get("is_assistant"), errors="coerce").fillna(0) > 0
        is_head = ctype.str.contains("head", case=False) & ~ctype.str.contains("assistant", case=False)
        is_head = is_head | (~is_asst & ctype.eq(""))
        return pd.DataFrame({"year": int(year), "team": df["team"], "coach_id": df.get("coach_id"),
                             "coach_name": df.get("coach_name"), "coach_type": ctype, "is_head": is_head.values})[cols]

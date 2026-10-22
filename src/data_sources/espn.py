"""ESPN public JSON endpoints (no HTML scraping).

injuries        site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
current roster  site.api.espn.com/apis/site/v2/sports/basketball/nba/teams[/{id}/roster]  (fallback)
consensus       lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{year}/... (ESPN ADP / rank)

ESPN ids differ from NBA ids, so every ESPN table carries names + team and is matched to the
canonical player ids by `src/consensus.py::match_names` (unmatched rows are reported, not dropped).
The consensus data is used ONLY for comparison and never enters the predictive model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..scraping import validators as V
from ..utils import log, utc_now_iso
from .base import DataSource, SourceUnavailable

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
FANTASY = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{end_year}/segments/0/leaguedefaults/3"

ESPN_ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX"}
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
POS_MAP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C", "GF": "G-F", "FC": "F-C"}


def nba_abbr(a) -> str | None:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return None
    a = str(a).upper()
    return ESPN_ABBR.get(a, a)


def parse_injuries(payload: dict) -> pd.DataFrame:
    rows = []
    for t in payload.get("injuries", []) or []:
        team_abbr = TEAM_NAME_TO_ABBR.get(t.get("displayName"))
        for inj in t.get("injuries", []) or []:
            ath = inj.get("athlete") or {}
            det = inj.get("details") or {}
            typ = inj.get("type") or {}
            ab = nba_abbr((ath.get("team") or {}).get("abbreviation")) or team_abbr
            rows.append({
                "source": "espn", "player_name": ath.get("displayName") or ath.get("fullName"),
                "espn_id": ath.get("id"), "team": ab, "status": inj.get("status"),
                "injury_type": det.get("type") or (typ.get("description") if isinstance(typ, dict) else None),
                "detail": det.get("detail"), "return_date": det.get("returnDate"),
                "comment": inj.get("shortComment") or inj.get("longComment"), "updated": inj.get("date"),
            })
    cols = ["source", "player_name", "espn_id", "team", "status", "injury_type", "detail",
            "return_date", "comment", "updated"]
    df = pd.DataFrame(rows, columns=cols)
    df["return_date"] = pd.to_datetime(df["return_date"], errors="coerce", format="mixed")
    return df.dropna(subset=["player_name"]).reset_index(drop=True)


def parse_fantasy_players(payload: dict) -> pd.DataFrame:
    rows = []
    for p in payload.get("players", []) or []:
        pl = p.get("player") or {}
        own = pl.get("ownership") or {}
        ranks = pl.get("draftRanksByRankType") or {}
        pick = ranks.get("STANDARD") or (next(iter(ranks.values())) if ranks else {}) or {}
        rows.append({"player_name": pl.get("fullName"), "espn_id": p.get("id"),
                     "adp": own.get("averageDraftPosition"), "espn_rank": pick.get("rank"),
                     "percent_owned": own.get("percentOwned"),
                     "pro_team_id": pl.get("proTeamId")})
    df = pd.DataFrame(rows, columns=["player_name", "espn_id", "adp", "espn_rank", "percent_owned", "pro_team_id"])
    for c in ("adp", "espn_rank", "percent_owned"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["player_name"]).reset_index(drop=True)


class ESPNSource(DataSource):
    name = "espn"
    capabilities = frozenset({"injuries", "consensus", "current_rosters"})

    def _get(self, url, params=None, max_age=None, validator=None, headers=None):
        if self.http is None:
            raise SourceUnavailable("espn: no HTTP client configured")
        return self.http.get(url, params, max_age_hours=max_age, validator=validator, headers=headers)

    def injuries(self) -> pd.DataFrame:
        payload = self._get(f"{SITE}/injuries", None, self.ttl("injuries"), V.json_keys("injuries"))
        df = parse_injuries(payload)
        df["fetched_at"] = utc_now_iso()
        return df

    def consensus(self, year: int) -> pd.DataFrame:
        flt = {"players": {"filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]}, "limit": 450,
                           "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"}}}
        payload = self._get(FANTASY.format(end_year=int(year) + 1), {"view": "kona_player_info"},
                            self.ttl("consensus"), V.json_keys("players"),
                            headers={"X-Fantasy-Filter": json.dumps(flt), "Accept": "application/json"})
        df = parse_fantasy_players(payload)
        if len(df) < 50:
            raise SourceUnavailable("espn: fantasy endpoint returned too few players")
        return df

    def current_rosters(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        teams = self._get(f"{SITE}/teams", {"limit": 40}, self.ttl("current_rosters"), V.json_keys("sports"))
        try:
            tlist = [t["team"] for t in teams["sports"][0]["leagues"][0]["teams"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise SourceUnavailable(f"espn: unexpected teams payload ({exc})")
        year = self.cfg.target_year
        ros, coa = [], []
        for t in tlist:
            abbr = nba_abbr(t.get("abbreviation"))
            payload = self._get(f"{SITE}/teams/{t['id']}/roster", None, self.ttl("current_rosters"),
                                V.json_keys("athletes"))
            for a in payload.get("athletes", []) or []:
                pos = POS_MAP.get(((a.get("position") or {}).get("abbreviation") or "").upper())
                ros.append({"year": year, "team": abbr, "player_id": np.nan,
                            "player_name": a.get("fullName") or a.get("displayName"),
                            "position": pos, "height_in": pd.to_numeric(a.get("height"), errors="coerce"),
                            "weight": pd.to_numeric(a.get("weight"), errors="coerce"),
                            "birth_date": pd.to_datetime(a.get("dateOfBirth"), errors="coerce", utc=True),
                            "age": pd.to_numeric(a.get("age"), errors="coerce"),
                            "exp": pd.to_numeric((a.get("experience") or {}).get("years"), errors="coerce"),
                            "school": (a.get("college") or {}).get("name") if isinstance(a.get("college"), dict) else None,
                            "how_acquired": None})
            for c in payload.get("coach", []) or []:
                coa.append({"year": year, "team": abbr, "coach_id": c.get("id"),
                            "coach_name": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip(),
                            "coach_type": "Head Coach", "is_head": True})
        if len(ros) < 300:
            raise SourceUnavailable(f"espn: only {len(ros)} roster rows")
        r = pd.DataFrame(ros)
        r["birth_date"] = r["birth_date"].dt.tz_localize(None) if hasattr(r["birth_date"], "dt") else r["birth_date"]
        log.info("espn roster fallback: %d players (player_id unresolved -> matched by name later)", len(r))
        return r, pd.DataFrame(coa, columns=["year", "team", "coach_id", "coach_name", "coach_type", "is_head"])

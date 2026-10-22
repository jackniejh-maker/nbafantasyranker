"""ESPN as a complete replacement for stats.nba.com (select with `--source espn`).

Endpoints (all public JSON, no HTML scraping)
---------------------------------------------
teams       site.api.espn.com/apis/site/v2/sports/basketball/nba/teams
schedule    .../teams/{id}/schedule?season=YYYY&seasontype=2     every regular-season game id (30 requests / season)
scoreboard  .../scoreboard?dates=YYYYMMDD                        fallback game discovery
box score   .../summary?event={game id}                          BOTH teams' player lines, starter flags, DNPs
roster      .../teams/{id}/roster[?season=YYYY]                  athletes + head coach
athlete     site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}   birth date, size, draft, debut
coaches     sports.core.api.espn.com/v2/.../seasons/{YYYY}/teams/{id}/coaches          fallback when a roster has no coach

ESPN labels a season by its END year (season=2025 is 2024-25); this project uses the START year.

There is no league-wide game-log endpoint, so every game is one request (~1,230 / season).  To keep that practical:
* only a compact parsed box score is cached (a few KB instead of a 300 KB play-by-play document),
* downloads run on a small thread pool behind one global rate limit (`espn_stats.workers`, `scraping.min_interval_seconds.espn_stats`),
* completed games are cached forever, so later runs only fetch games played since,
* network/schema failures still raise rather than storing partial results. If ESPN marks every active player's minutes
  unavailable for a completed game, that event is skipped with a warning and the returned season is explicitly partial.

All player ids in this profile are ESPN athlete ids (rookies included) - do not mix with data downloaded from stats.nba.com
(the ingestion layer refuses to).  If a response is not shaped as expected the error names the missing piece and shows the
labels/keys ESPN actually returned, so a schema change is easy to diagnose.
"""
from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..scraping import validators as V
from ..scraping.scraper import FetchError
from ..scraping.validators import ValidationError
from ..utils import load_table, log, parse_height_inches
from .base import DataSource, SourceUnavailable
from .espn import POS_MAP, nba_abbr

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
WEB = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

FALLBACK_TEAM_IDS = {
    "ATL": 1, "BOS": 2, "NOP": 3, "CHI": 4, "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GSW": 9, "HOU": 10, "IND": 11,
    "LAC": 12, "LAL": 13, "MIA": 14, "MIL": 15, "MIN": 16, "BKN": 17, "NYK": 18, "ORL": 19, "PHI": 20, "PHX": 21,
    "POR": 22, "SAC": 23, "SAS": 24, "OKC": 25, "UTA": 26, "WAS": 27, "TOR": 28, "MEM": 29, "CHA": 30}

PLAYER_COLS = ["year", "game_id", "game_date", "player_id", "player_name", "team_id", "team", "matchup", "wl", "min",
               "fgm", "fga", "ftm", "fta", "fg3m", "fg3a", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
               "pts", "plus_minus"]
SUM_COLS = ["fgm", "fga", "ftm", "fta", "fg3m", "fg3a", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]

# ESPN box-score columns are identified by key/name/label; their ORDER differs between seasons, so we resolve by name.
STAT_ALIASES = {
    "min": {"min", "minutes"},
    "fg": {"fg", "fieldgoalsmade-fieldgoalsattempted", "fgm-fga"},
    "fg3": {"3pt", "3p", "threepointfieldgoalsmade-threepointfieldgoalsattempted", "3pm-3pa"},
    "ft": {"ft", "freethrowsmade-freethrowsattempted", "ftm-fta"},
    "oreb": {"oreb", "offensiverebounds"},
    "dreb": {"dreb", "defensiverebounds"},
    "reb": {"reb", "rebounds", "totalrebounds"},
    "ast": {"ast", "assists"},
    "stl": {"stl", "steals"},
    "blk": {"blk", "blocks"},
    "tov": {"to", "tov", "turnovers"},
    "pf": {"pf", "fouls", "personalfouls"},
    "plus_minus": {"+/-", "plusminus"},
    "pts": {"pts", "points"},
}
REQUIRED_STATS = ("min", "fg", "fg3", "ft", "ast", "stl", "blk", "tov")


# ====================================================================== pure parsing helpers (unit-tested)
def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def _num(v) -> float:
    if v is None:
        return np.nan
    s = str(v).strip()
    if s in ("", "--", "-", "None", "null", "nan"):
        return np.nan
    if ":" in s:
        m, _, sec = s.partition(":")
        try:
            return float(m) + float(sec) / 60.0
        except ValueError:
            return np.nan
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return np.nan


def _made_att(v) -> tuple[float, float]:
    m = re.match(r"^\s*(\d+)\s*[-/]\s*(\d+)\s*$", str(v))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def _none_if_nan(x):
    return None if isinstance(x, float) and np.isnan(x) else x


def column_index(block: dict) -> dict[str, int]:
    """canonical stat -> column position, using whichever of keys / names / labels identifies each column."""
    fields = [block.get(k) or [] for k in ("keys", "names", "labels")]
    cols: dict[str, int] = {}
    for i in range(max(len(f) for f in fields) if fields else 0):
        ids = {_norm(f[i]) for f in fields if i < len(f)}
        for stat, aliases in STAT_ALIASES.items():
            if ids & aliases:
                cols.setdefault(stat, i)
    return cols


def parse_stats(vals: list, idx: dict[str, int]) -> dict | None:
    """One athlete's stat strings -> canonical counting stats (None when the player did not play)."""
    def g(stat):
        i = idx.get(stat)
        return vals[i] if i is not None and i < len(vals) else None

    mins = _num(g("min"))
    if not mins > 0:
        return None
    fgm, fga = _made_att(g("fg"))
    tpm, tpa = _made_att(g("fg3"))
    ftm, fta = _made_att(g("ft"))
    oreb, dreb, reb = _num(g("oreb")), _num(g("dreb")), _num(g("reb"))
    if np.isnan(reb) and not (np.isnan(oreb) or np.isnan(dreb)):
        reb = oreb + dreb
    if np.isnan(oreb) and not np.isnan(reb) and not np.isnan(dreb):
        oreb = reb - dreb
    if np.isnan(dreb) and not np.isnan(reb) and not np.isnan(oreb):
        dreb = reb - oreb
    pts = _num(g("pts"))
    if np.isnan(pts):
        pts = 2 * (fgm - tpm) + 3 * tpm + ftm
    return {"min": mins, "fgm": fgm, "fga": fga, "ftm": ftm, "fta": fta, "fg3m": tpm, "fg3a": tpa, "oreb": oreb,
            "dreb": dreb, "reb": reb, "ast": _num(g("ast")), "stl": _num(g("stl")), "blk": _num(g("blk")),
            "tov": _num(g("tov")), "pf": _num(g("pf")), "pts": pts, "plus_minus": _num(g("plus_minus"))}


def compact_box(payload: dict) -> dict:
    """ESPN `summary` document -> {'rows': [...]} with one dict per player who played (JSON-safe, a few KB)."""
    groups = ((payload.get("boxscore") or {}).get("players")) or []
    rows = []
    group_shapes = []
    active_minutes = []
    for grp in groups:
        team = grp.get("team") or {}
        ab, tid = nba_abbr(team.get("abbreviation")), team.get("id")
        shape = {"team": ab, "athletes": 0, "did_not_play": 0, "without_stats": 0, "invalid_id": 0,
                 "unparsed_minutes": 0, "parsed": 0, "samples": []}
        for block in grp.get("statistics") or []:
            idx = column_index(block)
            missing = [k for k in REQUIRED_STATS if k not in idx]
            if missing and block.get("athletes"):
                raise ValidationError(f"box-score columns {missing} not recognised; ESPN returned labels={block.get('labels')} "
                                      f"keys={block.get('keys')} names={block.get('names')}")
            for a in block.get("athletes") or []:
                shape["athletes"] += 1
                stats = a.get("stats")
                stat_idx = idx.get("min")
                raw_min = stats[stat_idx] if stats and stat_idx is not None and stat_idx < len(stats) else None
                if len(shape["samples"]) < 2:
                    ath = a.get("athlete") or {}
                    shape["samples"].append({"id": ath.get("id"), "dnp": bool(a.get("didNotPlay")),
                                             "stat_count": len(stats) if stats else 0, "raw_min": raw_min})
                if a.get("didNotPlay"):
                    shape["did_not_play"] += 1
                    continue
                if not stats:
                    shape["without_stats"] += 1
                    continue
                ath = a.get("athlete") or {}
                try:
                    aid = int(ath.get("id"))
                except (TypeError, ValueError):
                    shape["invalid_id"] += 1
                    continue
                active_minutes.append(raw_min)
                st = parse_stats(stats, idx)
                if st is None:
                    shape["unparsed_minutes"] += 1
                    continue
                shape["parsed"] += 1
                rows.append({"id": aid, "name": ath.get("displayName") or ath.get("fullName"), "team": ab, "tid": tid,
                             "pos": (ath.get("position") or {}).get("abbreviation"), "starter": bool(a.get("starter")),
                             **{k: _none_if_nan(v) for k, v in st.items()}})
        group_shapes.append(shape)
    if len(rows) < 10:
        box = payload.get("boxscore") or {}
        event_id = (payload.get("header") or {}).get("id")
        active_players = sum(group["athletes"] - group["did_not_play"] for group in group_shapes)
        all_minutes_scrubbed = (
            not rows
            and active_minutes
            and len(active_minutes) == active_players
            and all(str(minutes).strip() in ("--", "-", "") for minutes in active_minutes)
        )
        if all_minutes_scrubbed:
            return {"rows": [], "unavailable": f"event={event_id}: ESPN returned '--' for every active player's minutes"}
        raise ValidationError(f"box score has only {len(rows)} player lines (not final yet, or schema change); "
                              f"event={event_id}; top-level keys={sorted(payload)[:12]}; "
                              f"boxscore keys={sorted(box)[:12]}; player groups={len(groups)} {group_shapes}")
    return {"rows": rows}


def parse_event(ev: dict, id_to_abbr: dict | None = None) -> dict | None:
    """Schedule / scoreboard event -> {event_id, date, completed, season_type, home, away, scores, ids}."""
    comp = (ev.get("competitions") or [{}])[0]
    st = ((comp.get("status") or ev.get("status") or {}).get("type")) or {}
    sides = {}
    for c in comp.get("competitors") or []:
        t = c.get("team") or {}
        score = c.get("score")
        if isinstance(score, dict):
            score = score.get("value", score.get("displayValue"))
        ab = nba_abbr(t.get("abbreviation")) or (id_to_abbr or {}).get(str(t.get("id")))
        sides[c.get("homeAway")] = (ab, _num(score), t.get("id"))
    if "home" not in sides or "away" not in sides or not ev.get("id"):
        return None
    stype = (ev.get("seasonType") or {}).get("type", (ev.get("season") or {}).get("type"))
    try:
        stype = int(stype) if stype is not None else None
    except (TypeError, ValueError):
        stype = None
    return {"event_id": str(ev["id"]), "date": ev.get("date") or comp.get("date"),
            "completed": bool(st.get("completed")) or str(st.get("name", "")).upper() == "STATUS_FINAL",
            "season_type": stype, "home": sides["home"][0], "away": sides["away"][0],
            "home_score": sides["home"][1], "away_score": sides["away"][1],
            "home_tid": sides["home"][2], "away_tid": sides["away"][2]}


def to_et_date(iso) -> pd.Timestamp:
    """ESPN timestamps are UTC; the NBA calendar date is US/Eastern (a 10:30pm ET tip is the next day in UTC)."""
    ts = pd.to_datetime(iso, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert("America/New_York").tz_localize(None).normalize()


def season_position(positions) -> str | None:
    """Per-game ESPN positions (PG, SF, C...) -> season position G / G-F / F / F-C / C from the share of games."""
    g = [POS_MAP.get(str(p).upper()) for p in positions if p]
    g = [x for x in g if x]
    if not g:
        return None
    share = Counter()
    for x in g:
        for grp in x.split("-"):
            share[grp] += 1.0 / len(x.split("-"))
    tot = sum(share.values())
    sg, sf, sc = (share[k] / tot for k in ("G", "F", "C"))
    if sg >= 0.25 and sf >= 0.25:
        return "G-F"
    if sf >= 0.25 and sc >= 0.25:
        return "F-C"
    return max((("G", sg), ("F", sf), ("C", sc)), key=lambda kv: kv[1])[0]


def _dob(s):
    return str(s)[:10] if s else None


def _height(a: dict):
    h = a.get("height")
    try:
        h = float(h)
        if h > 20:
            return h
    except (TypeError, ValueError):
        pass
    v = parse_height_inches(a.get("displayHeight"))
    return None if np.isnan(v) else v


def _college(a: dict):
    c = a.get("college")
    return c.get("name") if isinstance(c, dict) else (c if isinstance(c, str) else None)


def compact_roster(payload: dict) -> dict:
    ath = payload.get("athletes") or []
    if ath and isinstance(ath[0], dict) and "items" in ath[0]:          # grouped by position in some sports
        ath = [x for grp in ath for x in grp.get("items", [])]
    out = []
    for a in ath:
        try:
            aid = int(a.get("id"))
        except (TypeError, ValueError):
            continue
        out.append({"id": aid, "name": a.get("fullName") or a.get("displayName"),
                    "pos": (a.get("position") or {}).get("abbreviation"), "dob": _dob(a.get("dateOfBirth")),
                    "height_in": _height(a), "weight": a.get("weight"),
                    "exp": (a.get("experience") or {}).get("years") if isinstance(a.get("experience"), dict) else None,
                    "college": _college(a)})
    coaches = [{"id": c.get("id"), "name": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()}
               for c in (payload.get("coach") or []) if c]
    return {"athletes": out, "coach": coaches}


def compact_athlete(payload: dict) -> dict:
    a = payload.get("athlete") or payload
    if not a.get("id"):
        raise ValidationError(f"athlete payload has no id; keys={sorted(payload)[:10]}")
    d = a.get("draft") if isinstance(a.get("draft"), dict) else {}
    yr, rd, sel = d.get("year"), d.get("round"), d.get("selection", d.get("pick"))
    if yr is None and d.get("displayText"):
        m = re.search(r"(\d{4}).*?Round\s*(\d+).*?Pick\s*(\d+)", str(d["displayText"]), re.I)
        if m:
            yr, rd, sel = (int(x) for x in m.groups())
    return {"id": int(a["id"]), "name": a.get("fullName") or a.get("displayName"), "dob": _dob(a.get("dateOfBirth")),
            "height_in": _height(a), "weight": a.get("weight"), "pos": (a.get("position") or {}).get("abbreviation"),
            "debut": a.get("debutYear"), "exp": (a.get("experience") or {}).get("years") if isinstance(a.get("experience"), dict) else None,
            "college": _college(a), "country": (a.get("birthPlace") or {}).get("country") if isinstance(a.get("birthPlace"), dict) else None,
            "draft_year": yr, "draft_round": rd, "draft_pick": sel}


def compact_coach(payload: dict) -> dict:
    return {"id": payload.get("id"), "name": f"{payload.get('firstName', '')} {payload.get('lastName', '')}".strip()}


# ====================================================================== the source
class ESPNStatsSource(DataSource):
    name = "espn_stats"
    capabilities = frozenset({"game_logs", "team_game_logs", "starters", "rosters", "current_rosters", "player_index"})

    def __init__(self, cfg, http=None):
        super().__init__(cfg, http)
        self.workers = max(1, int(cfg.get("espn_stats.workers", 4)))
        self._teams: dict[str, int] | None = None
        self._games: dict[int, list[dict]] = {}
        self._details: dict[int, dict] = {}

    # ------------------------------------------------------------------ plumbing
    def _get(self, url, params=None, max_age=None, validator=None, transform=None):
        if self.http is None:
            raise SourceUnavailable("espn_stats: no HTTP client configured")
        return self.http.get(url, params, max_age_hours=max_age, validator=validator, transform=transform)

    def team_ids(self) -> dict[str, int]:
        if self._teams is None:
            try:
                payload = self._get(f"{SITE}/teams", {"limit": 40}, 24 * 7, V.json_keys("sports"))
                tl = payload["sports"][0]["leagues"][0]["teams"]
                self._teams = {nba_abbr(t["team"]["abbreviation"]): int(t["team"]["id"]) for t in tl}
            except Exception as exc:                                       # fall back to the well-known ids
                log.warning("espn teams endpoint unavailable (%s: %s) - using built-in team ids", type(exc).__name__, exc)
                self._teams = {}
            if len(self._teams) < int(self.cfg.get("espn_stats.min_teams", 25)):
                self._teams = dict(FALLBACK_TEAM_IDS)
        return self._teams

    def _max_age(self, year: int):
        return None if self.cfg.is_completed(year) else self.ttl("game_logs", year)

    # ------------------------------------------------------------------ game discovery
    def _events(self, year: int) -> pd.DataFrame:
        id_to_abbr = {str(v): k for k, v in self.team_ids().items()}
        found: dict[str, dict] = {}
        failed = 0
        for ab, tid in self.team_ids().items():
            try:
                payload = self._get(f"{SITE}/teams/{tid}/schedule", {"season": year + 1, "seasontype": 2},
                                    self._max_age(year), V.json_keys("events"))
            except (FetchError, SourceUnavailable) as exc:
                failed += 1
                log.warning("espn schedule %s %s failed: %s", ab, year, exc)
                continue
            for ev in payload.get("events") or []:
                e = parse_event(ev, id_to_abbr)
                if e and e["season_type"] in (None, 2):
                    found.setdefault(e["event_id"], e)
        expected = len(self.team_ids()) * self.cfg.season_length / 2.0
        if self.cfg.is_completed(year) and len(found) < 0.8 * expected:
            log.warning("espn schedules gave only %d of ~%d games for %s - sweeping the scoreboard by date", len(found), expected, year)
            for e in self._scoreboard_sweep(year, id_to_abbr):
                found.setdefault(e["event_id"], e)
            if len(found) < 0.8 * expected:
                raise SourceUnavailable(f"espn_stats: only {len(found)} of ~{expected:.0f} regular-season games found for {year} "
                                        f"({failed} schedule requests failed)")
        return pd.DataFrame(list(found.values()))

    def _scoreboard_sweep(self, year: int, id_to_abbr: dict) -> list[dict]:
        days = pd.date_range(f"{year}-10-01", f"{year + 1}-06-30", freq="D")

        def one(d):
            try:
                payload = self._get(f"{SITE}/scoreboard", {"dates": d.strftime("%Y%m%d"), "limit": 100},
                                    None if self.cfg.is_completed(year) else 6, V.json_keys("events"))
            except (FetchError, SourceUnavailable):
                return []
            return [e for e in (parse_event(ev, id_to_abbr) for ev in payload.get("events") or [])
                    if e and e["season_type"] in (None, 2)]

        out = []
        with ThreadPoolExecutor(self.workers) as ex:
            for res in ex.map(one, days):
                out.extend(res)
        return out

    # ------------------------------------------------------------------ box scores
    def _game_rows(self, ev: dict, year: int) -> list[dict]:
        box = self._get(f"{SITE}/summary", {"event": ev["event_id"]}, None, V.json_keys("boxscore"), transform=compact_box)
        if box.get("unavailable"):
            return []
        date = to_et_date(ev["date"])
        rows = []
        for r in box["rows"]:
            home = r["team"] == ev["home"] or (ev.get("home_tid") is not None and str(r.get("tid")) == str(ev["home_tid"]))
            own, opp_pts = (ev["home_score"], ev["away_score"]) if home else (ev["away_score"], ev["home_score"])
            wl = None if (np.isnan(own) or np.isnan(opp_pts)) else ("W" if own > opp_pts else "L")
            opp = ev["away"] if home else ev["home"]
            rows.append({"year": year, "game_id": ev["event_id"], "game_date": date, "player_id": r["id"],
                         "player_name": r["name"], "team_id": pd.to_numeric(r.get("tid"), errors="coerce"), "team": r["team"],
                         "matchup": f"{r['team']} vs. {opp}" if home else f"{r['team']} @ {opp}", "wl": wl,
                         "min": r["min"], "fgm": r["fgm"], "fga": r["fga"], "ftm": r["ftm"], "fta": r["fta"],
                         "fg3m": r["fg3m"], "fg3a": r["fg3a"], "oreb": r["oreb"], "dreb": r["dreb"], "reb": r["reb"],
                         "ast": r["ast"], "stl": r["stl"], "blk": r["blk"], "tov": r["tov"], "pf": r["pf"], "pts": r["pts"],
                         "plus_minus": r["plus_minus"], "position": r["pos"], "starter": r["starter"]})
        return rows

    def _season_games(self, year: int) -> list[dict]:
        if year in self._games:
            return self._games[year]
        ev = self._events(year)
        if ev.empty:
            self._games[year] = []
            return []
        done = ev[ev["completed"]].to_dict("records")
        rows, failed, skipped = [], [], []
        with ThreadPoolExecutor(self.workers) as ex:
            futs = {ex.submit(self._game_rows, e, year): e["event_id"] for e in done}
            for i, f in enumerate(as_completed(futs), 1):
                try:
                    game_rows = f.result()
                    if game_rows:
                        rows.extend(game_rows)
                    else:
                        skipped.append(futs[f])
                except Exception as exc:
                    failed.append((futs[f], f"{type(exc).__name__}: {exc}"))
                if i % 250 == 0:
                    log.info("espn box scores %s: %d/%d games", year, i, len(done))
        if failed:
            raise SourceUnavailable(
                f"espn_stats: {len(failed)} of {len(done)} box scores failed for {year}; first: event {failed[0][0]} -> "
                f"{failed[0][1][:1000]}. Successful games are cached - rerun to retry only the failed ones.")
        if skipped:
            log.warning("espn_stats: excluded %d of %d completed games for %s because ESPN has no player minutes "
                        "(sample event IDs: %s); game logs are incomplete",
                        len(skipped), len(done), year, skipped[:10])
        if done and len(skipped) == len(done):
            raise SourceUnavailable(f"espn_stats: all {len(done)} completed games for {year} lacked player minutes")
        self._games[year] = rows
        return rows

    def _frame(self, year: int, date_from=None) -> pd.DataFrame:
        df = pd.DataFrame(self._season_games(year))
        if df.empty:
            return pd.DataFrame(columns=PLAYER_COLS + ["position", "starter"])
        df["game_date"] = pd.to_datetime(df["game_date"])
        if date_from is not None:
            df = df[df["game_date"] >= pd.Timestamp(date_from).normalize()]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ datasets
    def player_game_logs(self, year: int, date_from=None) -> pd.DataFrame:
        df = self._frame(year, date_from)
        out = df[PLAYER_COLS].copy() if len(df) else df.reindex(columns=PLAYER_COLS)
        out["player_id"] = out["player_id"].astype("int64") if len(out) else out["player_id"]
        return out.reset_index(drop=True)

    def team_game_logs(self, year: int, date_from=None) -> pd.DataFrame:
        df = self._frame(year, date_from)
        cols = ["year", "game_id", "game_date", "team_id", "team", "matchup", "wl", "min"] + SUM_COLS + ["plus_minus"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        keys = ["year", "game_id", "game_date", "team", "matchup", "wl"]
        g = df.groupby(keys, as_index=False, dropna=False).agg(team_id=("team_id", "first"), tot_min=("min", "sum"),
                                                              **{c: (c, "sum") for c in SUM_COLS})
        g["min"] = 240.0 + 25.0 * np.maximum(0, np.round((g["tot_min"] - 240.0) / 25.0))      # regulation + overtimes
        g["plus_minus"] = np.nan
        return g[cols].reset_index(drop=True)

    def starters(self, year: int) -> pd.DataFrame:
        df = self._frame(year)
        if df.empty:
            return pd.DataFrame(columns=["year", "player_id", "gs"])
        s = df.groupby("player_id", as_index=False)["starter"].sum().rename(columns={"starter": "gs"})
        s["year"] = int(year)
        s["gs"] = s["gs"].astype(float)
        return s[["year", "player_id", "gs"]]

    # ------------------------------------------------------------------ athletes
    def _current_ids(self) -> set[int]:
        cur = load_table(self.cfg.raw_dir / "current_rosters" / "rosters")
        return set(cur["player_id"].dropna().astype(int)) if cur is not None else set()

    def _athlete_details(self, ids) -> dict[int, dict]:
        todo = [int(i) for i in ids if int(i) not in self._details]
        cur = self._current_ids()

        def one(i):
            age = self.ttl("player_index") if i in cur else None
            err = None
            for url in (f"{WEB}/athletes/{i}", f"{CORE}/athletes/{i}"):          # common v3 first, core API as fallback
                try:
                    return i, self._get(url, None, age, V.json_nonempty, transform=compact_athlete)
                except Exception as exc:
                    err = exc
            return i, {"_error": f"{type(err).__name__}: {err}"}

        with ThreadPoolExecutor(self.workers) as ex:
            for n, (i, d) in enumerate(ex.map(one, todo), 1):
                self._details[i] = d
                if n % 250 == 0:
                    log.info("espn athlete details: %d/%d", n, len(todo))
        bad = [i for i in todo if "_error" in self._details[i]]
        if bad:
            log.warning("espn athlete details failed for %d of %d athletes (first: %s)", len(bad), len(todo),
                        self._details[bad[0]]["_error"][:160])
        return self._details

    def _roster_rows(self, year: int, items: list[dict], coaches: list[dict]):
        """items: dicts with player_id, name, team, position, official (roster dict or None)."""
        ids = [it["player_id"] for it in items]
        det = self._athlete_details(ids)
        ref = pd.Timestamp(year + 1, 2, 1)
        rows = []
        for it in items:
            o, d = it.get("official") or {}, det.get(it["player_id"], {})
            dob = pd.to_datetime(o.get("dob") or d.get("dob"), errors="coerce")
            debut = d.get("debut")
            exp = (year - debut) if isinstance(debut, (int, float)) and not pd.isna(debut) else o.get("exp")
            pos = it.get("position") or POS_MAP.get(str(o.get("pos") or d.get("pos") or "").upper())
            rows.append({"year": year, "team": it["team"], "player_id": it["player_id"],
                         "player_name": it["name"] or o.get("name") or d.get("name"), "position": pos,
                         "height_in": o.get("height_in") or d.get("height_in"),
                         "weight": pd.to_numeric(o.get("weight") if o.get("weight") is not None else d.get("weight"), errors="coerce"),
                         "birth_date": dob, "age": (ref - dob).days / 365.25 if pd.notna(dob) else np.nan,
                         "exp": max(0, exp) if exp is not None and not pd.isna(exp) else np.nan,
                         "school": o.get("college") or d.get("college"), "how_acquired": None})
        ros = pd.DataFrame(rows)
        coa = pd.DataFrame(coaches, columns=["year", "team", "coach_id", "coach_name", "coach_type", "is_head"])
        return ros, coa

    # ------------------------------------------------------------------ rosters
    def _core_coach(self, tid: int, season: int, completed: bool):
        try:
            lst = self._get(f"{CORE}/seasons/{season}/teams/{tid}/coaches", {"lang": "en", "region": "us"},
                            None if completed else 24, V.json_keys("items"))
            ref = ((lst.get("items") or [{}])[0]).get("$ref")
            if not ref:
                return None
            return self._get(ref.replace("http://", "https://"), None, None if completed else 24, V.json_nonempty,
                             transform=compact_coach)
        except Exception:
            return None

    def rosters(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """End-of-season rosters.  Players come from the season's box scores (always correct for that season); the
        official roster endpoint adds players who never played and the head coach - but only for teams where it
        demonstrably returns THAT season's roster (the endpoint may silently ignore `season`)."""
        g = pd.DataFrame(self._season_games(year))
        if g.empty:
            raise SourceUnavailable(f"espn_stats: no box scores for {year}; rosters are derived from them")
        per_team = g.groupby(["player_id", "team"]).agg(n=("game_id", "nunique"), last=("game_date", "max"),
                                                       name=("player_name", "last")).reset_index()
        final = per_team.sort_values(["player_id", "last", "n"]).drop_duplicates("player_id", keep="last")
        positions = g.groupby("player_id")["position"].agg(list)
        completed = self.cfg.is_completed(year)
        season = year + 1
        official, ignored = {}, []
        for ab, tid in self.team_ids().items():
            try:
                cp = self._get(f"{SITE}/teams/{tid}/roster", {"season": season}, None if completed else self.ttl("rosters", year),
                               V.json_keys("athletes"), transform=compact_roster)
            except (FetchError, SourceUnavailable) as exc:
                log.warning("espn roster %s %s failed: %s", ab, year, exc)
                continue
            mine = set(per_team[(per_team["team"] == ab) & (per_team["n"] >= 10)]["player_id"])
            if mine and len({a["id"] for a in cp["athletes"]} & mine) / len(mine) < 0.5:
                ignored.append(ab)
                continue
            official[ab] = cp
        if ignored:
            log.warning("espn roster endpoint ignored season=%s for %d teams (%s...) - using box-score-derived rosters there",
                        season, len(ignored), ignored[:5])
        items = []
        official_team = {a["id"]: ab for ab, cp in official.items() for a in cp["athletes"]}
        for r in final.itertuples():
            team = official_team.get(int(r.player_id), r.team)          # validated end-of-season roster wins over "last game"
            o = next((a for a in official.get(team, {}).get("athletes", []) if a["id"] == r.player_id), None)
            items.append({"player_id": int(r.player_id), "name": r.name, "team": team, "official": o,
                          "position": season_position(positions.get(r.player_id, []))})
        seen = set(final["player_id"])
        for ab, cp in official.items():                                    # on the roster, never played
            for a in cp["athletes"]:
                if a["id"] not in seen:
                    items.append({"player_id": a["id"], "name": a["name"], "team": ab, "official": a, "position": None})
                    seen.add(a["id"])
        coaches = []
        for ab, tid in self.team_ids().items():
            c = (official.get(ab, {}).get("coach") or [None])[0] or self._core_coach(tid, season, completed)
            if c and c.get("name"):
                coaches.append({"year": year, "team": ab, "coach_id": c.get("id"), "coach_name": c["name"],
                                "coach_type": "Head Coach", "is_head": True})
        return self._roster_rows(year, items, coaches)

    def current_rosters(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        year = self.cfg.target_year
        items, coaches = [], []
        for ab, tid in self.team_ids().items():
            try:
                cp = self._get(f"{SITE}/teams/{tid}/roster", None, self.ttl("current_rosters"), V.json_keys("athletes"),
                               transform=compact_roster)
            except (FetchError, SourceUnavailable) as exc:
                raise SourceUnavailable(f"espn_stats: current roster for {ab} failed ({exc}); successful teams are cached, rerun to retry")
            for a in cp["athletes"]:
                items.append({"player_id": a["id"], "name": a["name"], "team": ab, "official": a,
                              "position": POS_MAP.get(str(a.get("pos") or "").upper())})
            if cp["coach"] and cp["coach"][0].get("name"):
                c = cp["coach"][0]
                coaches.append({"year": year, "team": ab, "coach_id": c.get("id"), "coach_name": c["name"],
                                "coach_type": "Head Coach", "is_head": True})
        if len({i["team"] for i in items}) < int(self.cfg.get("espn_stats.min_teams", 25)) \
                or len(items) < int(self.cfg.get("espn_stats.min_players", 300)):
            raise SourceUnavailable(f"espn_stats: current rosters look incomplete ({len(items)} players on "
                                    f"{len({i['team'] for i in items})} teams)")
        return self._roster_rows(year, items, coaches)

    def player_index(self) -> pd.DataFrame:
        ids = set(self._current_ids())
        for y in self.cfg.history_years() + [self.cfg.target_year]:
            if y in self._games:
                ids |= {r["player_id"] for r in self._games[y]}
            else:
                t = load_table(self.cfg.raw_dir / "game_logs" / str(y))
                if t is not None and len(t):
                    ids |= set(t["player_id"].dropna().astype(int))
        if not ids:
            raise SourceUnavailable("espn_stats: no player ids known yet (download game logs first)")
        det = self._athlete_details(sorted(ids))
        good = {i: d for i, d in det.items() if i in ids and "_error" not in d}
        if len(good) < 0.5 * len(ids):
            raise SourceUnavailable(f"espn_stats: athlete details available for only {len(good)} of {len(ids)} players "
                                    f"(first error: {next((d['_error'] for d in det.values() if '_error' in d), 'n/a')[:160]})")
        rows = [{"player_id": i, "player_name": d.get("name"), "position": POS_MAP.get(str(d.get("pos") or "").upper()),
                 "height_in": d.get("height_in"), "weight": d.get("weight"), "college": d.get("college"),
                 "country": d.get("country"), "draft_year": d.get("draft_year"), "draft_round": d.get("draft_round"),
                 "draft_pick": d.get("draft_pick"), "from_year": d.get("debut"), "to_year": np.nan, "team": None}
                for i, d in good.items()]
        out = pd.DataFrame(rows)
        for c in ("height_in", "weight", "draft_year", "draft_round", "draft_pick", "from_year"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    # ------------------------------------------------------------------ diagnostics (used by check_sources.py)
    def probe(self) -> list[tuple[str, bool, str]]:
        """A handful of tiny live requests; returns (label, ok, detail)."""
        res = []

        def run(label, fn):
            try:
                res.append((label, True, str(fn())[:120]))
            except Exception as exc:
                res.append((label, False, f"{type(exc).__name__}: {str(exc)[:200]}"))

        y = self.cfg.last_completed_year
        tid = next(iter(self.team_ids().values()))
        state = {}
        run("teams", lambda: f"{len(self.team_ids())} teams")
        def sched():
            p = self._get(f"{SITE}/teams/{tid}/schedule", {"season": y + 1, "seasontype": 2}, 24, V.json_keys("events"))
            evs = [e for e in (parse_event(x) for x in p["events"]) if e and e["completed"]]
            state["ev"] = evs[0]
            return f"{len(evs)} completed games; first {evs[0]['event_id']} {evs[0]['away']}@{evs[0]['home']}"
        run("schedule", sched)
        if "ev" in state:
            run("box score", lambda: f"{len(self._get(f'{SITE}/summary', {'event': state['ev']['event_id']}, 24, V.json_keys('boxscore'), transform=compact_box)['rows'])} player lines")
        run("roster (current)", lambda: f"{len(self._get(f'{SITE}/teams/{tid}/roster', None, 24, V.json_keys('athletes'), transform=compact_roster)['athletes'])} athletes")
        run("roster (season param)", lambda: f"{len(self._get(f'{SITE}/teams/{tid}/roster', {'season': y + 1}, 24, V.json_keys('athletes'), transform=compact_roster)['athletes'])} athletes")
        run("athlete details", lambda: self._get(f"{WEB}/athletes/3032977", None, 24, V.json_nonempty, transform=compact_athlete))
        return res

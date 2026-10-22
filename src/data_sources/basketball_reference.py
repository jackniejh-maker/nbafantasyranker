"""Basketball-Reference season tables (HTML) - secondary source used to CROSS-CHECK the NBA data.

Sports-Reference asks scrapers to stay under ~20 requests/minute; the default throttle here is
one request every 3.6 s and every page is cached forever once a season is complete.
Run `python project.py --crosscheck` to compare games/minutes against the NBA-derived totals;
discrepancies are written into the data-quality report.  The model itself never depends on it.
"""
from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

from ..scraping import validators as V
from ..utils import log, normalize_name
from .base import DataSource, SourceUnavailable

URL = "https://www.basketball-reference.com/leagues/NBA_{end_year}_{kind}.html"


def parse_season_table(html: str, kind: str = "totals") -> pd.DataFrame:
    """Parse the `<kind>_stats` table; keeps the 'TOT' row for traded players, drops split rows."""
    tables = pd.read_html(StringIO(html), attrs={"id": f"{kind}_stats"})
    if not tables:
        raise SourceUnavailable(f"basketball_reference: table {kind}_stats not found")
    df = tables[0]
    df = df[df["Player"].astype(str) != "Player"].copy()
    team_col = "Team" if "Team" in df.columns else "Tm"
    df = df.rename(columns={team_col: "team", "Player": "player_name"})
    df.columns = [str(c).lower().replace("%", "_pct").replace("/", "_") for c in df.columns]
    for c in df.columns:
        if c not in ("player_name", "team", "pos", "awards"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # traded players: keep only the combined TOT row
    tot_players = set(df.loc[df["team"].isin(["TOT", "2TM", "3TM", "4TM"]), "player_name"])
    keep = df["team"].isin(["TOT", "2TM", "3TM", "4TM"]) | ~df["player_name"].isin(tot_players)
    return df[keep].reset_index(drop=True)


class BasketballReferenceSource(DataSource):
    name = "basketball_reference"
    capabilities = frozenset({"season_totals"})

    def season_totals(self, year: int, kind: str = "totals") -> pd.DataFrame:
        if self.http is None:
            raise SourceUnavailable("basketball_reference: no HTTP client configured")
        html = self.http.get(URL.format(end_year=int(year) + 1, kind=kind), None,
                             max_age_hours=None if self.cfg.is_completed(year) else 24,
                             validator=V.html_contains(f"{kind}_stats"), expect="text")
        return parse_season_table(html, kind)


def crosscheck(player_seasons: pd.DataFrame, bbref: pd.DataFrame, year: int,
               tol_games: int = 2, tol_minutes_pct: float = 0.03) -> pd.DataFrame:
    """Compare games/minutes per player between NBA-derived and Basketball-Reference totals."""
    mine = player_seasons[player_seasons["year"] == year].copy()
    mine["key"] = mine["player_name"].map(normalize_name)
    ref = bbref.copy()
    ref["key"] = ref["player_name"].map(normalize_name)
    m = mine.merge(ref[["key", "g", "mp"]], on="key", how="inner")
    m["games_diff"] = m["gp"] - m["g"]
    m["minutes_diff_pct"] = (m["min"] - m["mp"]) / m["mp"].replace(0, np.nan)
    bad = m[(m["games_diff"].abs() > tol_games) | (m["minutes_diff_pct"].abs() > tol_minutes_pct)]
    log.info("crosscheck %s: matched %d players, %d discrepancies", year, len(m), len(bad))
    return bad[["player_id", "player_name", "year", "gp", "g", "games_diff", "min", "mp", "minutes_diff_pct"]]

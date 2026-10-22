"""Rotowire injury report (JSON table endpoint used by their public injury page).

Secondary injuries source: merged behind ESPN so a single outage never blanks the injury context.
Best effort - the endpoint is undocumented and may change; parsing is defensive.
"""
from __future__ import annotations

import pandas as pd

from ..scraping import validators as V
from ..utils import utc_now_iso
from .base import DataSource, SourceUnavailable

URL = "https://www.rotowire.com/basketball/tables/injury-report.php"


def parse_injury_table(payload) -> pd.DataFrame:
    rows = []
    for r in payload if isinstance(payload, list) else []:
        name = r.get("player") or f"{r.get('firstname', '')} {r.get('lastname', '')}".strip()
        rows.append({"source": "rotowire", "player_name": name, "espn_id": None, "team": r.get("team"),
                     "status": r.get("status"), "injury_type": r.get("injury"), "detail": None,
                     "return_date": r.get("est_return"), "comment": None, "updated": None})
    cols = ["source", "player_name", "espn_id", "team", "status", "injury_type", "detail",
            "return_date", "comment", "updated"]
    df = pd.DataFrame(rows, columns=cols)
    # est_return strings such as "Oct 25" carry no year -> let pandas guess, NaT when impossible
    df["return_date"] = pd.to_datetime(df["return_date"], errors="coerce", format="mixed")
    return df[df["player_name"].astype(str).str.len() > 0].reset_index(drop=True)


class RotowireSource(DataSource):
    name = "rotowire"
    capabilities = frozenset({"injuries"})

    def injuries(self) -> pd.DataFrame:
        if self.http is None:
            raise SourceUnavailable("rotowire: no HTTP client configured")
        payload = self.http.get(URL, {"team": "ALL", "pos": "ALL"}, max_age_hours=self.ttl("injuries"),
                                validator=V.json_nonempty)
        df = parse_injury_table(payload)
        df["fetched_at"] = utc_now_iso()
        return df

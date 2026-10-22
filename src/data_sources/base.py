"""Data-source abstraction.

Every source implements a subset of the dataset methods below and advertises which ones in
`capabilities`.  The ingestion layer (src/ingest.py) walks the priority list configured in
`sources` (league_settings.json) and falls through to the next source when one raises
`SourceUnavailable` / `FetchError`, so any website can be swapped out without touching the model.

Canonical output schemas (lower-case column names) are documented in README.md.
"""
from __future__ import annotations

import pandas as pd


class SourceUnavailable(RuntimeError):
    """The source cannot provide this dataset (blocked, schema change, not implemented...)."""


class DataSource:
    name = "base"
    capabilities: frozenset = frozenset()

    def __init__(self, cfg, http=None):
        self.cfg = cfg
        self.http = http

    def supports(self, dataset: str) -> bool:
        return dataset in self.capabilities

    # ---- cache lifetime policy (hours; None = never expires) -------------------------------
    def ttl(self, dataset: str, year: int | None = None):
        ttl = self.cfg.get("scraping.ttl_hours", {})
        if dataset in ("game_logs", "team_game_logs", "starters"):
            return None if (year is not None and self.cfg.is_completed(year)) else ttl.get("current_season_games", 20)
        if dataset == "rosters":
            return None if (year is not None and self.cfg.is_completed(year)) else ttl.get("rosters", 12)
        key = {"current_rosters": "rosters"}.get(dataset, dataset)
        return ttl.get(key, 24)

    # ---- dataset API ------------------------------------------------------------------------
    def player_game_logs(self, year: int, date_from=None) -> pd.DataFrame:
        raise SourceUnavailable(f"{self.name}: player_game_logs not supported")

    def team_game_logs(self, year: int, date_from=None) -> pd.DataFrame:
        raise SourceUnavailable(f"{self.name}: team_game_logs not supported")

    def starters(self, year: int) -> pd.DataFrame:
        raise SourceUnavailable(f"{self.name}: starters not supported")

    def rosters(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise SourceUnavailable(f"{self.name}: rosters not supported")

    def current_rosters(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise SourceUnavailable(f"{self.name}: current_rosters not supported")

    def player_index(self) -> pd.DataFrame:
        raise SourceUnavailable(f"{self.name}: player_index not supported")

    def injuries(self) -> pd.DataFrame:
        raise SourceUnavailable(f"{self.name}: injuries not supported")

    def consensus(self, year: int) -> pd.DataFrame:
        raise SourceUnavailable(f"{self.name}: consensus not supported")

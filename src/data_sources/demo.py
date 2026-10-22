"""Offline synthetic league exposed through the normal DataSource interface (`--demo`)."""
from __future__ import annotations

import pandas as pd

from ..demo_data import DemoLeague, generate_demo_league
from .base import DataSource

_CACHE: dict[tuple, DemoLeague] = {}


def get_league(cfg) -> DemoLeague:
    key = (cfg.first_year, cfg.target_year, cfg.get("demo.seed", 7), cfg.get("demo.n_teams", 30),
           cfg.get("demo.n_rounds", 82), cfg.get("demo.partial_target_rounds", 0),
           cfg.get("demo.inject_issues", True))
    if key not in _CACHE:
        _CACHE[key] = generate_demo_league(
            first_year=cfg.first_year, target_year=cfg.target_year, n_teams=cfg.get("demo.n_teams", 30),
            n_rounds=cfg.get("demo.n_rounds", 82), seed=cfg.get("demo.seed", 7),
            partial_target_rounds=cfg.get("demo.partial_target_rounds", 0),
            inject_issues=cfg.get("demo.inject_issues", True), today=cfg.today)
    return _CACHE[key]


class DemoSource(DataSource):
    name = "demo"
    capabilities = frozenset({"game_logs", "team_game_logs", "starters", "rosters", "current_rosters",
                              "player_index", "injuries", "consensus"})

    @property
    def league(self) -> DemoLeague:
        return get_league(self.cfg)

    def player_game_logs(self, year: int, date_from=None) -> pd.DataFrame:
        d = self.league.player_games
        d = d[d["year"] == year]
        return d[d["game_date"] >= date_from] if date_from is not None else d.copy()

    def team_game_logs(self, year: int, date_from=None) -> pd.DataFrame:
        d = self.league.team_games
        d = d[d["year"] == year]
        return d[d["game_date"] >= date_from] if date_from is not None else d.copy()

    def starters(self, year: int) -> pd.DataFrame:
        s = self.league.starters
        return s[s["year"] == year][["year", "player_id", "gs"]].copy()

    def rosters(self, year: int):
        L = self.league
        return (L.rosters[L.rosters["year"] == year].copy(), L.coaches[L.coaches["year"] == year].copy())

    def current_rosters(self):
        L = self.league
        return L.current_rosters.copy(), L.current_coaches.copy()

    def player_index(self) -> pd.DataFrame:
        return self.league.player_index.copy()

    def injuries(self) -> pd.DataFrame:
        return self.league.injuries.copy()

    def consensus(self, year: int) -> pd.DataFrame:
        return self.league.consensus.copy()

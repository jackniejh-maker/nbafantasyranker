"""Configuration loading (config/league_settings.json) and season/path helpers."""
from __future__ import annotations

import copy
import json
import os
from datetime import date
from pathlib import Path

from .fantasy_scoring import STAT_KEYS
from .utils import stable_hash, year_from_label

DATASETS = ("game_logs", "team_game_logs", "starters", "rosters", "current_rosters",
            "player_index", "injuries", "consensus")


class Config:
    ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_PATH = ROOT / "config" / "league_settings.json"

    def __init__(self, data: dict, data_dir=None, output_dir=None):
        self.data = data
        self._data_dir = Path(data_dir) if data_dir else None
        self._out_dir = Path(output_dir) if output_dir else None
        self.demo = False
        self._validate()

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path=None, data_dir=None, output_dir=None) -> "Config":
        p = Path(path) if path else cls.DEFAULT_PATH
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cfg = cls(data, data_dir=data_dir, output_dir=output_dir)
        if data.get("source_profile") == "espn":          # permanent switch: set "source_profile": "espn" in the config
            cfg.use_espn_stats()
        return cfg

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def _validate(self) -> None:
        missing = [k for k in STAT_KEYS if k not in self.data.get("scoring", {})]
        if missing:
            raise ValueError(f"scoring block is missing stat keys: {missing}")
        w = self.get("weighting.recency_weights")
        if not w or any(x < 0 for x in w) or sum(w) <= 0:
            raise ValueError("weighting.recency_weights must be non-negative and not all zero")

    # ------------------------------------------------------------ access
    def get(self, dotted: str, default=None):
        cur = self.data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    def set(self, dotted: str, value) -> None:
        parts = dotted.split(".")
        cur = self.data
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def __getitem__(self, key):
        return self.data[key]

    @property
    def scoring(self) -> dict:
        return {k: float(self.data["scoring"][k]) for k in STAT_KEYS}

    @property
    def season_length(self) -> int:
        return int(self.get("seasons.season_length", 82))

    @property
    def history_window(self) -> int:
        return int(self.get("seasons.history_window", 3))

    @property
    def aliases(self) -> dict:
        return self.get("name_aliases", {}) or {}

    # ------------------------------------------------------------ seasons
    @property
    def today(self) -> date:
        raw = os.environ.get("FANTASY_TODAY") or self.get("today")
        return date.fromisoformat(raw) if raw else date.today()

    @property
    def target_year(self) -> int:
        t = self.get("seasons.target_season")
        if t not in (None, "", 0):
            return year_from_label(t)
        d = self.today
        return d.year if d.month >= int(self.get("seasons.rollover_month", 7)) else d.year - 1

    @property
    def last_completed_year(self) -> int:
        return self.target_year - 1

    @property
    def first_year(self) -> int:
        return int(self.get("seasons.first_season"))

    def history_years(self) -> list[int]:
        return list(range(self.first_year, self.last_completed_year + 1))

    def is_completed(self, year: int) -> bool:
        return int(year) <= self.last_completed_year

    # ------------------------------------------------------------ paths
    @property
    def data_dir(self) -> Path:
        return self._data_dir or (self.ROOT / "data")

    @property
    def outputs_dir(self) -> Path:
        return self._out_dir or (self.ROOT / "outputs")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def models_dir(self) -> Path:
        return self.processed_dir / "models"

    @property
    def plots_dir(self) -> Path:
        return self.outputs_dir / "plots"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "player_reports"

    def project_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.ROOT / p

    def make_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.cache_dir, self.external_dir,
                  self.models_dir, self.plots_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ ESPN profile
    def use_espn_stats(self) -> "Config":
        """Download everything from ESPN instead of stats.nba.com.  Player ids become ESPN athlete ids, so this profile
        keeps its own data folder (data_espn/) unless one is given; the ingestion layer refuses to mix id namespaces."""
        self.set("source_profile", "espn")
        for ds in ("game_logs", "team_game_logs", "starters", "rosters", "player_index"):
            self.set(f"sources.{ds}", ["espn_stats"])
        self.set("sources.current_rosters", ["espn_stats", "espn"])
        self._data_dir = self._data_dir or (self.ROOT / "data_espn")
        return self

    # ------------------------------------------------------------ demo mode
    def apply_demo(self, root=None) -> "Config":
        """Switch every source to the offline synthetic league and use separate folders."""
        self.demo = True
        base = Path(root) if root else self.ROOT / "demo_run"
        self._data_dir = self._data_dir or base / "data"
        self._out_dir = self._out_dir or base / "outputs"
        for ds in DATASETS:
            self.set(f"sources.{ds}", ["demo"])
        self.set("seasons.first_season", 2017)
        if self.get("seasons.target_season") in (None, "", 0):
            self.set("seasons.target_season", 2026)
        self.set("validation.n_test_seasons", 3)
        self.set("validation.min_train_seasons", 3)
        return self

    # ------------------------------------------------------------ fingerprints
    def model_fingerprint(self) -> str:
        keys = ("scoring", "seasons", "weighting", "eligibility", "validation", "models",
                "minutes_reconciliation", "simulation", "game_model")
        return stable_hash({k: self.data.get(k) for k in keys})

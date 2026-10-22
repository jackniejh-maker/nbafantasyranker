"""Data-source registry.

Add a new source by writing a `DataSource` subclass and registering it in `_REGISTRY`.
`stathead.py` is intentionally absent: Stathead is a paid, login-gated service whose terms do not
permit automated access, so it is not an "actually accessible" source.
"""
from __future__ import annotations

from ..scraping.cache import DiskCache
from ..scraping.scraper import HttpClient
from .base import DataSource, SourceUnavailable

_REGISTRY = {
    "nba": ("src.data_sources.nba", "NBAStatsSource"),
    "espn": ("src.data_sources.espn", "ESPNSource"),
    "espn_stats": ("src.data_sources.espn_stats", "ESPNStatsSource"),
    "rotowire": ("src.data_sources.rotowire", "RotowireSource"),
    "basketball_reference": ("src.data_sources.basketball_reference", "BasketballReferenceSource"),
    "demo": ("src.data_sources.demo", "DemoSource"),
}


def make_http(name: str, cfg, offline: bool = False) -> HttpClient:
    sc = cfg.get("scraping", {})
    return HttpClient(
        name, DiskCache(cfg.cache_dir / "http"),
        user_agent=sc.get("user_agent", "Mozilla/5.0"),
        min_interval=sc.get("min_interval_seconds", {}).get(name, 1.5),
        timeout=sc.get("timeout_seconds", 45), max_retries=sc.get("max_retries", 4),
        backoff=sc.get("backoff_seconds", 2.0), offline=offline)


def build_source(name: str, cfg, offline: bool = False) -> DataSource:
    if name not in _REGISTRY:
        raise KeyError(f"unknown data source {name!r}; known: {sorted(_REGISTRY)}")
    import importlib
    mod_name, cls_name = _REGISTRY[name]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    http = None if name == "demo" else make_http(name, cfg, offline)
    return cls(cfg, http)


__all__ = ["DataSource", "SourceUnavailable", "build_source", "make_http"]

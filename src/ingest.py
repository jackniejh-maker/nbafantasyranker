"""Data ingestion: decides what to (re)download, walks the configured source priority list,
stores canonical raw tables in data/raw/ and records provenance in data/raw/manifest.json.

Freshness policy
----------------
* completed seasons (game logs, team logs, starts, rosters/coaches)      -> downloaded ONCE
* the target season's game logs                                          -> INCREMENTAL (DateFrom), TTL 20h
* current rosters / player index / injuries / consensus                  -> TTL (config scraping.ttl_hours)
* --update-data refreshes the current-context datasets regardless of TTL;
  --refresh-history re-downloads completed seasons as well.
A user-supplied CSV in data/external/<dataset>/<year>.csv (canonical schema) is accepted for any
season the sources cannot deliver.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from .data_sources import build_source
from .data_sources.base import SourceUnavailable
from .scraping.scraper import FetchError, network_reachable
from .utils import df_hash, load_table, log, save_table, utc_now_iso

PLAYER_KEYS = ["game_id", "player_id"]
TEAM_KEYS = ["game_id", "team"]


class IngestError(RuntimeError):
    pass


@dataclass
class RawData:
    player_games: pd.DataFrame
    team_games: pd.DataFrame | None = None
    starters: pd.DataFrame | None = None
    rosters: pd.DataFrame | None = None
    coaches: pd.DataFrame | None = None
    current_rosters: pd.DataFrame | None = None
    current_coaches: pd.DataFrame | None = None
    player_index: pd.DataFrame | None = None
    injuries: pd.DataFrame | None = None
    consensus: pd.DataFrame | None = None
    manifest: dict = field(default_factory=dict)
    missing: dict = field(default_factory=dict)


class Ingestor:
    def __init__(self, cfg, offline: bool = False, force_current: bool = False, force_history: bool = False):
        self.cfg = cfg
        if not offline and not getattr(cfg, "demo", False) and not network_reachable():
            log.warning("no network connection detected - running in OFFLINE mode (cached data only). "
                        "Use `--demo` for the synthetic league, or reconnect and rerun.")
            offline = True
        self.offline = offline
        self.force_current = force_current
        self.force_history = force_history
        self._sources: dict = {}
        self.manifest_path = cfg.raw_dir / "manifest.json"
        self.manifest: dict = {}
        if self.manifest_path.exists():
            try:
                self.manifest = json.loads(self.manifest_path.read_text())
            except Exception:
                log.warning("manifest unreadable - starting a fresh one")
        self.missing: dict[str, list] = {}
        self._check_profile()

    def _check_profile(self) -> None:
        """Raw tables from different source profiles use different player-id namespaces (NBA ids vs ESPN ids)."""
        if getattr(self.cfg, "demo", False):
            return
        profile = self.cfg.get("source_profile", "nba")
        p = self.cfg.raw_dir / "_profile.json"
        have = json.loads(p.read_text()).get("profile") if p.exists() else ("nba" if (self.cfg.raw_dir / "game_logs").exists() else None)
        if have is not None and have != profile:
            raise IngestError(f"{self.cfg.raw_dir} holds data from the '{have}' source profile but this run uses '{profile}'. "
                              "Player ids differ between profiles, so they must not be mixed: use a separate --data-dir "
                              f"(--source espn defaults to data_espn/) or delete {self.cfg.raw_dir} and {self.cfg.processed_dir}.")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"profile": profile}))

    # ------------------------------------------------------------------ plumbing
    def _src(self, name: str):
        if name not in self._sources:
            self._sources[name] = build_source(name, self.cfg, offline=self.offline)
        return self._sources[name]

    def _write_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1, default=str))

    def _save(self, dataset: str, name: str, df: pd.DataFrame, source: str) -> None:
        save_table(df, self.cfg.raw_dir / dataset / name)
        self.manifest[f"{dataset}/{name}"] = {"source": source, "fetched_at": utc_now_iso(),
                                              "rows": int(len(df)), "sha256": df_hash(df)}
        self._write_manifest()

    def _load(self, dataset: str, name: str):
        return load_table(self.cfg.raw_dir / dataset / name)

    def _fresh(self, dataset: str, name: str, ttl_hours) -> bool:
        ent = self.manifest.get(f"{dataset}/{name}")
        if ent is None or self._load(dataset, name) is None:
            return False
        if ttl_hours is None:
            return True
        try:
            t = datetime.strptime(ent["fetched_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            return False
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0 <= ttl_hours

    def _external(self, dataset: str, name: str):
        p = self.cfg.external_dir / dataset / f"{name}.csv"
        if p.exists():
            log.info("using user-supplied %s", p)
            head = pd.read_csv(p, nrows=0).columns
            return pd.read_csv(p, parse_dates=[c for c in ("game_date", "birth_date") if c in head])
        return None

    def _fetch(self, dataset: str, method: str, *args, force: bool = False, **kw):
        errors = []
        for sname in self.cfg.get(f"sources.{dataset}", []):
            try:
                src = self._src(sname)
                if not src.supports(dataset):
                    continue
                if src.http is not None:
                    src.http.force_all = force
                try:
                    return getattr(src, method)(*args, **kw), sname
                finally:
                    if src.http is not None:
                        src.http.force_all = False
            except Exception as exc:            # any source problem (network, schema, parsing) -> try the next source
                errors.append(f"{sname}: {type(exc).__name__}: {exc}")
                log.warning("source %s failed for %s: %s: %s", sname, dataset, type(exc).__name__, exc)
        raise IngestError(f"{dataset}: all sources failed -> {errors}")

    # ------------------------------------------------------------------ historical datasets
    def _ensure_year(self, dataset: str, method: str, year: int, required: bool):
        name = str(year)
        if not self.force_history and self._fresh(dataset, name, None):
            return self._load(dataset, name)
        ext = self._external(dataset, name)
        if ext is not None:
            self._save(dataset, name, ext, "external_csv")
            return ext
        try:
            df, sname = self._fetch(dataset, method, year, force=self.force_history)
        except IngestError as exc:
            self.missing.setdefault(dataset, []).append(year)
            old = self._load(dataset, name)
            if old is not None:
                log.warning("%s %s: refresh failed, keeping previous raw copy", dataset, year)
                return old
            (log.error if required else log.warning)("%s %s unavailable: %s", dataset, year, exc)
            return None
        self._save(dataset, name, df, sname)
        log.info("downloaded %-14s %s  rows=%-7d source=%s", dataset, year, len(df), sname)
        return df

    def _ensure_rosters(self, year: int):
        """Roster + coaches arrive together as a tuple and are stored as two raw tables."""
        if not self.force_history and self._fresh("rosters", str(year), None):
            return self._load("rosters", str(year)), self._load("coaches", str(year))
        ext_r, ext_c = self._external("rosters", str(year)), self._external("coaches", str(year))
        if ext_r is not None:
            self._save("rosters", str(year), ext_r, "external_csv")
            if ext_c is not None:
                self._save("coaches", str(year), ext_c, "external_csv")
            return ext_r, ext_c
        try:
            (ros, coa), sname = self._fetch("rosters", "rosters", year, force=self.force_history)
        except IngestError as exc:
            self.missing.setdefault("rosters", []).append(year)
            log.warning("rosters %s unavailable: %s", year, exc)
            return self._load("rosters", str(year)), self._load("coaches", str(year))
        self._save("rosters", str(year), ros, sname)
        self._save("coaches", str(year), coa, sname)
        log.info("downloaded rosters        %s  rows=%d coaches=%d", year, len(ros), len(coa))
        return ros, coa

    # ------------------------------------------------------------------ current / in-progress
    def _update_current_games(self, dataset: str, method: str, keys: list[str], year: int):
        ttl = self.cfg.get("scraping.ttl_hours.current_season_games", 20)
        existing = self._load(dataset, str(year))
        if existing is not None and not self.force_current and self._fresh(dataset, str(year), ttl):
            return existing
        if self.offline:
            return existing
        date_from = None
        if existing is not None and len(existing):
            date_from = pd.Timestamp(existing["game_date"].max()) - pd.Timedelta(days=1)
        try:
            new, sname = self._fetch(dataset, method, year, force=self.force_current, date_from=date_from)
        except IngestError as exc:
            log.warning("current-season %s refresh failed: %s", dataset, exc)
            return existing
        if new is None or len(new) == 0:
            if existing is not None:
                self._save(dataset, str(year), existing, self.manifest.get(f"{dataset}/{year}", {}).get("source", "?"))
            return existing
        combined = pd.concat([existing, new], ignore_index=True) if existing is not None else new
        combined = combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
        self._save(dataset, str(year), combined, sname)
        log.info("updated current-season %-14s +%d new rows (total %d)", dataset, len(new), len(combined))
        return combined

    def _ensure_current(self, dataset: str, method: str, name: str, ttl_key: str, *args, tuple_result=False):
        ttl = self.cfg.get(f"scraping.ttl_hours.{ttl_key}", 24)
        if not self.force_current and self._fresh(dataset, name, ttl):
            return self._load(dataset, name)
        if self.offline and self._load(dataset, name) is not None:
            return self._load(dataset, name)
        try:
            res, sname = self._fetch(dataset, method, *args, force=self.force_current)
        except IngestError as exc:
            log.warning("%s refresh failed (%s); using last saved copy if any", dataset, exc)
            return self._load(dataset, name)
        self._save(dataset, name, res, sname)
        return res

    def _ensure_current_rosters(self):
        ttl = self.cfg.get("scraping.ttl_hours.rosters", 12)
        have = self._load("current_rosters", "rosters") is not None
        if have and not self.force_current and (self.offline or self._fresh("current_rosters", "rosters", ttl)):
            return self._load("current_rosters", "rosters"), self._load("current_rosters", "coaches")
        try:
            (ros, coa), sname = self._fetch("current_rosters", "current_rosters", force=self.force_current)
        except IngestError as exc:
            log.warning("current rosters refresh failed (%s); using last saved copy if any", exc)
            return self._load("current_rosters", "rosters"), self._load("current_rosters", "coaches")
        self._save("current_rosters", "rosters", ros, sname)
        self._save("current_rosters", "coaches", coa, sname)
        log.info("current rosters: %d players on %d teams (source=%s)", len(ros), ros["team"].nunique(), sname)
        return ros, coa

    def _ensure_injuries(self) -> pd.DataFrame | None:
        ttl = self.cfg.get("scraping.ttl_hours.injuries", 3)
        if not self.force_current and self._fresh("injuries", "current", ttl):
            return self._load("injuries", "current")
        frames, used = [], []
        for sname in self.cfg.get("sources.injuries", []):
            try:
                src = self._src(sname)
                if not src.supports("injuries"):
                    continue
                if src.http is not None:
                    src.http.force_all = self.force_current
                frames.append(src.injuries())
                used.append(sname)
            except (SourceUnavailable, FetchError, KeyError, ValueError) as exc:
                log.warning("injuries from %s failed: %s", sname, exc)
            finally:
                if self._sources.get(sname) is not None and self._sources[sname].http is not None:
                    self._sources[sname].http.force_all = False
        if not frames:
            return self._load("injuries", "current")
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["player_name"], keep="first").reset_index(drop=True)  # first source wins
        self._save("injuries", "current", df, "+".join(used))
        return df

    # ------------------------------------------------------------------ entry point
    def run(self) -> RawData:
        cfg = self.cfg
        cfg.make_dirs()
        pgs, tgs, sts, ros, coa = [], [], [], [], []
        for y in cfg.history_years():
            d = self._ensure_year("game_logs", "player_game_logs", y, required=True)
            if d is not None and len(d):
                pgs.append(d)
            t = self._ensure_year("team_game_logs", "team_game_logs", y, required=False)
            if t is not None and len(t):
                tgs.append(t)
            s = self._ensure_year("starters", "starters", y, required=False)
            if s is not None and len(s):
                sts.append(s)
            r, c = self._ensure_rosters(y)
            if r is not None and len(r):
                ros.append(r)
            if c is not None and len(c):
                coa.append(c)
        if len(pgs) < 3:
            raise IngestError(
                f"only {len(pgs)} historical season(s) of player game logs available "
                f"(missing: {self.missing.get('game_logs')}). Check your internet connection / source settings, "
                "or drop canonical CSVs into data/external/game_logs/<year>.csv. "
                "Use `python project.py --demo` to run on the offline synthetic league.")
        if cfg.last_completed_year in self.missing.get("game_logs", []):
            raise IngestError(
                f"the most recent completed season ({cfg.last_completed_year}) could not be downloaded - refusing to project "
                "from stale data. Check your connection / stats.nba.com access (see `python check_sources.py`) and rerun.")
        ty = cfg.target_year
        cur_pg = self._update_current_games("game_logs", "player_game_logs", PLAYER_KEYS, ty)
        cur_tg = self._update_current_games("team_game_logs", "team_game_logs", TEAM_KEYS, ty)
        if cur_pg is not None and len(cur_pg):
            pgs.append(cur_pg)
        if cur_tg is not None and len(cur_tg):
            tgs.append(cur_tg)
        cur_st = self._ensure_current("starters", "starters", str(ty), "current_season_games", ty) \
            if cur_pg is not None and len(cur_pg) else None
        if cur_st is not None and len(cur_st):
            sts.append(cur_st)
        cur_ros, cur_coa = self._ensure_current_rosters()
        pidx = self._ensure_current("player_index", "player_index", "all", "player_index")
        inj = self._ensure_injuries() if cfg.get("injuries.enabled", True) else None
        cons = self._ensure_current("consensus", "consensus", str(ty), "consensus", ty)
        nets = sum(len([r for r in s.http.requests_log if not r["from_cache"]]) for s in self._sources.values()
                   if s.http is not None)
        hits = sum(len([r for r in s.http.requests_log if r["from_cache"]]) for s in self._sources.values()
                   if s.http is not None)
        log.info("ingestion done: %d network responses, %d served from cache", nets, hits)
        return RawData(
            player_games=pd.concat(pgs, ignore_index=True),
            team_games=pd.concat(tgs, ignore_index=True) if tgs else None,
            starters=pd.concat(sts, ignore_index=True) if sts else None,
            rosters=pd.concat(ros, ignore_index=True) if ros else None,
            coaches=pd.concat(coa, ignore_index=True) if coa else None,
            current_rosters=cur_ros, current_coaches=cur_coa, player_index=pidx, injuries=inj, consensus=cons,
            manifest=dict(self.manifest), missing=dict(self.missing))

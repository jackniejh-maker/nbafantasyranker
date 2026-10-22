"""Rate-limited, caching, retrying HTTP client shared by every data source.

* one throttle per *source* (shared across instances) so we never exceed a polite request rate
* exponential backoff + jitter, honouring `Retry-After`
* responses are validated BEFORE they are cached, so block pages never poison the cache
* stale cache is used as a last resort (and in offline mode) with a loud warning
* every request is recorded in `requests_log` (url, params, cache hit, timestamp, checksum)
"""
from __future__ import annotations

import json
import random
import threading
import time

import requests

from ..utils import log
from .cache import DiskCache
from .validators import ValidationError

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """Network/HTTP/validation failure after all retries."""


class OfflineError(FetchError):
    """Raised in offline mode when a response is not in the cache."""


class HttpClient:
    _last_call: dict[str, float] = {}  # per-source throttle, shared by all instances
    _locks: dict[str, threading.Lock] = {}

    def __init__(self, source: str, cache: DiskCache, *, user_agent: str, min_interval: float = 1.0,
                 timeout: float = 30.0, max_retries: int = 3, backoff: float = 2.0,
                 headers: dict | None = None, offline: bool = False, session=None,
                 sleep=time.sleep, clock=time.monotonic):
        self.source = source
        self.cache = cache
        self.min_interval = float(min_interval)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.offline = offline
        self.headers = {"User-Agent": user_agent, "Accept-Language": "en-US,en;q=0.9"}
        if headers:
            self.headers.update(headers)
        self._session = session
        self._local = threading.local()          # one requests.Session per worker thread
        self._sleep = sleep
        self._clock = clock
        self.requests_log: list[dict] = []
        self.force_all = False          # set by the ingestion layer for --update-data / --refresh-history

    # ------------------------------------------------------------------ helpers
    @property
    def session(self):
        if self._session is not None:            # injected (tests)
            return self._session
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._local.session = requests.Session()
        return s

    def _throttle(self) -> None:
        """Global per-source rate limit that is safe with several worker threads (request STARTS are serialised)."""
        with self._locks.setdefault(self.source, threading.Lock()):
            last = self._last_call.get(self.source)
            if last is not None:
                wait = self.min_interval - (self._clock() - last)
                if wait > 0:
                    self._sleep(wait)
            self._last_call[self.source] = self._clock()

    def _backoff(self, attempt: int, retry_after=None) -> None:
        wait = None
        if retry_after:
            try:
                wait = min(float(retry_after), 60.0)
            except ValueError:
                wait = None
        if wait is None:
            wait = self.backoff * (2 ** (attempt - 1)) * (1 + 0.25 * random.random())
        if attempt < self.max_retries:
            self._sleep(wait)

    @staticmethod
    def _decode(content: bytes, expect: str, validator):
        if expect == "json":
            payload = json.loads(content.decode("utf-8"))
        elif expect == "text":
            payload = content.decode("utf-8", errors="replace")
        else:
            payload = content
        if validator is not None:
            validator(payload)
        return payload

    def _record(self, url, params, meta, from_cache: bool) -> None:
        self.requests_log.append({"source": self.source, "url": url, "params": params,
                                  "from_cache": from_cache, "fetched_at": meta.get("fetched_at"),
                                  "sha256": meta.get("sha256")})

    # ------------------------------------------------------------------ public API
    def get(self, url: str, params: dict | None = None, *, max_age_hours: float | None = None,
            validator=None, expect: str = "json", force: bool = False, headers: dict | None = None, transform=None):
        """GET with cache/retry/validation.

        `transform(payload) -> JSON-serialisable` is applied to the validated response BEFORE caching, so only a
        compact extract is stored (e.g. one parsed box score instead of a 300 KB play-by-play document).  Cached entries
        are therefore already transformed; a transform that raises marks the response invalid (retried, never cached)."""
        key = DiskCache.make_key(url, params)
        force = force or self.force_all
        read_validator = None if transform is not None else validator
        if not force:
            entry = self.cache.get(self.source, key, max_age_hours=max_age_hours)
            if entry is not None:
                try:
                    payload = self._decode(entry.content, expect, read_validator)
                    self._record(url, params, entry.meta, True)
                    return payload
                except (ValueError, ValidationError) as exc:
                    log.warning("cached response for %s failed validation (%s); refetching", url, exc)
                    self.cache.delete(self.source, key)

        if self.offline:
            stale = self.cache.get(self.source, key, max_age_hours=None)
            if stale is not None:
                log.warning("OFFLINE: using cached copy of %s (fetched %s)", url, stale.meta.get("fetched_at"))
                return self._decode(stale.content, expect, read_validator)
            raise OfflineError(f"offline mode and no cached copy of {url}")

        last_exc: Exception | None = None
        req_headers = {**self.headers, **(headers or {})}
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, headers=req_headers, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("[%s] attempt %d/%d failed: %s: %s", self.source, attempt, self.max_retries,
                            type(exc).__name__, exc)
                self._backoff(attempt)
                continue
            if resp.status_code == 200:
                try:
                    payload = self._decode(resp.content, expect, validator)
                    content = resp.content
                    if transform is not None:
                        payload = transform(payload)
                        content = json.dumps(payload, default=str).encode("utf-8")
                except (ValueError, ValidationError, KeyError, TypeError, AttributeError, IndexError) as exc:
                    last_exc = FetchError(f"invalid response from {url}: {type(exc).__name__}: {exc}")
                    log.warning("[%s] attempt %d/%d invalid payload: %s: %s", self.source, attempt, self.max_retries,
                                type(exc).__name__, exc)
                    self._backoff(attempt)
                    continue
                meta = self.cache.put(self.source, key, content, url=url, params=params,
                                      source=self.source, status=200)
                self._record(url, params, meta, False)
                return payload
            last_exc = FetchError(f"HTTP {resp.status_code} from {url}")
            log.warning("[%s] attempt %d/%d -> HTTP %s", self.source, attempt, self.max_retries, resp.status_code)
            if resp.status_code in RETRY_STATUS:
                self._backoff(attempt, resp.headers.get("Retry-After"))
                continue
            break  # 401/403/404...: retrying will not help

        stale = self.cache.get(self.source, key, max_age_hours=None)
        if stale is not None:
            try:
                payload = self._decode(stale.content, expect, read_validator)
                log.warning("[%s] request failed; using STALE cached copy of %s (fetched %s)",
                            self.source, url, stale.meta.get("fetched_at"))
                return payload
            except (ValueError, ValidationError):
                pass
        raise FetchError(f"giving up on {url}: {type(last_exc).__name__}: {last_exc}") from last_exc


def network_reachable(hosts=("stats.nba.com", "site.api.espn.com"), timeout: float = 5.0) -> bool:
    """Fast TCP probe so a missing connection is detected in seconds, not after dozens of retried downloads."""
    import socket
    for h in hosts:
        try:
            with socket.create_connection((h, 443), timeout=timeout):
                return True
        except OSError:
            continue
    return False

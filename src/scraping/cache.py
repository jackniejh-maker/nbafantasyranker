"""On-disk HTTP response cache.

Each response is stored as `<namespace>/<key>.bin` (raw bytes) plus `<key>.json` metadata
(source URL, params, fetch timestamp, sha256).  Reads verify the checksum, so a truncated or
tampered file is treated as a cache miss.  `max_age_hours=None` means "never expires"
(used for completed historical seasons); `0` forces a refetch.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from ..utils import log, utc_now_iso


class CacheEntry(NamedTuple):
    content: bytes
    meta: dict


class DiskCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(url: str, params: dict | None = None) -> str:
        payload = json.dumps({"u": url, "p": params or {}}, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _paths(self, namespace: str, key: str) -> tuple[Path, Path]:
        d = self.root / namespace
        return d / f"{key}.bin", d / f"{key}.json"

    @staticmethod
    def age_hours(meta: dict) -> float | None:
        ts = meta.get("fetched_at")
        if not ts:
            return None
        try:
            t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0

    def get(self, namespace: str, key: str, max_age_hours: float | None = None,
            verify: bool = True) -> CacheEntry | None:
        data_p, meta_p = self._paths(namespace, key)
        if not (data_p.exists() and meta_p.exists()):
            return None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            content = data_p.read_bytes()
        except Exception as exc:
            log.warning("cache read error (%s): %s", data_p.name, exc)
            return None
        if verify and meta.get("sha256") and hashlib.sha256(content).hexdigest() != meta["sha256"]:
            log.warning("cache checksum mismatch for %s/%s - ignoring entry", namespace, key[:10])
            return None
        if max_age_hours is not None:
            age = self.age_hours(meta)
            if age is None or age > max_age_hours:
                return None
        return CacheEntry(content, meta)

    def put(self, namespace: str, key: str, content: bytes, **meta) -> dict:
        data_p, meta_p = self._paths(namespace, key)
        data_p.parent.mkdir(parents=True, exist_ok=True)
        record = dict(meta)
        record.update({"fetched_at": utc_now_iso(), "n_bytes": len(content),
                       "sha256": hashlib.sha256(content).hexdigest()})
        for path, blob in ((data_p, content), (meta_p, json.dumps(record, default=str).encode())):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, path)
        return record

    def delete(self, namespace: str, key: str) -> None:
        for p in self._paths(namespace, key):
            p.unlink(missing_ok=True)

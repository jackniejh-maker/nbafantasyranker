"""Shared helpers: season labels, name normalisation, table IO, hashing, logging."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("fantasy_ranker")


# ------------------------------------------------------------------ logging
def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Console + optional file logging (idempotent)."""
    root = logging.getLogger("fantasy_ranker")
    root.setLevel(level)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ seasons
def season_label(year: int) -> str:
    """2023 -> '2023-24'.  `year` is always the season START year."""
    return f"{int(year)}-{(int(year) + 1) % 100:02d}"


def year_from_label(label) -> int:
    return int(str(label).strip()[:4])


def season_ref_date(year: int) -> date:
    """Reference date for ages: Feb 1 of the season's second calendar year."""
    return date(int(year) + 1, 2, 1)


# ------------------------------------------------------------------ names
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# Common nickname/legal-name mismatches between data sources.  Extend through the
# "name_aliases" block of config/league_settings.json (variant -> canonical).
DEFAULT_ALIASES = {
    "moe wagner": "moritz wagner",
    "alex sarr": "alexandre sarr",
    "ron holland": "ronald holland",
    "kj martin": "kenyon martin",
    "bub carrington": "carlton carrington",
    "herb jones": "herbert jones",
}


def normalize_name(name, aliases: dict | None = None) -> str:
    """Lower-case ASCII name without punctuation / generational suffixes."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    out = " ".join(t for t in s.split() if t not in _SUFFIXES)
    table = DEFAULT_ALIASES if aliases is None else {**DEFAULT_ALIASES, **aliases}
    return table.get(out, out)


def slugify(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_name(name)).strip("_") or "player"


def parse_height_inches(value) -> float:
    """'6-10' / "6'10\"" / '82' -> inches (NaN when unparseable)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    m = re.match(r"^(\d)\s*[-'’]\s*(\d{1,2})", s)
    if m:
        return float(int(m.group(1)) * 12 + int(m.group(2)))
    try:
        v = float(s)
        return v if v > 20 else np.nan
    except ValueError:
        return np.nan


def position_group(pos) -> str:
    """Primary position group G/F/C from strings such as 'G-F', 'PF', 'C-F'."""
    if pos is None or (isinstance(pos, float) and np.isnan(pos)) or str(pos).strip() == "":
        return ""
    p = str(pos).upper().strip()
    first = re.split(r"[-/ ]", p)[0]
    if first in ("PG", "SG", "G"):
        return "G"
    if first in ("SF", "PF", "F", "GF"):
        return "F"
    if first == "C":
        return "C"
    return ""


# ------------------------------------------------------------------ table IO
def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def save_table(df: pd.DataFrame, stem) -> Path:
    """Persist a DataFrame (parquet when pyarrow is installed, pickle otherwise)."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pq, pk = Path(f"{stem}.parquet"), Path(f"{stem}.pkl")
    if _has_pyarrow():
        try:
            df.to_parquet(pq)
            pk.unlink(missing_ok=True)
            return pq
        except Exception as exc:  # pragma: no cover - parquet edge cases
            log.warning("parquet write failed for %s (%s); falling back to pickle", stem.name, exc)
    df.to_pickle(pk)
    return pk


def load_table(stem) -> pd.DataFrame | None:
    for ext, reader in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
        p = Path(f"{stem}{ext}")
        if p.exists():
            try:
                return reader(p)
            except Exception as exc:
                log.warning("could not read %s (%s); treating as missing", p, exc)
    return None


def table_exists(stem) -> bool:
    return Path(f"{stem}.parquet").exists() or Path(f"{stem}.pkl").exists()


# ------------------------------------------------------------------ hashing
def stable_hash(obj, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def df_hash(df: pd.DataFrame | None, n: int = 16) -> str:
    if df is None or len(df) == 0:
        return "empty"
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:n]


# ------------------------------------------------------------------ numerics
def safe_div(a, b, fill=np.nan):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((b != 0) & np.isfinite(b), a / b, fill)
    return out


def pct_rank(x) -> np.ndarray:
    """Percentile rank in [0, 100] (average ranks for ties)."""
    s = pd.Series(np.asarray(x, dtype=float))
    return (s.rank(pct=True, method="average") * 100.0).to_numpy()


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / sd

"""Data-quality system.

Nothing is dropped or altered silently: every removed or repaired row is appended to
`dropped_rows.csv` / `transformed_rows.csv` with the reason, and every check outcome is listed in
`data_quality_report.md` / `.json` (written to outputs/data_quality/).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import log, normalize_name


class DataQuality:
    def __init__(self):
        self.issues: list[dict] = []
        self._drops: list[pd.DataFrame] = []
        self._xforms: list[pd.DataFrame] = []

    # ------------------------------------------------------------------ recording
    def issue(self, check: str, severity: str, count: int, detail: str = "", examples=None) -> None:
        self.issues.append({"check": check, "severity": severity, "count": int(count), "detail": detail,
                            "examples": "; ".join(str(e) for e in list(examples or [])[:5])})

    @staticmethod
    def _rows(table, reason, rows, id_cols, detail) -> pd.DataFrame | None:
        if rows is None or len(rows) == 0:
            return None
        cols = [c for c in id_cols if c in rows.columns]
        r = rows[cols].copy()
        r.insert(0, "detail", detail)
        r.insert(0, "reason", reason)
        r.insert(0, "table", table)
        return r

    @staticmethod
    def _examples(r: pd.DataFrame) -> list[str]:
        body = r.drop(columns=["table", "reason", "detail"]).head(3)
        return ["/".join(f"{v}" for v in row) for row in body.itertuples(index=False)]

    def drop(self, table: str, reason: str, rows: pd.DataFrame, id_cols: list[str], detail: str = "") -> int:
        r = self._rows(table, reason, rows, id_cols, detail)
        if r is None:
            return 0
        self._drops.append(r)
        ex = self._examples(r)
        self.issue(f"{table}: rows dropped", "warning", len(r), reason, ex)
        return len(r)

    def transform(self, table: str, reason: str, rows: pd.DataFrame, id_cols: list[str], detail: str = "") -> int:
        r = self._rows(table, reason, rows, id_cols, detail)
        if r is None:
            return 0
        self._xforms.append(r)
        ex = self._examples(r)
        self.issue(f"{table}: rows repaired", "warning", len(r), reason, ex)
        return len(r)

    # ------------------------------------------------------------------ output
    def frames(self):
        iss = pd.DataFrame(self.issues, columns=["check", "severity", "count", "detail", "examples"])
        drops = pd.concat(self._drops, ignore_index=True) if self._drops else pd.DataFrame(
            columns=["table", "reason", "detail"])
        xf = pd.concat(self._xforms, ignore_index=True) if self._xforms else pd.DataFrame(
            columns=["table", "reason", "detail"])
        return iss, drops, xf

    def markdown(self) -> str:
        iss, drops, xf = self.frames()
        order = {"error": 0, "warning": 1, "info": 2}
        iss = iss.sort_values("severity", key=lambda s: s.map(order), kind="stable")
        lines = ["# Data-quality report", "",
                 f"* checks / events recorded: **{len(iss)}**",
                 f"* rows dropped: **{len(drops)}** (see `dropped_rows.csv`)",
                 f"* rows repaired: **{len(xf)}** (see `transformed_rows.csv`)", "",
                 "| severity | check | count | detail | examples |", "|---|---|---:|---|---|"]
        for _, r in iss.iterrows():
            lines.append(f"| {r['severity']} | {r['check']} | {r['count']} | {str(r['detail'])[:110]} | "
                         f"{str(r['examples'])[:110]} |")
        if len(drops):
            lines += ["", "## Dropped rows by reason", ""]
            lines += [f"* {t} / {r}: {n}" for (t, r), n in drops.groupby(["table", "reason"]).size().items()]
        if len(xf):
            lines += ["", "## Repaired rows by reason", ""]
            lines += [f"* {t} / {r}: {n}" for (t, r), n in xf.groupby(["table", "reason"]).size().items()]
        return "\n".join(lines) + "\n"

    def write(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        iss, drops, xf = self.frames()
        (out / "data_quality_report.md").write_text(self.markdown(), encoding="utf-8")
        (out / "data_quality_report.json").write_text(json.dumps(self.issues, indent=1, default=str))
        drops.to_csv(out / "dropped_rows.csv", index=False)
        xf.to_csv(out / "transformed_rows.csv", index=False)
        n_warn = int((iss["severity"] == "warning").sum()) if len(iss) else 0
        n_err = int((iss["severity"] == "error").sum()) if len(iss) else 0
        log.info("data-quality report: %d warnings, %d errors, %d rows dropped, %d repaired -> %s",
                 n_warn, n_err, len(drops), len(xf), out)
        return out


# ====================================================================== checks on processed data
def check_identity(pg: pd.DataFrame, dq: DataQuality, aliases: dict | None = None) -> None:
    """Inconsistent ids / duplicate names."""
    ids = pg[["player_id", "player_name"]].drop_duplicates()
    multi = ids.groupby("player_id")["player_name"].nunique()
    multi = multi[multi > 1]
    dq.issue("one player_id with several names", "warning" if len(multi) else "info", len(multi),
             "name spellings differ across rows", [f"{i}: {ids[ids.player_id == i].player_name.tolist()}" for i in multi.index[:3]])
    ids = ids.assign(key=ids["player_name"].map(lambda n: normalize_name(n, aliases)))
    dup = ids.groupby("key")["player_id"].nunique()
    dup = dup[dup > 1]
    dq.issue("duplicate player names (different player_id)", "warning" if len(dup) else "info", len(dup),
             "matching by name is ambiguous for these; ids are used everywhere in the model", list(dup.index[:5]))


def check_coverage(pg: pd.DataFrame, tg: pd.DataFrame | None, cfg, dq: DataQuality) -> None:
    years = sorted(pg["year"].unique())
    expected = cfg.history_years()
    missing = [y for y in expected if y not in years]
    dq.issue("missing seasons", "error" if missing else "info", len(missing),
             f"expected {expected[0]}..{expected[-1]}", missing)
    per_year = pg.groupby("year")["game_id"].nunique()
    typical = per_year[per_year.index.isin([y for y in years if cfg.is_completed(y)])]
    if len(typical):
        ref = typical.median()
        short = typical[typical < 0.85 * ref]
        dq.issue("seasons with unusually few games", "warning" if len(short) else "info", len(short),
                 f"median {ref:.0f} games per season", [f"{y}: {n}" for y, n in short.items()])
    if tg is not None and len(tg):
        n_teams = tg.groupby("year")["team"].nunique()
        dq.issue("seasons with fewer than expected teams", "info", int((n_teams < n_teams.max()).sum()),
                 f"max teams per season {n_teams.max()}", [f"{y}: {n}" for y, n in n_teams[n_teams < n_teams.max()].items()])


def check_minutes_consistency(pg: pd.DataFrame, dq: DataQuality, tol: float = 0.04) -> None:
    """Sum of player minutes per team-game should be ~240 (+25 per overtime)."""
    s = pg.groupby(["game_id", "team"])["min"].sum()
    resid = (s - 240) % 25
    resid = np.minimum(resid, 25 - resid)
    bad = s[(resid > 240 * tol)]
    dq.issue("team-games whose player minutes do not add up to 240+5k*OT", "warning" if len(bad) else "info",
             len(bad), f"tolerance {tol:.0%} of 240", [f"{i[0]}/{i[1]}: {v:.0f}" for i, v in bad.head(3).items()])


def check_player_seasons(ps: pd.DataFrame, stints: pd.DataFrame, dq: DataQuality, season_length: int) -> None:
    imp = ps[ps["gp"] > ps["team_games"] + 0.5]
    dq.issue("impossible games played (GP > team games)", "error" if len(imp) else "info", len(imp),
             "", [f"{r.player_id}/{r.year}: {r.gp}>{r.team_games}" for r in imp.head(3).itertuples()])
    over = ps[ps["gp"] > season_length]
    dq.issue("GP greater than the regular-season length", "error" if len(over) else "info", len(over))
    badpct = ps[(ps["fgm"] > ps["fga"]) | (ps["ftm"] > ps["fta"]) | (ps["fg3m"] > ps["fg3a"])]
    dq.issue("season totals with makes > attempts", "error" if len(badpct) else "info", len(badpct))
    rng = ps[(ps["ts"] < 0) | (ps["ts"] > 1.5) | (ps["usg"] < 0) | (ps["usg"] > 60)]
    dq.issue("implausible TS% / usage values", "warning" if len(rng) else "info", len(rng), "",
             [f"{r.player_id}/{r.year}" for r in rng.head(3).itertuples()])
    multi = ps[ps["n_teams"] > 1]
    dq.issue("players with several team entries in a season (traded)", "info", len(multi),
             f"{multi['player_id'].nunique()} distinct players; each stint preserved in `stints`")
    chg = ps.sort_values(["player_id", "year"]).copy()
    chg["prev_last"] = chg.groupby("player_id")["last_team"].shift()
    chg["prev_year"] = chg.groupby("player_id")["year"].shift()
    moved = chg[(chg["prev_year"] == chg["year"] - 1) & (chg["prev_last"] != chg["first_team"])]
    dq.issue("players changing teams between consecutive seasons", "info", len(moved))


def check_current(cur: pd.DataFrame | None, ps: pd.DataFrame, target_year: int, dq: DataQuality) -> None:
    if cur is None or len(cur) == 0:
        dq.issue("current roster", "error", 1, "no current roster available - cannot produce projections")
        return
    no_team = cur["team"].isna() | (cur["team"].astype(str).str.strip() == "")
    dq.issue("missing current-team information", "warning" if no_team.any() else "info", int(no_team.sum()))
    dq.issue("players on duplicate current rosters", "warning" if cur["player_id"].duplicated().any() else "info",
             int(cur["player_id"].duplicated().sum()))
    last = ps[ps["year"] == target_year - 1]
    known = set(ps["player_id"])
    rookies = cur[~cur["player_id"].isin(known)]
    dq.issue("rostered players with no NBA game history (rookies / new)", "info", len(rookies),
             "routed to the rookie/newcomer model", rookies["player_name"].head(5).tolist())
    gone = last[(last["min"] >= 500) & ~last["player_id"].isin(set(cur["player_id"]))]
    dq.issue("last-season rotation players (>=500 min) not on any current roster", "info", len(gone),
             "free agents / retired / unsigned - excluded from rankings", gone["player_name"].head(5).tolist())

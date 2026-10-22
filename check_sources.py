#!/usr/bin/env python
"""Smoke-test the data sources with a few tiny requests (needs internet) and pinpoint WHY a request fails.

    python check_sources.py                   test stats.nba.com / ESPN-injuries / Rotowire (default profile)
    python check_sources.py --source espn     test the ESPN-only profile (teams, schedule, box score, rosters, athletes)
    python check_sources.py --diagnose        low-level network diagnosis per host: DNS, TCP, TLS/HTTP, timing, exception class
    python check_sources.py --verbose         also print a sample row / full tracebacks

Typical causes (see --diagnose output):  ReadTimeout after a successful TCP connect = the server accepted the connection but never
answered (stats.nba.com does this to cloud / VPN / non-browser clients);  ConnectTimeout = firewall / routing;  SSLError = proxy
or clock problem;  HTTP 403/429 = blocked / rate limited.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))


def roster_check(s, y):
    from src.data_sources.nba import parse_result_sets
    from src.scraping import validators as V
    from src.utils import season_label
    payload = s._call("commonteamroster", {"LeagueID": "00", "Season": season_label(y), "TeamID": 1610612737}, 24,
                      V.nba_resultsets(0, ("PLAYER_ID",), set_name="CommonTeamRoster"))
    return s._canon_roster(parse_result_sets(payload)["CommonTeamRoster"].assign(team="ATL"), y)


def diagnose(cfg) -> int:
    import requests
    from src.data_sources.nba import NBA_HEADERS
    season = cfg.last_completed_year
    targets = [
        ("stats.nba.com  roster", f"https://stats.nba.com/stats/commonteamroster?LeagueID=00&Season={season}-{(season + 1) % 100:02d}&TeamID=1610612737", NBA_HEADERS),
        ("ESPN teams", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams", {}),
        ("ESPN schedule", f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/1/schedule?season={season + 1}&seasontype=2", {}),
        ("ESPN roster", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/1/roster", {}),
        ("ESPN athlete", "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/3032977", {}),
        ("ESPN injuries", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries", {}),
    ]
    ua = {"User-Agent": cfg.get("scraping.user_agent", "Mozilla/5.0"), "Accept": "application/json"}
    bad = 0
    for label, url, hdr in targets:
        host = urlparse(url).hostname
        t = time.time()
        try:
            ip = socket.gethostbyname(host)
        except OSError as exc:
            print(f"FAIL  {label:24} DNS lookup failed: {exc}")
            bad += 1
            continue
        dns = time.time() - t
        t = time.time()
        try:
            socket.create_connection((host, 443), timeout=8).close()
        except OSError as exc:
            print(f"FAIL  {label:24} TCP connect to {ip}:443 failed after {time.time() - t:.1f}s: {type(exc).__name__}: {exc}")
            bad += 1
            continue
        conn = time.time() - t
        t = time.time()
        try:
            r = requests.get(url, headers={**ua, **hdr}, timeout=(8, 25))
            dt = time.time() - t
            try:
                keys = list(r.json())[:5] if r.headers.get("content-type", "").startswith("application/json") else "non-JSON body"
            except ValueError:
                keys = "unparseable JSON"
            ok = r.status_code == 200
            bad += 0 if ok else 1
            print(f"{'PASS' if ok else 'FAIL'}  {label:24} HTTP {r.status_code}  {len(r.content) / 1024:.0f} KB in {dt:.1f}s  (dns {dns:.2f}s, tcp {conn:.2f}s)  keys={keys}")
        except requests.exceptions.ReadTimeout:
            bad += 1
            print(f"FAIL  {label:24} ReadTimeout: TCP/TLS connected in {conn:.2f}s but NO response within 25 s - the server is "
                  "ignoring this client/IP (typical of stats.nba.com for cloud, VPN or script traffic)")
        except requests.exceptions.ConnectTimeout:
            bad += 1
            print(f"FAIL  {label:24} ConnectTimeout during TLS/HTTP setup (firewall, proxy or routing)")
        except requests.exceptions.SSLError as exc:
            bad += 1
            print(f"FAIL  {label:24} SSLError: {str(exc)[:160]} (proxy interception or wrong system clock?)")
        except requests.RequestException as exc:
            bad += 1
            print(f"FAIL  {label:24} {type(exc).__name__}: {str(exc)[:160]}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--diagnose", action="store_true", help="per-host network diagnosis (DNS/TCP/HTTP/timing/exception class)")
    ap.add_argument("--source", choices=["nba", "espn"], default="nba")
    ap.add_argument("--config", type=str)
    args = ap.parse_args(argv)
    from src.config import Config
    from src.data_sources import build_source
    from src.scraping import validators as V
    from src.scraping.scraper import network_reachable
    from src.utils import season_label

    cfg = Config.load(args.config)
    if args.source == "espn":
        cfg.use_espn_stats()
    y = cfg.last_completed_year
    print(f"profile '{args.source}'; target season {season_label(cfg.target_year)}; testing with completed season {season_label(y)}\n")
    if args.diagnose:
        return diagnose(cfg)
    if not network_reachable():
        print("FAIL  no network connection (could not reach stats.nba.com or site.api.espn.com) - run with --diagnose for details")
        return 1

    fails = 0
    if args.source == "espn":
        src = build_source("espn_stats", cfg)
        for label, ok, detail in src.probe():
            fails += 0 if ok else 1
            print(f"{'PASS' if ok else 'FAIL'}  [espn_stats] {label:24} {detail}")
        for name, label, fn in (("espn", "injury report", lambda s: s.injuries()),
                                ("espn", "fantasy ADP / rank (comparison only)", lambda s: s.consensus(cfg.target_year))):
            try:
                res = fn(build_source(name, cfg))
                print(f"PASS  [{name:10}] {label:24} rows={len(res)}")
            except Exception as exc:
                print(f"WARN  [{name:10}] {label:24} {type(exc).__name__}: {str(exc)[:150]}  (optional)")
        print("\n" + ("ESPN core checks passed - run: python project.py --source espn" if not fails else
                      f"{fails} ESPN check(s) failed - the message above shows the exact payload problem; run with --diagnose for network details."))
        return 1 if fails else 0

    checks = [
        ("nba", "core", "team game logs", lambda s: s.team_game_logs(y)),
        ("nba", "core", "starters (leaguedashplayerstats)", lambda s: s.starters(y)),
        ("nba", "core", "roster + coaches (ATL)", lambda s: roster_check(s, y)),
        ("nba", "core", "player index (draft info)", lambda s: s.player_index()),
        ("nba", "core", "player game logs (largest request)", lambda s: s.player_game_logs(y)),
        ("espn", "optional", "injury report", lambda s: s.injuries()),
        ("espn", "optional", "fantasy ADP / rank (comparison only)", lambda s: s.consensus(cfg.target_year)),
        ("rotowire", "optional", "injury report", lambda s: s.injuries()),
    ]
    core_fail, objs = 0, {}
    for src, level, label, fn in checks:
        try:
            objs.setdefault(src, build_source(src, cfg))
            res = fn(objs[src])
            n = len(res) if hasattr(res, "__len__") else "?"
            print(f"PASS  [{src:8}] {label:42} rows={n}")
            if args.verbose and hasattr(res, "head") and len(res):
                print("        ", res.iloc[0].to_dict())
        except Exception as exc:
            core_fail += level == "core"
            print(f"FAIL  [{src:8}] {label:42} {type(exc).__name__}: {str(exc)[:150]}")
            if args.verbose:
                traceback.print_exc()
    print("\n" + ("All core NBA checks passed." if not core_fail else
                  f"{core_fail} core check(s) failed. If these are timeouts, run --diagnose, then use the ESPN profile: "
                  "python project.py --source espn"))
    return 1 if core_fail else 0


if __name__ == "__main__":
    sys.exit(main())

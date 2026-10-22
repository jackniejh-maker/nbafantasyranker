#!/usr/bin/env python
"""NBA fantasy projection & ranking system for an ESPN points league.

    python project.py                       full run using cached data (refreshes stale current-context data)
    python project.py --top 200             print the top 200
    python project.py --player "Name"       detailed quantitative report for one player
    python project.py --update-data         force-refresh rosters / injuries / consensus / in-season games
    python project.py --retrain             force model retraining
    python project.py --demo                run the whole pipeline on an offline synthetic league (no network)
    python project.py --source espn         use ESPN's public API instead of stats.nba.com (first run: ~1,300 requests/season)

No AI/LLM API is used anywhere: every number comes from the statistical models trained on local data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top", type=int, default=50, help="number of players to print (default 50)")
    p.add_argument("--player", type=str, help="print the detailed report for this player")
    p.add_argument("--update-data", action="store_true", help="force refresh of current-context data (rosters, injuries, consensus, in-progress season)")
    p.add_argument("--refresh-history", action="store_true", help="re-download completed historical seasons too (rarely needed)")
    p.add_argument("--retrain", action="store_true", help="ignore saved models and retrain")
    p.add_argument("--tune-weights", action="store_true", help="re-tune the recency-weight adjustments (seasons before the first test season only), then retrain")
    p.add_argument("--source", choices=["nba", "espn"], default="nba",
                   help="data provider: 'nba' = stats.nba.com (default); 'espn' = ESPN public API only (own data folder data_espn/)")
    p.add_argument("--demo", action="store_true", help="use the offline synthetic league (demo_run/ folders)")
    p.add_argument("--offline", action="store_true", help="never touch the network; use cached data only")
    p.add_argument("--crosscheck", action="store_true", help="cross-check totals against Basketball-Reference (secondary source)")
    p.add_argument("--season", type=str, help="target season, e.g. 2026 or 2026-27 (default: upcoming/current season)")
    p.add_argument("--n-sims", type=int, help="Monte Carlo draws per player (default 10000)")
    p.add_argument("--seed", type=int, help="simulation seed")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-reports", action="store_true")
    p.add_argument("--config", type=str, help="path to a league_settings.json")
    p.add_argument("--data-dir", type=str)
    p.add_argument("--output-dir", type=str)
    return p


def make_config(args):
    from src.config import Config
    cfg = Config.load(args.config, data_dir=args.data_dir, output_dir=args.output_dir)
    if getattr(args, "source", "nba") == "espn":
        cfg.use_espn_stats()
    if args.demo:
        cfg.apply_demo()
    if args.season:
        cfg.set("seasons.target_season", args.season)
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = make_config(args)
    from src import pipeline
    from src.utils import setup_logging
    cfg.make_dirs()
    setup_logging(cfg.outputs_dir / "run.log")
    res = None
    if args.player and not (args.update_data or args.retrain or args.refresh_history or args.tune_weights):
        res = pipeline.cached_result(cfg)
    if res is None:
        from src.ingest import IngestError
        try:
            res = pipeline.run_pipeline(cfg, args)
        except (IngestError, RuntimeError) as exc:
            print(f"\nERROR: {exc}\n\nNothing was projected.  Try:  python project.py --demo   (offline synthetic league)\n"
                  "or fix connectivity / drop canonical CSVs into data/external/ (see README, 'Data sources').", file=sys.stderr)
            return 1
        if not args.player:
            pipeline.print_top(res.rank, args.top)
    if args.player:
        row, suggestions = pipeline.find_player(res.rank, args.player)
        if row is None:
            print(f"No unique player matches {args.player!r}." + (f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""))
            return 2
        text = pipeline._report_text(res, row)
        print("\n" + text)
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        from src.utils import slugify
        (cfg.reports_dir / f"{int(row['rank_total']):03d}_{slugify(row['player_name'])}.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

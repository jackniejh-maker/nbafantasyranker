#!/usr/bin/env python
"""Train / validate the models only (no projections).

    python train.py                 train if data or config changed (otherwise reuse saved models)
    python train.py --retrain       force retraining
    python train.py --tune-weights  tune the recency-weight adjustments on pre-test seasons, then retrain
    python train.py --demo          same, on the offline synthetic league
Writes outputs/model_performance.csv and outputs/feature_importance.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--tune-weights", action="store_true")
    p.add_argument("--update-data", action="store_true")
    p.add_argument("--refresh-history", action="store_true")
    p.add_argument("--source", choices=["nba", "espn"], default="nba")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--season", type=str)
    p.add_argument("--config", type=str)
    p.add_argument("--data-dir", type=str)
    p.add_argument("--output-dir", type=str)
    args = p.parse_args(argv)
    from project import make_config
    from src import pipeline
    from src.utils import setup_logging
    cfg = make_config(args)
    cfg.make_dirs()
    setup_logging(cfg.outputs_dir / "run.log")
    pdx, models, _ = pipeline.prepare(cfg, args, train=True)
    models.performance.to_csv(cfg.outputs_dir / "model_performance.csv", index=False)
    models.importance.to_csv(cfg.outputs_dir / "feature_importance.csv", index=False)
    models.feature_policy.to_csv(cfg.outputs_dir / "feature_policy.csv", index=False)
    print("selected models:", {k: r.selected for k, r in models.targets.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

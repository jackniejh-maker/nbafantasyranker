"""Minutes-per-game-when-active model.

Baseline  recency-weighted historical MPG (credibility-blended with a draft-slot prior for thin histories)
A0        the same baseline redistributed over the preseason roster so team minutes sum to 240 (no fitting)
Ridge / Elastic Net / Gradient boosting on role, depth-chart, teammate-competition and coach features.
A manual `expected_minutes_override` (config/player_overrides.csv) replaces the final projection.
"""
from __future__ import annotations

from .validation import TargetSpec, run_target

def _weight(Y, cfg):
    return Y["y_gp"] / (Y["y_gp"] + 20.0)


SPEC = TargetSpec(name="mpg", y_col="y_mpg", base_col="base_mpg", elig_col="elig_mpg", alt_col="mpg_budget",
                  alt_label="A0_budget_redistributed", weight_fn=_weight)


def run(tab, cfg, test_years):
    return run_target(tab, SPEC, cfg, test_years)

"""Rookie / low-history ("newcomer") pathway.

Players with fewer than `eligibility.min_history_minutes` prior NBA minutes cannot be projected from a
track record.  A separate Ridge model (trained only on historical newcomers, with chronological validation)
maps pre-NBA information - draft slot, age, size, nationality, position - plus opportunity features (minutes
vacated on the team, same-position competition, coach rookie usage) and the credibility-weighted limited NBA
sample, onto FPPM, minutes and availability.  Output is flagged low-confidence and gets wider simulated ranges.

College / G-League / international / Summer-League numbers have no stable public structured API; supply them
through data/external/prospects.csv (columns: player_name, fppm_adjustment, mpg_adjustment - optional, applied
as additive manual priors) - see README.
"""
from __future__ import annotations

import numpy as np

from .validation import TargetSpec, run_target

NEWC_FEATURES = [
    "newc__log_pick", "newc__undrafted", "newc__lottery", "newc__height_in", "newc__intl", "newc__cred",
    "newc__obs_dev", "newc__is_rookie", "age__age", "age__c2", "age__exp", "pos__g", "pos__f", "pos__c",
    "team__min_vacated", "team__min_added", "team__min_net", "team__continuity", "team__oversub", "team__depth15",
    "mate__same_pos_min", "mate__rank_in_pos", "mate__usg_top2", "mate__best_fppm", "role__mpg_ref",
    "role__mpg_budget", "role__budget_ratio", "coach__rookie_mpg", "coach__y2_mpg", "hist__fppm_shrunk",
    "hist__avail_shrunk", "hist__eff_minutes_log"]


def _w_fppm(Y, cfg):
    return Y["y_min"] / (Y["y_min"] + 300.0)


def _w_mpg(Y, cfg):
    return Y["y_gp"] / (Y["y_gp"] + 10.0)


def _w_avail(Y, cfg):
    return np.ones(len(Y))


def specs() -> dict[str, TargetSpec]:
    return {
        "fppm": TargetSpec("newc_fppm", "y_fppm", "base_fppm", "elig_newc_fppm", feature_cols=NEWC_FEATURES, min_rows=60,
                           weight_fn=_w_fppm),
        "mpg": TargetSpec("newc_mpg", "y_mpg", "base_mpg", "elig_newc_mpg", feature_cols=NEWC_FEATURES, min_rows=60,
                          weight_fn=_w_mpg),
        "avail": TargetSpec("newc_avail", "y_avail", "base_avail", "elig_newc_avail", feature_cols=NEWC_FEATURES,
                            min_rows=60, weight_fn=_w_avail),
    }


def run(tab, cfg, test_years):
    return {k: run_target(tab, s, cfg, test_years, kinds=("baseline", "ridge")) for k, s in specs().items()}

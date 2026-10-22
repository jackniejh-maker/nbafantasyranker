"""Availability (games played / team games) model.

No attempt is made to predict individual injuries.  The expected availability rate comes from historical
patterns only (previous availability, age, workload, role); uncertainty comes from the empirical,
skewed distribution of out-of-sample residuals (see simulation.py), which includes full-season absences.
"""
from __future__ import annotations

import numpy as np

from .validation import TargetSpec, run_target

def _weight(Y, cfg):
    return np.ones(len(Y))


SPEC = TargetSpec(name="avail", y_col="y_avail", base_col="base_avail", elig_col="elig_avail", weight_fn=_weight)


def run(tab, cfg, test_years):
    return run_target(tab, SPEC, cfg, test_years, kinds=("baseline", "ridge", "elastic_net", "gbm"))

"""FPPM (fantasy points per minute) forecasting - the central production target.

Model A  recency-weighted, role/team/age-adjusted, credibility-shrunk historical average (benchmark)
Model A0 same with the fixed 50/30/20 weights and no adjustments (tests whether the adjustments help)
Model B  Ridge       Model C  Elastic Net       Model D  Gradient boosting (LightGBM/XGBoost/sklearn)
B-D learn an adjustment on top of Model A from the engineered feature groups, using chronological CV.
"""
from __future__ import annotations

from .validation import TargetSpec, run_target

def _weight(Y, cfg):
    """Season FPPM is a noisier measurement for low-minute seasons: weight = minutes / (minutes + k)."""
    return Y["y_min"] / (Y["y_min"] + float(cfg.get("models.target_weight_k_minutes", 600)))


SPEC = TargetSpec(name="fppm", y_col="y_fppm", base_col="base_fppm", elig_col="elig_fppm",
                  alt_col="base_fppm_fixed", alt_label="A0_fixed_weights", weight_fn=_weight)


def run(tab, cfg, test_years):
    return run_target(tab, SPEC, cfg, test_years)

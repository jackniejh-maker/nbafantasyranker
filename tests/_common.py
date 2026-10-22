"""Shared helpers for the test-suite (stdlib unittest; also runs under pytest)."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config  # noqa: E402


def small_cfg(root=None, partial_rounds: int = 0, seed: int = 5) -> Config:
    """A tiny offline synthetic league: 10 teams, 50-game seasons, 2021-2025 history, 2026 target."""
    root = Path(root or tempfile.mkdtemp(prefix="fr_test_"))
    cfg = Config.load().apply_demo(root=root)
    for k, v in {"seasons.first_season": 2021, "seasons.target_season": 2026, "seasons.season_length": 50,
                 "demo.n_teams": 10, "demo.n_rounds": 50, "demo.seed": seed, "demo.partial_target_rounds": partial_rounds,
                 "validation.n_test_seasons": 2, "validation.min_train_seasons": 2, "validation.inner_cv_min_seasons": 2,
                 "models.gbm_grid": [{"max_depth": 2, "n_estimators": 60, "learning_rate": 0.08}],
                 "models.ridge_alphas": [10, 100, 1000], "models.enet_alphas": [0.03, 0.1], "models.enet_l1_ratios": [0.5],
                 "simulation.n_sims": 1000, "game_model.enabled": False, "eligibility.min_history_minutes": 300}.items():
        cfg.set(k, v)
    return cfg

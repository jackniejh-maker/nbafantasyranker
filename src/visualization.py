"""The ten required plots -> outputs/plots/*.png  (matplotlib, headless backend)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import log

PALETTE = {"main": "#2b6cb0", "alt": "#dd6b20", "good": "#2f855a", "bad": "#c53030", "grey": "#718096"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    return plt


def make_all(rank: pd.DataFrame, models, ps: pd.DataFrame, cfg, out_dir) -> list[str]:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    made = []
    rel = rank[(rank["projected_minutes"] >= 10) & (rank["projected_games"] >= 10)]
    top = rank.head(150)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name)
        plt.close(fig)
        made.append(name)

    # 1 projected FPPG distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rel["fppg_p50"], bins=30, color=PALETTE["main"], alpha=0.85)
    ax.axvline(rel["fppg_p50"].median(), color=PALETTE["alt"], ls="--", label=f"median {rel['fppg_p50'].median():.1f}")
    ax.set(title="Projected fantasy points per game (median of simulations)", xlabel="FPPG", ylabel="players")
    ax.legend()
    save(fig, "01_fppg_distribution.png")

    # 2 projected total fantasy points
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rel["total_p50"], bins=30, color=PALETTE["good"], alpha=0.85)
    ax.set(title="Projected season fantasy points (median of simulations)", xlabel="total fantasy points", ylabel="players")
    save(fig, "02_total_fantasy_points.png")

    # 3 FPPM vs MPG
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(rel["projected_minutes"], rel["projected_fppm"], c=rel["total_p50"], s=14 + rel["total_p50"] / 60,
                    cmap="viridis", alpha=0.85)
    for _, r in rank.head(10).iterrows():
        ax.annotate(r["player_name"], (r["projected_minutes"], r["projected_fppm"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, label="projected total FP")
    ax.set(title="Production vs opportunity", xlabel="projected minutes per game", ylabel="projected FPPM")
    save(fig, "03_fppm_vs_mpg.png")

    # 4 age vs FPPM (history + projection)
    fig, ax = plt.subplots(figsize=(7, 5))
    h = ps[(ps["min"] >= 800) & ps["age"].notna()]
    ax.scatter(h["age"], h["fppm"], s=5, color=PALETTE["grey"], alpha=0.35, label="historical player-seasons")
    bins = np.arange(np.floor(h["age"].min()), np.ceil(h["age"].max()) + 1)
    g = h.groupby(pd.cut(h["age"], bins))["fppm"].mean()
    ax.plot([i.mid for i in g.index], g.values, color=PALETTE["bad"], lw=2, label="mean by age")
    ax.scatter(rel["age"], rel["projected_fppm"], s=8, color=PALETTE["main"], alpha=0.7, label="projected (target season)")
    ax.set(title="Age vs FPPM", xlabel="age (Feb 1)", ylabel="FPPM")
    ax.legend()
    save(fig, "04_age_vs_fppm.png")

    # 5 historical vs projected FPPM
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    d = rel.dropna(subset=["historical_fppm"])
    ax.scatter(d["historical_fppm"], d["projected_fppm"], s=12, alpha=0.7, color=PALETTE["main"])
    lim = [min(d["historical_fppm"].min(), d["projected_fppm"].min()) - 0.05, max(d["historical_fppm"].max(), d["projected_fppm"].max()) + 0.05]
    ax.plot(lim, lim, color=PALETTE["bad"], ls="--", lw=1)
    ax.set(title="Last season's FPPM vs projection", xlabel="historical FPPM (last season)", ylabel="projected FPPM", xlim=lim, ylim=lim)
    save(fig, "05_historical_vs_projected_fppm.png")

    perf = models.performance
    # 6 projection error by season
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    tot = perf[(perf["target"] == "total_fp") & (perf["test_year"] != "all")].copy()
    tot["test_year"] = tot["test_year"].astype(int)
    for model, col in (("pipeline_selected", PALETTE["main"]), ("baseline_weighted_history", PALETTE["alt"]),
                       ("naive_last_season_total", PALETTE["grey"])):
        t = tot[tot["model"] == model].sort_values("test_year")
        axes[0].plot(t["test_year"], t["rmse"].astype(float), marker="o", color=col, label=model)
        axes[1].plot(t["test_year"], t["spearman"].astype(float), marker="o", color=col, label=model)
    yrs = sorted(tot["test_year"].unique())
    for ax in axes:
        ax.set_xticks(yrs)
        ax.set_xticklabels([f"{y}-{(y + 1) % 100:02d}" for y in yrs])
    axes[0].set(title="Out-of-sample RMSE of season total FP", xlabel="test season", ylabel="RMSE (fantasy points)")
    axes[1].set(title="Out-of-sample rank correlation (Spearman)", xlabel="test season", ylabel="Spearman")
    axes[0].legend(fontsize=7)
    save(fig, "06_projection_error_by_season.png")

    # 7 model comparison (selection metric, lower is better)
    metric = cfg.get("validation.selection_metric", "top_n_mae")
    order = ["A_weighted_average", "A0_fixed_weights", "A0_budget_redistributed", "B_ridge", "C_elastic_net", "D_gradient_boosting"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, tg, ttl in zip(axes, ("fppm", "mpg", "avail"), ("FPPM", "Minutes per game", "Availability")):
        m = perf[(perf["target"] == tg) & (perf["test_year"] == "mean")].copy()
        m[metric] = m[metric].astype(float)
        m["o"] = m["label"].map({k: i for i, k in enumerate(order)})
        m = m.sort_values("o")
        ax.barh(m["label"], m[metric], color=[PALETTE["good"] if s_ else PALETTE["main"] for s_ in m["selected"]])
        ax.set(title=f"{ttl}\n{metric} (green = selected)", xlabel=metric)
        ax.invert_yaxis()
    save(fig, "07_model_comparison.png")

    # 8 projection uncertainty
    fig, ax = plt.subplots(figsize=(8, 8))
    t = rank.head(40).iloc[::-1]
    y = np.arange(len(t))
    ax.hlines(y, t["total_p10"], t["total_p90"], color=PALETTE["grey"], lw=1.5, label="10th-90th pct")
    ax.hlines(y, t["total_p25"], t["total_p75"], color=PALETTE["main"], lw=5, label="25th-75th pct")
    ax.scatter(t["total_p50"], y, color="white", edgecolor="black", zorder=3, s=18, label="median")
    ax.set_yticks(y)
    ax.set_yticklabels(t["player_name"], fontsize=7)
    ax.set(title="Top 40: simulated season-total ranges", xlabel="total fantasy points")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "08_projection_uncertainty.png")

    # 9 current (market) rank vs model rank
    fig, ax = plt.subplots(figsize=(6.5, 6))
    has = top["consensus_rank"].notna()
    if has.any():
        ax.scatter(top.loc[has, "consensus_rank"], top.loc[has, "rank_total"], s=14, color=PALETTE["main"])
        mx = max(top.loc[has, "consensus_rank"].max(), top.loc[has, "rank_total"].max())
        ax.plot([0, mx], [0, mx], color=PALETTE["bad"], ls="--", lw=1)
        big = top[has].assign(d=lambda x: (x["rank_total"] - x["consensus_rank"]).abs()).nlargest(8, "d")
        for _, r in big.iterrows():
            ax.annotate(r["player_name"], (r["consensus_rank"], r["rank_total"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(title="Consensus (market) rank vs model rank - below the line = model likes more", xlabel="consensus rank", ylabel="model rank (total)")
    else:
        ax.scatter(top["rank_total"] + top["rank_vs_last_season"], top["rank_total"], s=14, color=PALETTE["main"])
        ax.set(title="No consensus data: last-season rank vs projected rank", xlabel="last-season rank (by total FP)", ylabel="model rank (total)")
    save(fig, "09_consensus_vs_model_rank.png")

    # 10 breakout candidates
    fig, ax = plt.subplots(figsize=(8, 6))
    b = rank.dropna(subset=["breakout_score"]).nlargest(20, "breakout_score").iloc[::-1]
    ax.barh(b["player_name"], b["breakout_score"], color=PALETTE["alt"])
    ax.set(title="Breakout candidates (0-100 percentile of measurable-factor score)", xlabel="breakout score")
    save(fig, "10_breakout_candidates.png")
    log.info("wrote %d plots to %s", len(made), out)
    return made

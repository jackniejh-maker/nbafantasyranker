"""Quantitative explanations - every statement is computed from model outputs, never hand-written.

group effects   For each context group g (team, teammates, coach, age, role, form) the explanatory model is
                re-evaluated with g's features replaced by their training medians ("neutral").  The change in
                projected fantasy points per game  (FPPM_hat x MPG_hat)  is the effect of that group, in FPPG.
feature terms   For the linear explanatory model each feature's contribution to the FPPM / minutes adjustment is
                coefficient x standardised value (exactly additive).
drivers         Observable changes vs last season (minutes, FPPM, games) and the group effects, converted to total
                fantasy points over the projected games so they can be ranked on one scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_MAP = {"team_effect": ("team",), "teammate_effect": ("mate",), "coach_effect": ("coach",),
             "age_effect": ("age",), "role_effect": ("role", "pos"), "form_effect": ("late", "trade"),
             "history_adjustment": ("hist",)}
MIN_IMPACT_FP = 40.0       # ignore drivers worth less than ~0.5 FPPG over a season


# ====================================================================== group effects
def _ablated_adjustment(tm, X: pd.DataFrame, prefixes) -> np.ndarray:
    clean = tm.est_.named_steps["clean"]
    med = dict(zip(clean.columns_, clean.median_))
    cols = [c for c in clean.columns_ if c.split("__")[0] in prefixes]
    X2 = X.copy()
    for c in cols:
        X2[c] = med[c]
    return tm.adjustment(X2)


def group_effects(models, X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Effect of each feature group on projected FPPG (FPPM x minutes), per player."""
    out = pd.DataFrame(0.0, index=X.index, columns=list(GROUP_MAP))
    for path in ("main", "newcomer"):
        rows = (meta["pathway"] == path).to_numpy()
        if not rows.any():
            continue
        if path == "main":
            tf, tmn = models.targets["fppm"].explain, models.targets["mpg"].explain
        else:
            tf, tmn = models.newcomer["fppm"].final, models.newcomer["mpg"].final
        Xr, mr = X[rows], meta[rows]
        af = tf.adjustment(Xr) if tf.est_ is not None else np.zeros(len(Xr))
        am = tmn.adjustment(Xr) if tmn.est_ is not None else np.zeros(len(Xr))
        bf, bm = mr["base_fppm"].to_numpy(), mr["base_mpg"].to_numpy()
        full = (bf + af) * (bm + am)
        for name, pref in GROUP_MAP.items():
            af2 = _ablated_adjustment(tf, Xr, pref) if tf.est_ is not None else af
            am2 = _ablated_adjustment(tmn, Xr, pref) if tmn.est_ is not None else am
            out.loc[rows, name] = full - (bf + af2) * (bm + am2)
    return out


def linear_contributions(tm, X_row: pd.DataFrame) -> pd.Series:
    """Per-feature additive contribution (target units) of a fitted linear TargetModel for one player."""
    if tm.est_ is None or not tm.is_linear:
        return pd.Series(dtype=float)
    est = tm.est_
    clean, scale, model = est.named_steps["clean"], est.named_steps["scale"], est.named_steps["model"]
    Z = scale.transform(clean.transform(X_row))
    return pd.Series(Z[0] * model.coef_ * tm.y_sd_, index=clean.columns_)


# ====================================================================== text helpers
def confidence_label(score: float) -> str:
    if not np.isfinite(score):
        return "Low"
    return "High" if score >= 66.7 else "Medium" if score >= 33.3 else "Low"


_GROUP_TEXT = {
    "teammate_effect": ("favorable teammate environment (usage / playmaking / rebounding opportunity)",
                        "stronger teammate competition for usage, shots and minutes"),
    "team_effect": ("favorable team environment (pace / efficiency / roster turnover)",
                    "unfavorable team environment (pace / efficiency / roster turnover)"),
    "coach_effect": ("coach tendencies (rotation / pace) help", "coach tendencies (rotation / pace) hurt"),
    "age_effect": ("age / development curve favors growth", "age-related decline expected"),
    "role_effect": ("role and depth-chart position favor more production", "role and depth-chart position limit production"),
    "form_effect": ("late-season / post-trade form is a positive signal", "late-season / post-trade form is a negative signal"),
}


def driver_lines(r: pd.Series, n: int = 4) -> tuple[list[str], list[str]]:
    """(positive, negative) driver strings from one row of the projection table, ranked by total-FP impact."""
    games = float(r.get("projected_games", np.nan))
    fppg, minutes, fppm = float(r["projected_fppg"]), float(r["projected_minutes"]), float(r["projected_fppm"])
    items: list[tuple[float, str]] = []
    if np.isfinite(games):
        ref_mpg, ref_fppm, ref_gp = r.get("ref_mpg", np.nan), r.get("ref_fppm", np.nan), r.get("ref_gp", np.nan)
        if np.isfinite(ref_mpg):
            dm = minutes - ref_mpg
            items.append((dm * fppm * games, f"{'more' if dm > 0 else 'fewer'} minutes ({ref_mpg:.1f} -> {minutes:.1f} mpg)"))
        if np.isfinite(ref_fppm):
            df = fppm - ref_fppm
            items.append((df * minutes * games, f"{'higher' if df > 0 else 'lower'} per-minute production "
                                                  f"(FPPM {ref_fppm:.2f} -> {fppm:.2f})"))
        if np.isfinite(ref_gp) and r.get("ref_team_games", 0) > 0:
            dg = games - ref_gp
            items.append((dg * fppg, f"{'more' if dg > 0 else 'fewer'} games expected ({ref_gp:.0f} -> {games:.0f})"))
        for g, (pos_t, neg_t) in _GROUP_TEXT.items():       # "history_adjustment" is shown in the effects line, not as a driver
            e = float(r.get(g, 0.0) or 0.0)
            if abs(e) > 1e-9:
                items.append((e * games, (pos_t if e > 0 else neg_t) + f" [{e:+.1f} FPPG]"))
    pos = [f"+ {t}  (~{v:+.0f} FP)" for v, t in sorted(items, reverse=True) if v >= MIN_IMPACT_FP][:n]
    neg = [f"- {t}  (~{v:+.0f} FP)" for v, t in sorted(items) if v <= -MIN_IMPACT_FP][:n]
    return pos, neg


def player_report(r: pd.Series, hist: pd.DataFrame, stints: pd.DataFrame | None, contribs: dict, cfg, n_players: int) -> str:
    """Plain-text report for one player (r = row of the full projection table)."""
    L = []
    L.append(f"PLAYER: {r['player_name']}   ({r['team']}, {r['position']}, age {r['age']:.1f})")
    L.append("")
    L.append(f"Projected Rank (total value): {int(r['rank_total'])} of {n_players}"
             f"    | by FPPG: {int(r['rank_fppg'])}    | risk-adjusted: {int(r['rank_risk_adjusted'])}")
    if np.isfinite(r.get("consensus_rank", np.nan)):
        L.append(f"Consensus rank (comparison only): {int(r['consensus_rank'])}    "
                 f"model-minus-consensus rank: {r['rank_diff_vs_consensus']:+.0f}    "
                 f"value vs consensus slot: {r['value_diff_vs_consensus']:+.0f} FP")
    L.append("")
    L.append(f"Projected FPPM:    {r['projected_fppm']:.2f}      (80% range {r['fppm_p10']:.2f} - {r['fppm_p90']:.2f})")
    L.append(f"Projected Minutes: {r['projected_minutes']:.1f}      (80% range {r['minutes_p10']:.1f} - {r['minutes_p90']:.1f})")
    L.append(f"Projected FPPG:    {r['projected_fppg']:.1f}      (80% range {r['fppg_p10']:.1f} - {r['fppg_p90']:.1f})")
    L.append(f"Projected Games:   {r['projected_games']:.0f}      (availability {r['projected_avail']:.0%}, "
             f"80% range {r['games_p10']:.0f} - {r['games_p90']:.0f} games)")
    L.append(f"Projected Total:   {r['projected_total_fantasy_points']:.0f}")
    if r.get("gp_to_date", 0) > 0:
        L.append(f"  (in-season: {r['fp_to_date']:.0f} FP already banked over {r['gp_to_date']:.0f} games)")
    L.append("")
    for p in (10, 25, 50, 75, 90):
        L.append(f"{p}th percentile total: {r[f'total_p{p}']:.0f}")
    L.append(f"Mean of simulated totals: {r['total_mean']:.0f}   (Projected Total above = projected FPPG x expected games; the median can "
             "differ because availability is skewed by injuries)")
    L.append(f"Downside tail (mean of worst 10%): {r['total_cvar']:.0f}    risk-adjusted value: {r['risk_adjusted_value']:.0f}")
    L.append("")
    L.append("Historical (FPPM / minutes per game / FPPG / games):")
    if len(hist):
        for _, h in hist.sort_values("year").iterrows():
            yr = f"{int(h['year'])}-{(int(h['year']) + 1) % 100:02d}"
            note = f"  (traded {h['first_team']} -> {h['last_team']})" if h.get("traded") else ""
            L.append(f"  {yr}: FPPM {h['fppm']:.2f} | {h['mpg']:.1f} mpg | {h['fppg']:.1f} FPPG | {int(h['gp'])} GP | "
                     f"{h['last_team']}{note}")
        if stints is not None and len(stints) > 1:
            L.append("  last-season split by team stint (before vs after a trade):")
            for _, s in stints.sort_values("first_date").iterrows():
                L.append(f"    {s['team']}: {int(s['gp'])} GP, {s['mpg']:.1f} mpg, FPPM {s['fppm']:.2f} ({s['min']:.0f} min)")
    else:
        L.append("  no NBA history (rookie / new player)")
    if len(hist):
        h = hist.sort_values("year").iloc[-1]
        g = max(float(h["gp"]), 1.0)
        L.append(f"  last-season fantasy profile per game: scoring block {h['fp_scoring'] / g:+.1f} | rebounds {h['fp_rebounds'] / g:+.1f} | "
                 f"assists {h['fp_assists'] / g:+.1f} | stocks {h['fp_stocks'] / g:+.1f} | turnovers {h['fp_tov_loss'] / g:+.1f} | "
                 f"missed-shot cost {h['fp_miss_loss'] / g:+.1f}")
        L.append(f"  FP per FGA {h['fp_per_fga']:.2f} | FP per possession used {h['fp_per_poss']:.2f} | "
                 f"FPPM per usage point {h['fp_per_usg_pt']:.3f} | usage {h['usg']:.1f}% | TS% {h['ts']:.3f}")
    L.append("")
    pos, neg = driver_lines(r)
    L.append("Main positive drivers:")
    L += [f"  {x}" for x in pos] or ["  (none above the materiality threshold)"]
    L.append("Main negative drivers:")
    L += [f"  {x}" for x in neg] or ["  (none above the materiality threshold)"]
    L.append("")
    L.append("Context effects on projected FPPG (model ablation, vs a neutral environment):")
    L.append("  " + " | ".join(f"{k.replace('_effect', '')} {r[k]:+.1f}" for k in GROUP_MAP if k in r))
    for title, key in (("FPPM", "fppm"), ("minutes", "mpg")):
        s = contribs.get(key)
        if s is not None and len(s):
            top = pd.concat([s.nlargest(3), s.nsmallest(3)]).drop_duplicates()
            L.append(f"Top feature terms, {title} adjustment: " + "; ".join(f"{k} {v:+.3f}" for k, v in top.items()))
    L.append("")
    L.append(f"Archetype: {r.get('archetype', 'n/a')}    scoring-block share of FP: {r.get('scoring_share', np.nan):.0%}"
             f"    breakout score: {r.get('breakout_score', np.nan):.0f}/100")
    L.append(f"Upside score: {r['upside_score']:.0f}   Risk score: {r['risk_score']:.0f}   "
             f"P(role expansion >= +{cfg.get('breakout.minutes_expansion_delta', 3.0):.0f} mpg): {r['p_role_expansion']:.0%}")
    if r.get("is_rookie") or r.get("pathway") == "newcomer":
        L.append("NOTE: rookie / low-history player - projected from draft slot, age and opportunity (wider ranges, lower confidence).")
    if r.get("inj_note"):
        L.append(f"Injury report: {r['inj_note']}  (known games missed assumed: {r['games_out']:.0f})")
    if r.get("override_applied"):
        L.append(f"Manual override applied: {r['override_applied']}")
    L.append("")
    L.append(f"Confidence: {confidence_label(r['confidence_score'])}  (score {r['confidence_score']:.0f}/100)")
    return "\n".join(L) + "\n"

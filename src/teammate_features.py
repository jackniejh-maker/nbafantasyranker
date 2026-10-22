"""Teammate-environment features.

For each rostered player we describe the OTHER players on the same projected roster using only
their pre-season information (recency-weighted usage / assist / rebound rates, baseline minutes,
availability).  Competition features measure how much usage, playmaking, rebounding, shots and
same-position minutes the teammates will absorb.  Complementarity is left to the models to learn: we
add products of the player's own profile (usage, assist rate, rebound rate, 3-point rate) with the
matching teammate quantity, which lets even a linear model estimate e.g. "high-assist teammates raise
a catch-and-shoot wing's production but cap a ball-dominant guard's".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROT_MIN = 15.0      # minutes per game that defines a rotation player
BASE_COLS = ["team", "pos_group", "mpg_ref0", "avail_ref", "usg_w", "ast_w", "reb_w", "fga_pm_w", "tpar_w", "fppm_w"]


def _z(x: pd.Series) -> pd.Series:
    sd = x.std()
    return (x - x.mean()) / sd if sd and np.isfinite(sd) else x * 0.0


def teammate_features(base: pd.DataFrame) -> pd.DataFrame:
    """`base` indexed by player_id (see BASE_COLS).  Returns features indexed by player_id."""
    feats = {}
    for team, g in base.groupby("team"):
        ids = g.index.to_numpy()
        mpg = g["mpg_ref0"].to_numpy(float)
        em = mpg * g["avail_ref"].to_numpy(float)                   # expected minutes per team game
        usg, ast, reb = (g[c].to_numpy(float) for c in ("usg_w", "ast_w", "reb_w"))
        fga = g["fga_pm_w"].to_numpy(float) * em                    # expected FGA per game
        tpa = fga * g["tpar_w"].to_numpy(float)
        fppm = g["fppm_w"].to_numpy(float)
        pos = g["pos_group"].to_numpy()
        front = np.isin(pos, ["F", "C"])
        for i, pid in enumerate(ids):
            o = np.arange(len(ids)) != i
            mm, uu, aa, rr = em[o], usg[o], ast[o], reb[o]
            rot = mpg[o] >= ROT_MIN
            tot = max(mm.sum(), 1e-9)
            u_rot = np.sort(uu[rot])[::-1]
            top4 = np.argsort(-mpg[o])[:4]
            same = pos[o] == pos[i]
            feats[pid] = {
                "mate__usg_wavg": float((mm * uu).sum() / tot),
                "mate__usg_top2": float(u_rot[:2].sum()),
                "mate__usg_top3": float(u_rot[:3].sum()),
                "mate__high_usg_n": float(((uu >= 24) & (mpg[o] >= 20)).sum()),
                "mate__starter_usg": float(uu[top4].mean()) if len(top4) else np.nan,
                "mate__ast_wavg": float((mm * aa).sum() / tot),
                "mate__ast_max": float(aa[rot].max()) if rot.any() else 0.0,
                "mate__reb_wavg": float((mm * rr).sum() / tot),
                "mate__front_min": float(mm[front[o]].sum()),
                "mate__fga_sum": float(fga[o].sum()),
                "mate__tpa_sum": float(tpa[o].sum()),
                "mate__same_pos_min": float(mm[same].sum()),
                "mate__same_pos_n": float((same & (mpg[o] >= 12)).sum()),
                "mate__rank_in_pos": float(1 + (same & (mpg[o] > mpg[i])).sum()),
                "mate__quality_wavg": float((mm * fppm[o]).sum() / tot),
                "mate__best_fppm": float(fppm[o][rot].max()) if rot.any() else np.nan,
                "role__rank_in_team": float(1 + (mpg[o] > mpg[i]).sum()),
            }
    f = pd.DataFrame.from_dict(feats, orient="index")
    f = f.reindex(base.index)
    z = lambda s: _z(base[s])
    f["mate__x_usg"] = z("usg_w") * _z(f["mate__usg_top2"])
    f["mate__x_ast"] = z("ast_w") * _z(f["mate__ast_max"])
    f["mate__x_reb"] = z("reb_w") * _z(f["mate__reb_wavg"])
    f["mate__x_3p_ast"] = z("tpar_w") * _z(f["mate__ast_max"])
    return f

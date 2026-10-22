"""Synthetic NBA-like league generator (offline demo + test fixture).

It produces the *same canonical schemas* the scrapers produce - player/team game logs, starts,
rosters, coaches, player index, injuries and a noisy "consensus" ranking - so the whole pipeline
runs without network access.  The simulated world has the structures the model is supposed to
learn: aging curves, zero-sum teammate competition for shots/assists/rebounds, coach styles
(pace, rotation size, star concentration), injuries (incl. full-season absences), mid-season
trades, free-agent moves, drafted rookies whose talent depends on draft slot, and coaching changes.

The target season is simulated too but kept in `holdout` (never exposed to the pipeline), so an
end-to-end accuracy check is possible.  Nothing here resembles real players: names are random.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .data_sources.nba import TEAM_IDS
from .fantasy_scoring import DEFAULT_ESPN, fantasy_points

FIRST = ("James Michael Chris Marcus Jalen Darius Tyrese Devin Jordan Anthony Malik Terrence Isaiah Cole Trey "
         "Caleb Jaylen Dante Xavier Elijah Kevin Andre Damian Rudy Nikola Luka Jabari Kyle Zach Brandon Austin "
         "Cameron DeAndre Donovan Eric Frank Gary Harrison Ivan Jerome Kendrick Lamar Mason Nate Omar Paul Quincy "
         "Reggie Shane Troy Victor Wesley Zane Aaron Bryce Corey Dwight Emeka Felix Gordon Hassan Idris Jamal Kobe"
         ).split()
LAST = ("Johnson Williams Brown Jones Davis Miller Wilson Moore Taylor Anderson Thomas Jackson White Harris Martin "
        "Thompson Garcia Martinez Robinson Clark Rodriguez Lewis Lee Walker Hall Allen Young King Wright Scott "
        "Green Baker Adams Nelson Hill Campbell Mitchell Roberts Carter Phillips Evans Turner Torres Parker "
        "Collins Edwards Stewart Flores Morris Murphy Cook Rogers Morgan Peterson Cooper Reed Bailey Bell Gomez "
        "Kelly Howard Ward Cox Diaz Richardson Wood Watson Brooks Bennett Gray James Reyes Cruz Hughes Price "
        "Myers Long Foster Sanders Ross Morales Powell Sullivan Russell Ortiz Jenkins Gutierrez Perry Butler "
        "Barnes Fisher Henderson Coleman Simmons Patterson Jordan Reynolds Hamilton Graham Kim Alexander Ramos "
        "Wallace Griffin Okafor Petrov Nowak Silva Costa Mbeki Tanaka Novak Larsen").split()
POS_LABELS = {0: (["G", "G-F"], [0.7, 0.3]), 1: (["F", "G-F", "F-C"], [0.55, 0.15, 0.30]),
              2: (["C", "F-C"], [0.75, 0.25])}
TRAITS = ("t_score", "t_eff", "t_play", "t_reb", "t_def")


@dataclass
class DemoLeague:
    player_games: pd.DataFrame
    team_games: pd.DataFrame
    starters: pd.DataFrame
    rosters: pd.DataFrame
    coaches: pd.DataFrame
    player_index: pd.DataFrame
    current_rosters: pd.DataFrame
    current_coaches: pd.DataFrame
    injuries: pd.DataFrame
    consensus: pd.DataFrame
    holdout: pd.DataFrame
    meta: dict = field(default_factory=dict)


class _Generator:
    def __init__(self, seed: int, n_teams: int, first_year: int, n_rounds: int):
        if n_teams % 2:
            raise ValueError("n_teams must be even")
        self.rng = np.random.default_rng(seed)
        self.nt = n_teams
        self.first_year = first_year
        self.n_rounds = n_rounds
        self.abbr = np.array(list(TEAM_IDS.keys())[:n_teams])
        self.next_pid = 1_630_000
        self.used_names: set[str] = set()
        self.P = pd.DataFrame()
        self.coach_pool = pd.DataFrame()
        self.team_coach = np.zeros(n_teams, dtype=int)   # coach_id per team
        self.team_def = self.rng.normal(0, 1, n_teams)
        self.team_pace = self.rng.normal(0, 1.2, n_teams)
        self.next_coach = 9000

    # ------------------------------------------------------------------ people
    def _name(self) -> str:
        for _ in range(200):
            nm = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)}"
            if nm not in self.used_names:
                self.used_names.add(nm)
                return nm
        return f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)} {self.next_pid}"

    def _make_players(self, ages, ability, team, draft_year, draft_pick, debut_year, year) -> pd.DataFrame:
        n = len(ages)
        rng = self.rng
        pos = rng.choice(3, size=n, p=[0.40, 0.35, 0.25])
        labels = [rng.choice(POS_LABELS[p][0], p=POS_LABELS[p][1]) for p in pos]
        df = pd.DataFrame({
            "pid": np.arange(self.next_pid, self.next_pid + n),
            "name": [self._name() for _ in range(n)],
            "pos": pos, "pos_label": labels, "age": np.asarray(ages, float), "ability": np.asarray(ability, float),
            "team": np.asarray(team, int), "draft_year": draft_year, "draft_pick": draft_pick,
            "debut_year": debut_year,
            "height_in": np.round(np.array([75, 79, 83])[pos] + rng.normal(0, 1.6, n)),
            "weight": np.round(np.array([192, 215, 247])[pos] + rng.normal(0, 9, n)),
            "country": rng.choice(["USA", "USA", "USA", "Canada", "France", "Serbia", "Nigeria"], n),
        })
        for t in TRAITS:
            df[t] = 0.55 * df["ability"] + 0.84 * rng.normal(size=n)
        df["birth_dt"] = pd.Series(pd.Timestamp(year, 10, 1), index=df.index) - pd.to_timedelta(df["age"] * 365.25, unit="D")
        self.next_pid += n
        return df

    def init_league(self) -> None:
        rng, nt = self.rng, self.nt
        n = nt * 15
        ages = np.clip(np.round(17 + rng.gamma(7, 1.3, n)), 19, 38)
        ab = rng.normal(0, 1, n)
        team = np.repeat(np.arange(nt), 15)
        debut = self.first_year - np.maximum(0, ages - 20 - rng.integers(0, 2, n))
        drafted = rng.random(n) < 0.78
        pick = np.where(drafted, np.clip(np.round(30 - 14 * ab + rng.normal(0, 12, n)), 1, 60), np.nan)
        self.P = self._make_players(ages, ab, team, np.where(drafted, debut, np.nan), pick, debut, self.first_year)
        self.coach_pool = pd.DataFrame({
            "coach_id": np.arange(self.next_coach, self.next_coach + nt + 6),
            "name": [self._name() + " (HC)" for _ in range(nt + 6)],
            "pace_adj": rng.normal(0, 2.2, nt + 6), "gamma": rng.uniform(0.45, 0.95, nt + 6),
            "rot": rng.integers(8, 12, nt + 6), "employed": [True] * nt + [False] * 6})
        self.next_coach += nt + 6
        self.team_coach = self.coach_pool["coach_id"].to_numpy()[:nt].copy()

    # ------------------------------------------------------------------ rates
    def _rates(self) -> dict[str, np.ndarray]:
        P = self.P
        pos = P["pos"].to_numpy()
        ts, te, tp, tr, td = (P[t].to_numpy() for t in TRAITS)
        return {
            "u": np.exp(0.42 * ts + np.array([0.15, 0.0, -0.25])[pos]),
            "fg2": np.clip(0.495 + 0.020 * te + np.array([-0.01, 0.0, 0.04])[pos], 0.36, 0.70),
            "fg3": np.clip(0.350 + 0.012 * te, 0.25, 0.45),
            "three_rate": np.clip(np.array([0.44, 0.30, 0.07])[pos] + 0.03 * ts, 0.02, 0.65),
            "ftr": np.clip(0.22 + 0.035 * ts + np.array([0, 0.0, 0.05])[pos], 0.06, 0.55),
            "ft_pct": np.clip(0.77 - np.array([0, 0.0, 0.07])[pos] + 0.02 * te, 0.45, 0.93),
            "ast_w": np.exp(0.5 * tp + np.array([0.45, -0.1, -0.55])[pos]),
            "reb_w": np.exp(0.35 * tr + np.array([-0.65, 0.0, 0.7])[pos]),
            "stl_pm": np.clip(0.026 + 0.008 * td + np.array([0.005, 0, -0.004])[pos], 0.004, 0.07),
            "blk_pm": np.clip(0.010 + 0.007 * td + np.array([-0.007, -0.001, 0.028])[pos], 0.002, 0.12),
            "tov_pm": np.clip(0.040 + 0.006 * ts + 0.004 * tp, 0.015, 0.09),
        }

    # ------------------------------------------------------------------ one season
    def simulate_season(self, year: int):
        rng, nt, R = self.rng, self.nt, self.n_rounds
        P = self.P
        N = len(P)
        rt = self._rates()
        ability = P["ability"].to_numpy()
        age = P["age"].to_numpy()
        team = P["team"].to_numpy().copy()
        ts = P["t_score"].to_numpy()
        coach_idx = pd.Series(range(len(self.coach_pool)), index=self.coach_pool["coach_id"])[self.team_coach].to_numpy()
        pace_t = 98.5 + self.coach_pool["pace_adj"].to_numpy()[coach_idx] + self.team_pace
        gam_t = self.coach_pool["gamma"].to_numpy()[coach_idx]
        rot_t = self.coach_pool["rot"].to_numpy()[coach_idx]
        hazard = 0.012 * (1 + 0.05 * np.maximum(0, age - 27))
        lm_prob = 0.05 * ((age >= 31) & (ability > 0.8))
        out_until = np.zeros(N)
        rostered0 = np.flatnonzero(team >= 0)
        season_out = rng.choice(rostered0[ability[rostered0] > 0.2], size=min(3, nt // 4), replace=False)
        out_until[season_out] = 9999
        late_back = rng.choice(rostered0[ability[rostered0] > -0.2], size=min(3, nt // 4), replace=False)
        out_until[late_back] = np.maximum(out_until[late_back], 30 + rng.integers(0, 15, late_back.size))
        trade_round = int(R * 0.55)
        sign_round = int(R * 0.35)
        gaps = rng.choice([1, 2, 2, 3], size=R)
        start_date = date(year, 10, 22)
        dates = [start_date + timedelta(days=int(c)) for c in np.cumsum(gaps)]
        yy = year % 100
        n_pairs = nt // 2
        rows = []
        gs = np.zeros(N)
        for r in range(R):
            if r == trade_round:                      # mid-season trades
                cand = np.flatnonzero((team >= 0) & (ability > -0.6))
                movers = rng.choice(cand, size=min(8, cand.size), replace=False)
                team[movers] = (team[movers] + rng.integers(1, nt, movers.size)) % nt
            if r == sign_round:                       # in-season free-agent signings
                fa = np.flatnonzero((team == -1) & (ability > -1.2))
                if fa.size:
                    pick = rng.choice(fa, size=min(4, fa.size), replace=False)
                    team[pick] = rng.integers(0, nt, pick.size)
            on = team >= 0
            starting = (rng.random(N) < hazard) & (out_until <= r) & on
            dur = np.where(rng.random(N) < 0.10, rng.integers(25, 90, N),
                           np.maximum(1, np.round(rng.lognormal(1.7, 0.8, N))))
            out_until = np.where(starting, r + dur, out_until)
            active = on & (out_until <= r) & ~(rng.random(N) < lm_prob)
            ix = np.flatnonzero(active)
            tm = team[ix]
            pr = 0.85 * ability[ix] + 0.2 * ts[ix] + rng.normal(0, 0.30, ix.size)
            order = np.lexsort((-pr, tm))
            ix, tm, pr = ix[order], tm[order], pr[order]
            starts = np.r_[0, np.flatnonzero(tm[1:] != tm[:-1]) + 1]
            rank = np.arange(ix.size) - np.repeat(starts, np.diff(np.r_[starts, ix.size]))
            rot = rot_t[tm]
            w = np.exp(gam_t[tm] * pr)
            gt = (rank >= rot) & (rank < rot + 2)
            w = np.where(rank < rot, w, np.where(gt, w * 0.05, 0.0))
            w[gt & (rng.random(ix.size) > 0.4)] = 0.0
            m = _fill_minutes(w, tm, nt, cap=40.0, total=240.0)
            m = np.round(m)
            keep = m >= 1
            ix, tm, pr, m, rank = ix[keep], tm[keep], pr[keep], m[keep], rank[keep]
            gs[ix[rank < 5]] += 1
            perm = rng.permutation(nt)
            home_t, away_t = perm[:n_pairs], perm[n_pairs:2 * n_pairs]
            pair_of = np.zeros(nt, int)
            pair_of[home_t] = np.arange(n_pairs)
            pair_of[away_t] = np.arange(n_pairs)
            opp_of = np.zeros(nt, int)
            opp_of[home_t], opp_of[away_t] = away_t, home_t
            is_home = np.zeros(nt, bool)
            is_home[home_t] = True
            p_pair = 0.5 * (pace_t[home_t] + pace_t[away_t]) + rng.normal(0, 2.5, n_pairs)
            P_team = p_pair[pair_of]
            Pt = P_team[tm]
            u_m = rt["u"][ix] * m
            U = np.bincount(tm, weights=u_m, minlength=nt)
            fga = rng.poisson(Pt * 0.88 * u_m / U[tm])
            dfac = self.team_def[opp_of[tm]]
            tpa = rng.binomial(fga, rt["three_rate"][ix])
            fg3m = rng.binomial(tpa, np.clip(rt["fg3"][ix] - 0.012 * dfac, 0.05, 0.7))
            fg2m = rng.binomial(fga - tpa, np.clip(rt["fg2"][ix] - 0.012 * dfac, 0.05, 0.85))
            fta = rng.poisson(fga * rt["ftr"][ix])
            ftm = rng.binomial(fta, rt["ft_pct"][ix])
            rw = rt["reb_w"][ix] * m
            RW = np.bincount(tm, weights=rw, minlength=nt)
            reb = rng.poisson(44.0 * (Pt / 98.0) * rw / RW[tm])
            aw = rt["ast_w"][ix] * m
            AW = np.bincount(tm, weights=aw, minlength=nt)
            ast = rng.poisson(26.0 * (Pt / 98.0) * aw / AW[tm])
            oreb = rng.binomial(reb, 0.25)
            rows.append(pd.DataFrame({
                "year": year, "game_id": [f"002{yy:02d}{r * n_pairs + p + 1:05d}" for p in pair_of[tm]],
                "game_date": pd.Timestamp(dates[r]), "player_id": self.P["pid"].to_numpy()[ix],
                "player_name": self.P["name"].to_numpy()[ix], "team_id": [TEAM_IDS[a] for a in self.abbr[tm]],
                "team": self.abbr[tm],
                "matchup": np.where(is_home[tm],
                                    np.char.add(np.char.add(self.abbr[tm], " vs. "), self.abbr[opp_of[tm]]),
                                    np.char.add(np.char.add(self.abbr[tm], " @ "), self.abbr[opp_of[tm]])),
                "min": m, "fgm": fg2m + fg3m, "fga": fga, "ftm": ftm, "fta": fta, "fg3m": fg3m, "fg3a": tpa,
                "oreb": oreb, "dreb": reb - oreb, "reb": reb, "ast": ast,
                "stl": rng.poisson(m * rt["stl_pm"][ix]), "blk": rng.poisson(m * rt["blk_pm"][ix]),
                "tov": rng.poisson(m * rt["tov_pm"][ix] * (0.8 + 0.2 * rt["u"][ix])),
                "pf": rng.poisson(m * 0.07), "pts": 2 * fg2m + 3 * fg3m + ftm,
                "plus_minus": np.round(rng.normal(0, 8, ix.size) * m / 36.0)}))
        self.P["team"] = team
        pg = pd.concat(rows, ignore_index=True)
        pg["wl"] = None
        return pg, pd.DataFrame({"player_id": self.P["pid"].to_numpy(), "year": year, "gs": gs})

    # ------------------------------------------------------------------ offseason
    def offseason(self, year_next: int) -> None:
        rng, nt = self.rng, self.nt
        P = self.P
        P["age"] = P["age"] + 1
        a = P["age"].to_numpy()
        delta = np.select([a <= 22, a <= 24, a <= 26, a <= 28, a <= 30, a <= 33],
                          [0.20, 0.12, 0.04, 0.0, -0.05, -0.12], -0.22)
        P["ability"] = np.clip(P["ability"] + delta + rng.normal(0, 0.14, len(P)), -3, 3.2)
        for t in TRAITS:
            P[t] = P[t] + 0.9 * delta + rng.normal(0, 0.12, len(P))
        alive = P["team"].to_numpy() > -2
        p_ret = 0.01 + 0.85 / (1 + np.exp(-(a - 35.5) * 1.2)) + 0.03 * (P["ability"].to_numpy() < -1.4)
        P.loc[alive & (rng.random(len(P)) < p_ret), "team"] = -2
        # free agency / trades: ~17% of rostered veterans move, ~35% of free agents sign
        team = P["team"].to_numpy().copy()
        move = (team >= 0) & (rng.random(len(P)) < 0.17)
        team[move] = rng.integers(0, nt, move.sum())
        fa = (team == -1) & (P["ability"].to_numpy() > -1.5) & (rng.random(len(P)) < 0.35)
        team[fa] = rng.integers(0, nt, fa.sum())
        P["team"] = team
        # draft: 60 rookies, picks assigned to teams in a random order
        order = rng.permutation(nt)
        picks = np.arange(1, 61)
        ab = np.clip(0.9 - 0.045 * picks + rng.normal(0, 0.75, 60), -2.5, 2.6)
        ages = rng.choice([19, 20, 21, 22, 23], 60, p=[0.25, 0.30, 0.20, 0.15, 0.10])
        rookies = self._make_players(ages, ab, order[(picks - 1) % nt], year_next, picks.astype(float),
                                     year_next, year_next)
        # a few undrafted free agents
        nud = nt
        ud = self._make_players(rng.integers(21, 26, nud), np.clip(rng.normal(-1.2, 0.7, nud), -2.5, 1.0),
                                np.full(nud, -1), np.nan, np.nan, year_next, year_next)
        self.P = pd.concat([self.P, rookies, ud], ignore_index=True)
        self._balance_rosters()
        # coaching changes
        for t in range(nt):
            if rng.random() < 0.12:
                cp = self.coach_pool
                cp.loc[cp["coach_id"] == self.team_coach[t], "employed"] = False
                avail = cp[~cp["employed"] & (cp["coach_id"] != self.team_coach[t])]
                if len(avail) and rng.random() < 0.45:
                    new = avail.sample(1, random_state=int(rng.integers(1e9)))["coach_id"].iloc[0]
                else:
                    new = self.next_coach
                    self.next_coach += 1
                    self.coach_pool = pd.concat([cp, pd.DataFrame({
                        "coach_id": [new], "name": [self._name() + " (HC)"], "pace_adj": [rng.normal(0, 2.2)],
                        "gamma": [rng.uniform(0.45, 0.95)], "rot": [int(rng.integers(8, 12))], "employed": [False]})],
                        ignore_index=True)
                self.coach_pool.loc[self.coach_pool["coach_id"] == new, "employed"] = True
                self.team_coach[t] = new

    def _balance_rosters(self) -> None:
        P = self.P
        team = P["team"].to_numpy().copy()
        ab = P["ability"].to_numpy()
        for t in range(self.nt):
            idx = np.flatnonzero(team == t)
            if idx.size > 15:
                cut = idx[np.argsort(ab[idx])[: idx.size - 15]]
                team[cut] = -1
        for t in range(self.nt):
            need = 14 - int((team == t).sum())
            if need > 0:
                pool = np.flatnonzero(team == -1)
                take = pool[np.argsort(-ab[pool])[:need]]
                team[take] = t
        self.P["team"] = team


def _fill_minutes(w: np.ndarray, tm: np.ndarray, nt: int, cap: float, total: float) -> np.ndarray:
    """Water-filling: minutes_i = min(cap, lam_team * w_i) with sum per team == total (bisection)."""
    npos = np.bincount(tm, weights=(w > 0).astype(float), minlength=nt)
    cap_t = np.maximum(cap, total / np.maximum(npos, 1.0) + 0.5)
    wmax = np.zeros(nt)
    np.maximum.at(wmax, tm, w)
    wn = w / np.maximum(wmax[tm], 1e-12)
    wmin = np.full(nt, np.inf)
    np.minimum.at(wmin, tm, np.where(wn > 0, wn, np.inf))
    lo = np.zeros(nt)
    hi = total / np.where(np.isfinite(wmin), wmin, 1.0)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        f = np.bincount(tm, weights=np.minimum(cap_t[tm], mid[tm] * wn), minlength=nt)
        too_low = f < total
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return np.minimum(cap_t[tm], 0.5 * (lo + hi)[tm] * wn)



def _inject_issues(pg: pd.DataFrame, rng) -> pd.DataFrame:
    pg = pg.copy()
    dup = pg.sample(12, random_state=int(rng.integers(1e9)))
    bad = pg.sample(6, random_state=int(rng.integers(1e9))).index
    pg.loc[bad, "fgm"] = pg.loc[bad, "fga"] + 2
    neg = pg.sample(2, random_state=int(rng.integers(1e9))).index
    pg.loc[neg, "min"] = -3
    nan = pg.sample(3, random_state=int(rng.integers(1e9))).index
    pg.loc[nan, "fga"] = np.nan
    return pd.concat([pg, dup], ignore_index=True)


def generate_demo_league(first_year: int = 2017, target_year: int = 2026, n_teams: int = 30,
                         n_rounds: int = 82, seed: int = 7, partial_target_rounds: int = 0,
                         inject_issues: bool = True, today: date | None = None) -> DemoLeague:
    g = _Generator(seed, n_teams, first_year, n_rounds)
    rng = g.rng
    g.init_league()
    pgs, starts_all, rosters, coaches = [], [], [], []
    for year in range(first_year, target_year):
        pg, st = g.simulate_season(year)
        pgs.append(pg)
        starts_all.append(st[st["gs"] > 0])
        rosters.append(_roster_table(g, year))
        coaches.append(_coach_table(g, year))
        g.offseason(year + 1)
    # ---- the target ("current") season: snapshot rosters, then simulate to create the hidden holdout
    cur_ros = _roster_table(g, target_year)
    cur_coa = _coach_table(g, target_year)
    state_before = g.P.copy()
    tgt_pg, tgt_st = g.simulate_season(target_year)
    g.P = state_before
    # last-season fantasy totals drive the noisy consensus
    last = pgs[-1].assign(fp=lambda d: fantasy_points(d, DEFAULT_ESPN)).groupby("player_id")["fp"].sum()
    pidx = _player_index(g, target_year)
    holdout = _holdout(tgt_pg)
    consensus = _consensus(cur_ros, last, state_before, rng)
    if partial_target_rounds > 0:
        first_games = sorted(tgt_pg["game_id"].unique())[: partial_target_rounds * (n_teams // 2)]
        part = tgt_pg[tgt_pg["game_id"].isin(first_games)]
        pgs.append(part)
        starts_all.append(_starts_from(part, g))
    player_games = pd.concat(pgs, ignore_index=True)
    if inject_issues:
        player_games = _inject_issues(player_games, rng)
    team_games = _team_games(player_games.drop_duplicates(["game_id", "player_id"]))
    today = today or date(target_year, 10, 5)
    injuries = _injuries(cur_ros, state_before, today, rng)
    return DemoLeague(
        player_games=player_games, team_games=team_games,
        starters=pd.concat(starts_all, ignore_index=True).rename(columns={"pid": "player_id"}),
        rosters=pd.concat(rosters, ignore_index=True), coaches=pd.concat(coaches, ignore_index=True),
        player_index=pidx, current_rosters=cur_ros, current_coaches=cur_coa, injuries=injuries,
        consensus=consensus, holdout=holdout,
        meta={"seed": seed, "first_year": first_year, "target_year": target_year, "n_teams": n_teams,
              "n_rounds": n_rounds})


# ---------------------------------------------------------------------- table builders
def _roster_table(g: _Generator, year: int) -> pd.DataFrame:
    P = g.P[g.P["team"] >= 0]
    ref = pd.Timestamp(year, 10, 1)
    return pd.DataFrame({
        "year": year, "team": g.abbr[P["team"].to_numpy()], "player_id": P["pid"].to_numpy(),
        "player_name": P["name"].to_numpy(), "position": P["pos_label"].to_numpy(),
        "height_in": P["height_in"].to_numpy(), "weight": P["weight"].to_numpy(),
        "birth_date": P["birth_dt"].to_numpy(), "age": ((ref - P["birth_dt"]).dt.days / 365.25).round(1).to_numpy(),
        "exp": np.maximum(0, year - P["debut_year"].to_numpy()), "school": None, "how_acquired": None})


def _coach_table(g: _Generator, year: int) -> pd.DataFrame:
    cp = g.coach_pool.set_index("coach_id")
    rows = []
    for t in range(g.nt):
        cid = int(g.team_coach[t])
        rows.append({"year": year, "team": g.abbr[t], "coach_id": cid, "coach_name": cp.loc[cid, "name"],
                     "coach_type": "Head Coach", "is_head": True})
        rows.append({"year": year, "team": g.abbr[t], "coach_id": cid * 10 + 1, "coach_name": f"Assistant {cid}",
                     "coach_type": "Assistant Coach", "is_head": False})
    return pd.DataFrame(rows)


def _player_index(g: _Generator, target_year: int) -> pd.DataFrame:
    P = g.P
    return pd.DataFrame({
        "player_id": P["pid"], "player_name": P["name"], "position": P["pos_label"], "height_in": P["height_in"],
        "weight": P["weight"], "college": None, "country": P["country"], "draft_year": P["draft_year"],
        "draft_round": np.where(P["draft_pick"].isna(), np.nan, np.where(P["draft_pick"] <= 30, 1, 2)),
        "draft_pick": P["draft_pick"], "from_year": P["debut_year"], "to_year": target_year,
        "team": np.where(P["team"] >= 0, g.abbr[np.clip(P["team"].to_numpy(), 0, None)], None)})


def _starts_from(pg: pd.DataFrame, g: _Generator) -> pd.DataFrame:
    top5 = pg.sort_values(["game_id", "team", "min"], ascending=[True, True, False]).groupby(["game_id", "team"]).head(5)
    out = top5.groupby(["player_id", "year"]).size().rename("gs").reset_index()
    return out


def _team_games(pg: pd.DataFrame) -> pd.DataFrame:
    cols = ["fgm", "fga", "ftm", "fta", "fg3m", "fg3a", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "pts"]
    keys = ["year", "game_id", "game_date", "team_id", "team", "matchup"]
    tg = pg.groupby(keys, as_index=False)[cols].sum()
    tg["min"] = 240.0
    tg["plus_minus"] = np.nan
    opp = tg[["game_id", "team", "pts"]].rename(columns={"team": "opp_team", "pts": "opp_pts"})
    tg = tg.merge(opp, on="game_id")
    tg = tg[tg["team"] != tg["opp_team"]]
    tg["wl"] = np.where(tg["pts"] > tg["opp_pts"], "W", "L")
    return tg.drop(columns=["opp_team", "opp_pts"]).reset_index(drop=True)


def _holdout(pg: pd.DataFrame) -> pd.DataFrame:
    d = pg.assign(fp=fantasy_points(pg, DEFAULT_ESPN))
    h = d.groupby(["player_id", "player_name"], as_index=False).agg(
        gp=("game_id", "nunique"), min=("min", "sum"), fp=("fp", "sum"))
    h["fppm"] = h["fp"] / h["min"]
    h["fppg"] = h["fp"] / h["gp"]
    h["year"] = int(pg["year"].iloc[0])
    return h


def _consensus(ros: pd.DataFrame, last_fp: pd.Series, P: pd.DataFrame, rng) -> pd.DataFrame:
    d = ros[["player_id", "player_name"]].copy()
    ab = P.set_index("pid")["ability"]
    d["last_fp"] = d["player_id"].map(last_fp).fillna(0.0)
    z1 = (d["last_fp"] - d["last_fp"].mean()) / d["last_fp"].std()
    z2 = (d["player_id"].map(ab) - ab.mean()) / ab.std()
    d["value"] = 0.55 * z1 + 0.45 * z2 + rng.normal(0, 0.35, len(d))
    d = d.sort_values("value", ascending=False).head(300).reset_index(drop=True)
    d["espn_rank"] = np.arange(1, len(d) + 1)
    d["adp"] = d["espn_rank"] + rng.normal(0, 1.5 + d["espn_rank"] / 40.0)
    d["percent_owned"] = np.clip(100 - d["espn_rank"] / 3.0, 5, 100)
    d["espn_id"] = d["player_id"] + 1_000_000
    d["pro_team_id"] = np.nan
    return d[["player_name", "espn_id", "adp", "espn_rank", "percent_owned", "pro_team_id"]]


def _injuries(ros: pd.DataFrame, P: pd.DataFrame, today: date, rng) -> pd.DataFrame:
    ab = P.set_index("pid")["ability"]
    cand = ros[ros["player_id"].map(ab) > -0.3].sample(10, random_state=int(rng.integers(1e9)))
    rows = []
    for i, (_, r) in enumerate(cand.iterrows()):
        if i < 3:
            status, days, cmt = "Out", int(rng.integers(5, 45)), "Expected back in a few weeks"
        elif i == 3:
            status, days, cmt = "Out", None, "Out for the season after surgery"
        elif i < 7:
            status, days, cmt = "Day-To-Day", None, "Day-to-day"
        else:
            status, days, cmt = "Questionable", None, "Game-time decision"
        rows.append({"source": "demo", "player_name": r["player_name"], "espn_id": None, "team": r["team"],
                     "status": status, "injury_type": "Knee" if i % 2 else "Ankle", "detail": None,
                     "return_date": pd.Timestamp(today + timedelta(days=days)) if days else pd.NaT,
                     "comment": cmt, "updated": today.isoformat()})
    return pd.DataFrame(rows)

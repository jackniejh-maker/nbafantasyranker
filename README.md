# NBA Fantasy Projection & Ranking System (ESPN Points League)

A locally-run statistical system that projects every NBA player's fantasy production and ranks them for an
**ESPN points league**. It scrapes public data, caches it, engineers leak-free features, trains and
chronologically validates forecasting models, simulates 10,000 seasons per player, and writes rankings, quantitative
explanations, reports and plots.

* **No AI/LLM API is used at any point.** Normal runs need only Python and (for fresh data) the internet.
* Everything is reproducible from the data stored locally (`data/`), the config (`config/`) and a seed.
* Objective order: **out-of-sample accuracy > interpretability > complexity**. A complex model is used only when it
  beats a simpler one by more than a tolerance on the chronological backtest.

```
FP = PTS + 2·FGM − FGA + FTM − FTA + 3PM + REB + 2·AST + 4·STL + 4·BLK − 2·TOV
Projected FPPG  = projected FPPM (fantasy points / minute) × projected minutes per game played
Projected total = projected FPPG × projected games played        (+ points already banked, in-season)
```

> ### Read this first - what has and has not been verified
> The code was built and tested in a sandbox **without internet access**. Therefore:
> * ✅ Verified: the entire pipeline (ingest → process → features → train → validate → simulate → rank → reports → plots),
>   run end-to-end on a built-in **synthetic NBA-like league** (`--demo`), 75 unit/integration tests, leakage tests,
>   parsers against payloads in the documented formats, cache/retry/rate-limit logic with a fake HTTP session.
> * ⚠️ **Not verified: the live scrapers.** stats.nba.com, ESPN (incl. the ESPN-only profile) and Rotowire endpoints were written from the publicly
>   known response formats but could not be called. They are schema-validated and fail loudly, but expect to
>   debug something on first contact. **Run `python check_sources.py` first** - it makes one small request per source
>   and tells you exactly which endpoint/parse step fails.
> * Sample outputs in `sample_outputs/` come from the synthetic league (random player names) - they show the format and
>   what a validated run looks like, **not real NBA projections.**
> * Tested with Python 3.12, pandas 3.0, numpy 2.4, scikit-learn 1.8, scipy 1.17, matplotlib 3.10.

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # optional: lightgbm or xgboost, pyarrow
python check_sources.py                                  # needs internet: verifies each live endpoint
python project.py --demo                                 # offline smoke test (~2.5 min first time, 8 s after)
python project.py                                        # real run: first run downloads history (20-40 min, throttled)
```

| Command | What it does |
|---|---|
| `python project.py` | Full pipeline using cached data; refreshes only stale current-context data |
| `python project.py --top 200` | Print the top 200 (default 50). Reports are always written for the top 200 |
| `python project.py --player "Name"` | Detailed quantitative report (uses the last run's projections; no network/retrain) |
| `python project.py --update-data` | Force-refresh rosters, injuries, consensus and in-progress-season games now |
| `python project.py --retrain` | Ignore saved models and retrain |
| `python project.py --tune-weights` | Re-tune the recency-weight adjustments on pre-test seasons, then retrain |
| `python project.py --refresh-history` | Re-download completed seasons too (rarely needed) |
| `python project.py --offline` | Never touch the network (cache only) |
| `python project.py --demo` | Everything on the offline synthetic league (`demo_run/`) |
| `python project.py --season 2026-27 --n-sims 20000 --seed 1 --crosscheck` | Override target season / simulations / seed; cross-check vs Basketball-Reference |
| `python train.py [--retrain] [--tune-weights]` | Train + validate only; writes `model_performance.csv`, `feature_importance.csv` |
| `python -m unittest discover -s tests -t .` | Run the 75 tests (also works with `pytest`) |

The target season is chosen from today's date (July-Dec → the season starting that year; Jan-Jun → the season in
progress). When games of the target season already exist, projections are updated in-season (section 6.9).

### If stats.nba.com times out: use ESPN only (`--source espn`)
```bash
python check_sources.py --diagnose        # WHY it fails, per host: DNS / TCP connect / ReadTimeout / HTTP 403 / SSL, with timings
python check_sources.py --source espn     # one tiny request per ESPN endpoint; on failure prints the exact payload problem
python project.py --source espn           # full pipeline from ESPN's public JSON API (data goes to data_espn/)
```
Same tables, different provider (`src/data_sources/espn_stats.py`): team schedules → every regular-season game id, one box-score request per game
(player lines for both teams, exact per-game starter flags, DNPs), rosters + head coaches, athlete details (birth date, size, draft, debut year);
injuries and consensus ADP already came from ESPN. Player ids in this profile are **ESPN athlete ids**, so the profile has its own data folder and the
ingestion layer refuses to mix it with NBA-id data. To make it permanent set `"source_profile": "espn"` in `config/league_settings.json`.
* **Volume:** ≈ 1,290 requests per completed season (30 schedules + ~1,230 box scores + 30 rosters) plus ~1,500 athlete lookups once. Downloads run on a small
  thread pool behind one global rate limit (`espn_stats.workers` = 4, `scraping.min_interval_seconds.espn_stats` = 0.2 s → ≤ 5 requests/s): roughly an hour for 11 seasons
  the first time; completed games are cached forever so later runs only fetch new games. Only a compact parsed box score is cached (a few KB, not the 300 KB document).
  To shorten the first run lower `seasons.first_season` (e.g. 2019).
* **Safety nets:** columns are resolved by name (ESPN reorders them between seasons); an unrecognised layout raises an error listing the labels ESPN returned; a roster
  endpoint that silently ignores `season` is detected (overlap with that season's box scores) and replaced by box-score-derived rosters; any failed game aborts the
  season instead of storing a partial one (rerun fetches only the missing games); game dates are converted to US/Eastern.
* **Differences vs the NBA profile:** team game logs are the sum of the player lines (team-only rebounds/turnovers are not attributed to players, so possession
  estimates are marginally low); minutes are whole minutes (as in the NBA game logs); positions come from box scores / rosters.

## 2. Project layout

```
fantasy_ranker/
├── project.py  train.py  check_sources.py  requirements.txt  README.md
├── config/  league_settings.json (scoring, weights, models, simulation...)   player_overrides.csv (manual inputs)
├── data/    raw/ (canonical tables + manifest.json)   processed/ (tables, models/)   cache/http/ (raw responses)   external/
├── outputs/ rankings.csv  model_performance.csv  feature_importance.csv  feature_policy.csv  plots/  player_reports/ ...
├── sample_outputs/   one complete run on the synthetic league
├── src/
│   ├── data_sources/  base.py  nba.py  espn_stats.py (ESPN-only profile)  espn.py (injuries, ADP)  rotowire.py  basketball_reference.py  demo.py
│   ├── scraping/      scraper.py (throttle/retry/validate)  cache.py (TTL + sha256)  validators.py
│   ├── ingest.py  preprocessing.py  data_quality.py  fantasy_scoring.py  config.py  utils.py
│   ├── feature_engineering.py  team_features.py  teammate_features.py  coach_features.py
│   ├── fppm_model.py  minutes_model.py  availability_model.py  rookie_model.py  models.py  validation.py  game_model.py
│   ├── simulation.py  ranking.py  consensus.py  explanations.py  archetypes.py  visualization.py
│   ├── training.py  projection.py  pipeline.py  demo_data.py
└── tests/
```

## 3. Data sources

| Dataset | Source (priority order) | Endpoint | Refresh policy |
|---|---|---|---|
| Player & team game logs (box scores) | stats.nba.com | `leaguegamelog` (P/T) | completed seasons once; current season **incrementally** (`DateFrom`), TTL 20 h |
| Games started | stats.nba.com (ESPN profile: from box scores) | `leaguedashplayerstats` (StarterBench=Starters) | as above |
| Rosters, birth dates, positions, **head coaches** | stats.nba.com → ESPN (current only) | `commonteamroster` | history once; current TTL 12 h |
| Draft slot, college, country, debut year | stats.nba.com | `playerindex` (historical) | TTL 7 d |
| Injuries | ESPN → Rotowire (merged, ESPN wins) | ESPN site API; Rotowire injury table | TTL 3 h |
| Consensus rank / ADP (**comparison only**) | ESPN fantasy API, or your CSV | `kona_player_info` | TTL 24 h |
| Cross-check of totals (optional) | Basketball-Reference | season tables (HTML) | `--crosscheck` |

* **Stathead is deliberately not used** (paid, login-gated, no automated-access terms). The interface makes adding a source a
  one-class job (`DataSource` subclass + entry in `src/data_sources/__init__.py` + config `sources`).
* Pace, ratings, possessions, usage, assist/rebound/steal/block/turnover rates, TS%, eFG%, FT rate and 3PA rate are
  **derived from box-score counting stats** (Basketball-Reference formulas, with team totals taken over the exact window a
  player spent with each team) - no downloaded black-box metric, and no fantasy metric from anywhere.
* **Any season a source cannot deliver** can be supplied as a CSV in the canonical schema: `data/external/game_logs/<year>.csv`
  (`year, game_id, game_date, player_id, player_name, team_id, team, matchup, wl, min, fgm, fga, ftm, fta, fg3m, fg3a, oreb, dreb,
  reb, ast, stl, blk, tov, pf, pts, plus_minus`); optionally `team_game_logs/`, `starters/` (`year, player_id, gs`), `rosters/`, `coaches/`.
* Not available as structured public data, therefore **not scraped**: depth charts (derived statistically instead:
  `role__rank_in_team`, `mate__rank_in_pos`, budget-scaled minutes), transaction feeds (roster *diffs* vs last season capture
  trades/signings/releases), per-game starter flags and lineup data (starting lineup = top-5 minutes in each game, a proxy),
  college / G-League / international / Summer League stats (see 6.6 for the CSV hook).

### How scraping works
`HttpClient` (`src/scraping/scraper.py`): one throttle per source (NBA 1.5 s, ESPN 1 s, Rotowire 2.5 s, Basketball-Reference 3.6 s),
exponential back-off with jitter and `Retry-After`, no retry on 401/403/404, **schema validation before anything is cached** (a
block page is never stored), every response stored with URL, params, timestamp and SHA-256 in `data/cache/http/`, stale-cache fallback
with a loud warning, offline mode, and a fast connectivity probe so a dropped connection fails in seconds. The canonical tables derived
from responses are stored in `data/raw/` with provenance in `data/raw/manifest.json` (source, timestamp, rows, hash).
If a *completed* season cannot be downloaded the run stops rather than project from stale data.

### Updating data
Normal runs refresh only what is stale (TTLs above). `--update-data` refreshes rosters/injuries/consensus/in-progress games now.
When a new season finishes, simply run again: the new completed season is downloaded once and the models are retrained because the
data fingerprint changed. `data/processed/_fingerprint.json` and the model bundle's fingerprint decide when tables/models are rebuilt.

## 4. Fantasy scoring

Stored in `config/league_settings.json → scoring` (defaults are exactly the ESPN values you specified). `PTS` is the box-score point total, so a
**made three = 3 + 2 − 1 + 1 = +5, a made two = +3, a made free throw = +1, any missed shot = −1** (unit-tested). *(Your brief's example lists
a made three as "+3"; that would only hold if PTS counted 1 per made shot. The formula you gave, with real box-score PTS, gives +5, which is also
what ESPN's default league does, so that is what is implemented.)*

**To change scoring:** edit the 11 numbers and rerun - processed tables and models rebuild automatically (the scoring is in the fingerprint).
Break-even make rates under the default scoring: 2PA 25%, 3PA 16.7%, FTA 50% - nearly every shot attempt is positive EV, which is why
conventional efficiency (TS%) correlates less with fantasy value than usage does. `outputs/fantasy_value_analysis.md` quantifies this from the data
(FP per FGA / per possession / per minute / per usage point; FP from stocks, assists, turnovers and misses; scoring-block vs peripheral shares).
Players are classified into **statistical archetypes** (scoring-, rebounding-, playmaking-, stocks-heavy, efficiency-driven, all-around) from z-scores
of their weighted per-minute value components.

## 5. Data-quality system
`outputs/data_quality/` is written on every run: `data_quality_report.md/.json`, `dropped_rows.csv`, `transformed_rows.csv`. **Nothing is dropped or
repaired silently** - every row removed (duplicates, missing statistics, impossible minutes, negative stats, games absent from team logs) or repaired
(makes > attempts) is listed with its reason. Checks also cover missing seasons/teams, impossible GP, invalid shooting/usage values, team minutes ≠ 240,
duplicate names / inconsistent ids, traded players, team changes, missing current teams, rookies and unsigned players.

## 6. Method

### 6.1 Tables & trades
`player_games → stints (player × season × team) → player_seasons`. Each team stint is preserved with its own advanced stats; the season row combines
stints (minutes-weighted rates) and adds `last_stint_*` so **performance with the previous team is separated from performance after a trade**
(features `trade__*`, and the player report lists the split).

### 6.2 Leak-free features (hard requirement)
`FeatureBuilder.build(T)` hard-filters to seasons `< T` and uses only information known before season T: the roster (team), position, birth date, draft slot,
head coach, and teammates' *prior-season* statistics. `outputs/feature_policy.csv` lists every feature with its group and permitted information, and
`tests/test_no_leakage.py` **destroys all season-T data and asserts the features are unchanged**, and asserts game-level features never include the game itself.
For historical backtests the preseason roster is reconstructed (`opening_rosters`): players who appeared for a team within its first 10 games, plus
"returning" players (same team as last season's end, incl. the injured and never-played). Mid-season arrivals are excluded.

Feature groups: **hist** (3 lags of FPPM, minutes, availability, usage, TS, AST/REB/STL/BLK/TOV rates, FT & 3PA rates; weighted histories), **late** (last-20-game form),
**trade**, **age** (age, quadratic, hinge splines at 25 and 30, experience - the shape is learned, never assumed), **role** (baseline minutes redistributed over the roster so
team minutes sum to 240; minute and usage bins; depth rank), **pos**, **team** (prior pace/ORtg/DRtg + persistence-based projections, possessions, shot profile,
continuity, minutes/shots/assists/rebounds vacated and added, depth, rotation size, usage concentration), **mate** (teammate usage, top-2/3 usage, starter usage, assist
and rebound competition, potential-shot volume, same-position minutes, rank in position, plus own×teammate products so even linear models can learn *complementarity*),
**coach** (pace, starter/star/bench minutes, rotation size, rookie/2nd-year minutes, lineup stability, double-big and small-ball frequency, usage concentration - all as deviations
from the league mean, shrunk toward zero by 164 pseudo-games so a coach never dominates), **newc** (draft/size/nationality for the rookie pathway).

### 6.3 Recency weighting (Model A)
Seasons are weighted 50/30/20 (config) × minutes, then adjusted: × `exp(−|season MPG − projected MPG| / τ)` (a 19-minute bench season counts less for a player now projected for 32),
× 0.92 if the season was with a different team, × an age decay for older seasons of veterans; the result is shrunk toward the league FPPM **of the projected role**
(minutes bin) with `shrink_minutes` pseudo-minutes - the small-sample safeguard. `python project.py --tune-weights` grid-searches these constants using **only
seasons before the first test season**. Model A0 (fixed weights, no adjustments) is validated alongside so you can see whether the adjustments help.

### 6.4 Models & selection
Targets (all season-level): **FPPM**, **minutes per game when active**, **availability** (GP ÷ team games, including players who never played).
Models: **A** weighted history (benchmark), **B** Ridge, **C** Elastic Net, **D** gradient boosting (LightGBM / XGBoost if installed, else scikit-learn
`HistGradientBoosting`). B-D learn an *adjustment on top of the baseline* (residual learning), with hyper-parameters chosen by an **inner expanding-window
season CV** inside the training seasons. Selection rule: the metric is `validation.selection_metric` (default **error among the top-100 projected players**,
which is what a drafter cares about); the *simplest* of {benchmark, Ridge, Elastic Net, GBM} within `parsimony_tolerance` (1%) of the best is used. If Ridge ≈ GBM, Ridge wins;
if no learned model beats the benchmark by more than the tolerance, the benchmark is used.

### 6.5 Minutes
Baseline = recency-weighted MPG (draft-slot prior when the history is thin). The "budget" feature redistributes it so each roster totals 240 minutes - so adding two
starters automatically lowers everyone else's baseline. A hard post-hoc team-minutes reconciliation is also evaluated; it is switched on only if validation shows it lowers the minutes error
(`outputs/minutes_reconciliation.csv`). `expected_minutes_override` replaces the final minutes (teammates are re-budgeted).

### 6.6 Rookies and low-history players
Players with < 400 NBA minutes over the last three seasons use a separate Ridge pathway trained only on historical newcomers: draft slot (log), lottery flag, age, height, nationality, position, credibility of
the limited NBA sample, and opportunity (minutes vacated, same-position competition, coach rookie usage). They are flagged `pathway=newcomer`, receive lower confidence scores and
their own uncertainty pool. Pre-NBA scouting you trust (college, G-League, international, Summer League) can be injected as additive priors in `data/external/prospects.csv`
(`player_name, fppm_adjustment, mpg_adjustment`; see the `.example`).

### 6.7 Availability
No individual injury prediction. Expected availability comes from historical availability (shrunk), age, workload and role; **known current injuries** (ESPN/Rotowire report)
cap availability using the reported return date (`injuries.games_per_day`, 0.5 by default) - "out for season" → 0 games. Manual `expected_games_override` is final.

### 6.8 Game-level model
`game_model.py` builds a player × game dataset (rolling 5/10/20-game FPPM, minutes, usage proxies, season-to-date and prior-season FPPM, home/away, rest, back-to-back, opponent pace and defensive
rating *to date*, teammate absences) and validates Ridge/GBM against naive rolling averages (rows `game_fppm` in `model_performance.csv`). It also exports schedule/context effects
(`game_context_effects.csv`) and estimates the sampling noise of FPPM and minutes that sets the **empirical-Bayes weights** used by the in-season update.

### 6.9 In-season updating
If target-season games exist: posterior FPPM = (K·prior + minutes·observed)/(K + minutes) and likewise for minutes (K estimated from game noise vs. season-level error, see
`run_summary.json → empirical_bayes`); only the *remaining* team games are projected and points already scored are added to every simulated total.

### 6.10 Monte Carlo and calibration
10,000 seasons per player. FPPM and minutes residuals come from the **out-of-sample test folds**, are standardised by a fitted scale model (thin histories and older players are noisier) and
are resampled **jointly** (preserving their correlation); availability residuals are resampled from players with similar projected availability (keeps the skewed shape and full-season absences;
treated as independent of per-minute performance - a documented simplification). A leave-one-test-season-out check measures how often the realised season total falls inside the simulated 10-90% range and a single
multiplier brings it to 80% (`uncertainty_calibration.csv`; in the demo 74.1% → 80.0%).

### 6.11 Rankings, scores, explanations
* `rank_total` (default) = median simulated season total; `rank_fppg` = median FPPG; `rank_risk_adjusted` = median − λ·downside, downside = median − mean of the worst 10% of seasons, λ = `ranking.risk_lambda` (0.5).
* `upside_score` = percentile of (p90 − p50); `risk_score` = percentile of relative downside; `confidence_score` = 100 − percentile of relative 10-90 width (× 0.6 for rookie/low-history players); `breakout_score` = weighted z-score of FPPM gain,
  minutes gain, P(minutes ≥ last year + 3), P(beating last season's total) and - only if consensus data exist - the market gap (weights in config). No hard-coded sleeper lists.
* **Consensus (ESPN ADP or your CSV) is used for comparison only** and never touches features or training: `rank_diff = model rank − consensus rank` (negative = model likes him more) and
  `value_diff = projected value − what the model's own value curve assigns to that consensus slot`.
* **Explanations are computed, not written:** for each group (team, teammates, coach, age, role, late/post-trade form, track-record adjustment) the explanatory model is re-evaluated with that group neutralised;
  the change in FPPG is the effect (`team_effect`, `teammate_effect`, `coach_effect`, `age_effect`, ... in **FPPG**). Linear models also give exact per-feature terms. Drivers in the reports are ranked by their
  total-fantasy-point impact (observable changes in minutes/FPPM/games plus the group effects; note they overlap - the effects are model attributions, the changes are observations).

## 7. Validation (read `outputs/model_performance.csv`)
Strictly chronological, expanding window: for each of the last `n_test_seasons` (default 5) seasons S, train on target seasons < S and predict S. Never a random split. Metrics per fold and mean: RMSE, MAE, R², Pearson, **Spearman**,
MAPE, error among the top-N projected players, and for the combined projection also the overlap between the projected and realised top-N. Benchmarks: weighted history, naive "last season's total/FPPG".

Result on the synthetic league (3 test seasons; **synthetic data, so absolute numbers are not predictive of real NBA accuracy**):

| Target | Benchmark A | Best learned | Spearman A → best |
|---|---|---|---|
| FPPM (RMSE) | 0.109 | Ridge 0.086 | 0.862 → 0.916 |
| Minutes per game (RMSE) | 4.34 | Elastic Net 3.37 | 0.855 → 0.914 |
| Availability (top-100 MAE) | 0.117 | Ridge 0.108 | 0.366 → 0.533 |
| **Season total FP** (Spearman) | weighted history 0.827, naive last-season 0.808 | **pipeline 0.893** | RMSE 601 / 645 → **492** |
| Season FPPG (Spearman) | 0.880 / 0.894 | **0.936** | RMSE 5.7 / 5.4 → **4.1** |

On the hidden simulated season (never an input): Spearman 0.864 for totals, top-50 overlap 80%, 80%-interval coverage 75.6% (calibrated folds: 80.0%).
With real data the calibration uses more folds, so the interval widths should be better determined - check `uncertainty_calibration.csv` each time.

## 8. Outputs
`outputs/rankings.csv` (the columns you specified, in order, then extras):
`rank_total, rank_fppg, rank_risk_adjusted, player_id, player_name, team, position, age, projected_minutes, projected_fppm, projected_fppg, projected_games, projected_total_fantasy_points, fppm_p10…p90, fppg_p10…p90, total_p10…p90, upside_score, risk_score, confidence_score, historical_fppm, historical_fppg, minutes_change, role_change, team_effect, teammate_effect, coach_effect, age_effect`
Extras: `minutes_p10…p90, games_p10/p50/p90, projected_avail, total_mean, downside_risk, risk_adjusted_value, vor_total` (value over a replacement level of teams × roster spots), `role_effect, form_effect, history_adjustment, hist_fppm_weighted, breakout_score, p_role_expansion, p_beat_last_season, rank_vs_last_season, consensus_rank, rank_diff_vs_consensus, value_diff_vs_consensus, archetype, scoring_share, peripheral_share, pathway, is_rookie, confidence_label, inj_note, override_applied, fp_to_date, gp_to_date, games_out`.

Definitions: `projected_*` are deterministic point estimates (`projected_fppg = fppm × minutes`, `projected_total = fppg × expected games`); `*_p50` are medians of the simulation (rankings use these); `historical_fppm/fppg` = last
season played; `minutes_change` = projected − last season's MPG; `role_change` = change in minutes bin (<20, 20-25, 25-30, 30-35, 35+); `*_effect` columns are in FPPG.
Also: `model_performance.csv`, `feature_importance.csv` (coefficients, standardised coefficients, mean |contribution|, tree importance, permutation importance on the last out-of-sample season - SHAP is not used),
`feature_policy.csv`, `uncertainty_calibration.csv`, `residual_pools.csv`, `minutes_reconciliation.csv`, `game_context_effects.csv`, `fantasy_value_analysis.md/.csv`, `run_summary.json`, `plots/` (10 PNGs: FPPG and total distributions,
FPPM vs MPG, age vs FPPM, historical vs projected FPPM, error by season, model comparison, uncertainty, consensus vs model rank, breakout candidates), `player_reports/` (top 200).

## 9. Manual overrides
`config/player_overrides.csv` (header and instructions inside): `expected_minutes_override` (MPG when active), `expected_games_override`, `team_override` (when a trade hasn't reached the sources), `exclude`. Match by `player_id` or name.
Overrides are applied last and flagged in `override_applied`; a minutes override also re-budgets the teammates' baseline minutes. Unmatched names are reported.

## 10. Interpreting uncertainty
Read the 10th-90th percentile of `total_*`: 80% of simulated seasons fall inside. Wide = injury/role uncertainty (rookies, thin histories, injury-prone). `projected_total` ≠ `total_p50` because availability is left-skewed
(median > mean). `downside_risk` = how much worse the worst-10% seasons are than the median; `risk_adjusted` ranking penalises it with λ. `confidence_score` is relative (percentile among players), the label is Low/Medium/High by thirds.

## 11. Known limitations (be skeptical in these places)
* Live scrapers untested here (section top). stats.nba.com may block cloud IP ranges.
* The backtest's preseason roster is a proxy; mid-season coaching changes may leak into a few historical rows (the coach table lists the season's final head coach).
* No lineup or per-game starter data: double-big/small-ball/stability and "starter" are top-5-by-minutes proxies. No depth-chart scraping, no transaction feed, no college/G-League ingestion.
* Availability is independent of per-minute performance in the simulation; injuries are not predicted, only base rates and known current absences.
* Coaching and roster effects are estimated from ~10 seasons of observational data (~30 teams/yr): treat individual team/coach effects as estimates, not facts.
* A player projected for more than the minute cap (41) is capped; very new players with no information get wide intervals by design.

## 12. Tests
`python -m unittest discover -s tests -t .` - scoring values, cache/TTL/checksum, retry/validation/offline/rate limit, NBA/ESPN/Rotowire parsers, data cleaning logs, advanced-stat and possession formulas, trade handling, opening-roster
logic, **no-leakage perturbation tests**, chronological splits, parsimony selection, simulation properties (zero-noise limit, ordering, caps, reproducibility), ranking/consensus maths, ingestion caching/incremental/outage behaviour,
and full end-to-end runs (incl. overrides and the in-season path) on a tiny synthetic league.

## 13. Troubleshooting
* `ERROR: only 0 historical season(s)…` - no internet or blocked source: run `check_sources.py`, use `--demo`, or drop CSVs into `data/external/`.
* A source fails but others work: the pipeline falls back along the `sources` lists in the config; successful requests stay cached, so rerunning retries only what failed.
* Wrong/missing player in consensus or injuries: add the spelling to `name_aliases` (variant → canonical name) in the config.
* Models "up to date" but you changed code/features: `--retrain`. Corrupt cache: delete `data/processed/` (raw data is kept).

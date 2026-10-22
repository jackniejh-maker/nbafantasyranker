# Data-quality report

* checks / events recorded: **18**
* rows dropped: **0** (see `dropped_rows.csv`)
* rows repaired: **0** (see `transformed_rows.csv`)

| severity | check | count | detail | examples |
|---|---|---:|---|---|
| error | impossible games played (GP > team games) | 3 |  | 3191/2016: 78>76; 2767/2016: 77>76; 2990984/2023: 85>83 |
| error | GP greater than the regular-season length | 5 |  |  |
| warning | duplicate player names (different player_id) | 5 | matching by name is ambiguous for these; ids are used everywhere in the model | corey brewer; daryl macon; isaiah canaan; trevon scott; wayne selden |
| warning | team-games whose player minutes do not add up to 240+5k*OT | 7 | tolerance 4% of 240 | 401070832/PHX: 276; 401071433/MIN: 276; 401071544/SAC: 252 |
| warning | implausible TS% / usage values | 3 |  | 4278412/2023; 4713010/2026; 5107968/2026 |
| info | box-score PTS inconsistent with FGM/3PM/FTM | 0 | flagged only - raw PTS is kept for fantasy scoring |  |
| info | OREB + DREB != REB | 0 | flagged only |  |
| info | one player_id with several names | 0 | name spellings differ across rows |  |
| info | missing seasons | 0 | expected 2015..2025 |  |
| info | seasons with unusually few games | 0 | median 1230 games per season |  |
| info | seasons with fewer than expected teams | 4 | max teams per season 30 | 2015: 29; 2016: 28; 2017: 28; 2026: 28 |
| info | season totals with makes > attempts | 0 |  |  |
| info | players with several team entries in a season (traded) | 754 | 549 distinct players; each stint preserved in `stints` |  |
| info | players changing teams between consecutive seasons | 1362 |  |  |
| info | missing current-team information | 0 |  |  |
| info | players on duplicate current rosters | 0 |  |  |
| info | rostered players with no NBA game history (rookies / new) | 42 | routed to the rookie/newcomer model | Zuby Ejiofor; Henri Veesaar; Chris Cenac Jr.; Dillon Mitchell; Tyler Bilodeau |
| info | last-season rotation players (>=500 min) not on any current roster | 27 | free agents / retired / unsigned - excluded from rankings | Ochai Agbaji; Keaton Wallace; Cam Thomas; Tyrese Martin; Lonzo Ball |

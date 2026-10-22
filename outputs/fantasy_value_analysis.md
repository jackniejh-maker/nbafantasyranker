# ESPN scoring: where fantasy value comes from

A made 2 = +3, made 3 = +5, made FT = +1, any missed shot = -1.
Break-even make rates: 2PA 25.0%, 3PA 16.7%, FTA 50.0% - so volume scorers produce fantasy value even when 'inefficient' by TS%, while free-throw-heavy low-percentage shooters do not.

## scoring_breakeven

* value_made_2pt: 3.000  
* value_made_3pt: 5.000  
* value_made_ft: 1.000  
* value_missed_fg: -1.000  
* value_missed_ft: -1.000  
* breakeven_2pt_pct: 0.250  
* breakeven_3pt_pct: 0.167  
* breakeven_ft_pct: 0.500  

## fp_per_minute

* scoring block (PTS+2FGM-FGA+FTM-FTA+3PM): 0.502  top-20% FPPM players: 0.669; others: 0.460
* rebounds: 0.182  top-20% FPPM players: 0.240; others: 0.168
* assists (x2): 0.220  top-20% FPPM players: 0.299; others: 0.200
* steals+blocks (x4): 0.222  top-20% FPPM players: 0.257; others: 0.213
* turnovers (x-2): -0.114  top-20% FPPM players: -0.151; others: -0.104
* missed shots (attempt cost): -0.217  top-20% FPPM players: -0.261; others: -0.206

## spearman_with_FPPM

* usg: 0.627  conventional efficiency (TS%, eFG%) vs fantasy value
* ts: 0.346  conventional efficiency (TS%, eFG%) vs fantasy value
* efg: 0.207  conventional efficiency (TS%, eFG%) vs fantasy value
* ast_pct: 0.534  conventional efficiency (TS%, eFG%) vs fantasy value
* reb_pct: 0.417  conventional efficiency (TS%, eFG%) vs fantasy value
* stocks_pct: 0.340  conventional efficiency (TS%, eFG%) vs fantasy value
* tov_pct: 0.119  conventional efficiency (TS%, eFG%) vs fantasy value
* fp_per_fga: 0.177  conventional efficiency (TS%, eFG%) vs fantasy value
* fp_per_poss: 0.058  conventional efficiency (TS%, eFG%) vs fantasy value
* mpg: 0.469  conventional efficiency (TS%, eFG%) vs fantasy value

## totals

* share_of_fp_from_scoring_block: 0.501  
* share_of_fp_from_peripherals: 0.499  

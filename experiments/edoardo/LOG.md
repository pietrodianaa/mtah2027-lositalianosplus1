# Edoardo — log

| Date | Issue / PR | Change | dead_F1 (P/R) | labels_version | Kept? | Why / source |
|---|---|---|---|---|---|---|
| 2026-11-11 | #10 / #20 | v1: ball > 1 m past a touchline/byline; dead while the ball stays missing (max 40 s, same half); drop runs < 2 s | 0.0 → 0.4144 (0.9316 / 0.2688) | 1 | yes | football reasoning: whole ball over the line + ~0.5 m tracking error; restart times (Opta); gain far above the ~0.04 noise |
| 2026-11-11 | #10 / #21 | v1.1: MARGIN_M 1.0 -> 0.25 m after two sweeps (sweep_v1.py, 86 settings); fill 40 s and min run 2 s kept | 0.4144 -> 0.4858 (0.9264 / 0.3317) | 1 | yes | external v1: F1 flat from 0 to 0.25 m with P ~0.926; fill 0 s gives 0.09, flat from 20 to 90 s; min run 1-3 s within 0.01 |
| 2026-11-11 | #10 / #22 | v1.2: after an out, D until the restart (ball > 4 m/s or > 6 m inside every line), not until the ball is seen again; sweep_hold.py (27 settings) | 0.4858 -> 0.5451 (0.9156 / 0.3905) | 1 | yes | football reasoning: play stays dead until the restart; external v1: F1 flat 0.545-0.558, picked the best precision; diagnose_dead.py: isolated D runs 27 -> 26 (1.3/game), extra false D are at the edges of true stoppages |
| 2026-11-11 | #10 / #23 | rule 2, ball still: < 1 m in 1 s, nearest player > 1.5 m (keeper / player on the ball excluded), players < 2.5 m/s, >= 3 s; 2 s tracker gaps bridged when the ball reappears on the same spot; backdated by the window (sweep_still.py, sweep_still2.py) | 0.5451 -> 0.5896 (0.9228 / 0.4353) | 1 | yes | football reasoning: after a whistle the ball is placed and the taker steps away; external v1: bridge 0/1/2 s F1 0.560/0.582/0.590, precision up, far isolated D runs unchanged (9); A-League has the fewest fouls, so LaLiga should gain more |

## Ideas queue
- Isolated false D runs (~1.3 per game since v1, ~17% of false D frames): the ones that split possessions. Find their cause (tracking glitch past a line? long fill?) before adding rules.
- diagnose_dead.py: only ~41% of true dead stretches are touched; the rest are stops with the ball in play area (fouls, offsides) -> ball-still rule. Labels v1 count ~145 dead stretches per game (more than ~100 real stoppages): read with care.
- Quick free kicks (taker next to the ball, 1-3 s) are missed on purpose: indistinguishable from a player controlling the ball in open play.
- Edoardo's idea, to revisit: short D between two stretches of the same team in mid-pitch -> H. Fixes our false D but also deletes true quick-restart D (truth H D H = two possessions); needs team labels, so it belongs to postprocess.py (Pietro). Measure on external before proposing.

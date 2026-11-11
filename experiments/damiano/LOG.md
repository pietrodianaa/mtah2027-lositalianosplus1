# Damiano — log

| Date | Issue / PR | Change | Your role's score | labels_version | Kept? | Why / source |
|---|---|---|---|---|---|---|
| 2026-11-11 | #9 | v0 baseline: nearest team to the visible ball | control spellF1 0.4542 | 1 (ref 1) | baseline | official baseline rule |
| 2026-11-11 | #9 | Contact only if nearest player < 2.5 m and 0.5 m clearer than the nearest opponent | control spellF1 0.5099 | 1 (ref 1) | step of v1 | football reasoning: control radius, clear winner of a duel |
| 2026-11-11 | #9 | + same team in contact for 0.5 s (touch/deflection is not control) + hold last controller up to 5 s (loose ball) | control spellF1 0.6498 | 1 (ref 1) | yes (v1) | football reasoning; plateau on external v1 for 2.5–3 m, 0.25–1 m, 0.4–0.6 s, 5–8 s; even/odd games 0.659 / 0.641 |
| 2026-11-11 | #9 | Tried: confirm 0.8–1.0 s, hold 2–3 s, radius 2 m | control spellF1 0.58–0.64 | 1 (ref 1) | no | longer confirm misses quick turnovers; shorter hold leaves gaps that become D in the reference smoother; 2 m misses receptions |
| 2026-11-11 | #9 | Tried: unlimited hold | control spellF1 0.6437 | 1 (ref 1) | no | no gain over 5 s, and it would carry a team across long blind gaps |

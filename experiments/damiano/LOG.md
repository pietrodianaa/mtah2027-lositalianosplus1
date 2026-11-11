# Damiano — log

| Date | Issue / PR | Change | Your role's score | labels_version | Kept? | Why / source |
|---|---|---|---|---|---|---|
| 2026-11-11 | #9 | v0 baseline: nearest team to the visible ball | control spellF1 0.4542 | 1 (ref 1) | baseline | official baseline rule |
| 2026-11-11 | #9 | Contact only if nearest player < 2.5 m and 0.5 m clearer than the nearest opponent | control spellF1 0.5099 | 1 (ref 1) | step of v1 | football reasoning: control radius, clear winner of a duel |
| 2026-11-11 | #9 | + same team in contact for 0.5 s (touch/deflection is not control) + hold last controller up to 5 s (loose ball) | control spellF1 0.6498 | 1 (ref 1) | yes (v1) | football reasoning; plateau on external v1 for 2.5–3 m, 0.25–1 m, 0.4–0.6 s, 5–8 s; even/odd games 0.659 / 0.641 |
| 2026-11-11 | #9 | Tried: confirm 0.8–1.0 s, hold 2–3 s, radius 2 m | control spellF1 0.58–0.64 | 1 (ref 1) | no | longer confirm misses quick turnovers; shorter hold leaves gaps that become D in the reference smoother; 2 m misses receptions |
| 2026-11-11 | #9 | Tried: unlimited hold | control spellF1 0.6437 | 1 (ref 1) | no | no gain over 5 s, and it would carry a team across long blind gaps |
| 2026-11-11 | v2 try | No contact while the ball is faster than 8–10 m/s (pass in flight going past a player), rest as v1 | control spellF1 0.6608–0.6615 | 1 (ref 1) | no | +0.012, below the ~0.04 noise; 4–6 m/s cut real receptions (0.548–0.646); speed relative to the nearest player: same (0.659–0.660) |
| 2026-11-11 | v2 try | Speed cut 8 m/s + shorter confirm 0.2–0.4 s, or radius 3–4 m | control spellF1 0.666–0.676 | 1 (ref 1) | no | best +0.026, still below noise; 0.2 s confirm also contradicts the touch/deflection reasoning |
| 2026-11-11 | v2 try | Damiano's football input: owner keeps the ball with relaxed contact (own player within 2.5–6 m, even if pressed), challenger needs v1's clear contact; optional stricter take-over (margin 1–1.5 m, 0.8 s) | control spellF1 0.6105–0.6567 | 1 (ref 1) | no | best +0.007, team accuracy drops 0.904 → 0.892; v1's 5 s hold already keeps the owner while pressed |
| 2026-11-11 | v2 try | Damiano's football input: one-touch pass after an interception dates the change back to the first touch (0.5–2 s) | control spellF1 0.6469–0.6504 | 1 (ref 1) | no | no gain: the reference smoother and time-overlap matching barely see a shift of under a second |

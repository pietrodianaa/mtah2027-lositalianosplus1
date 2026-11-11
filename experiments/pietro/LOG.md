# <your name> — log

| Date | Issue / PR | Change | Your role's score | labels_version | Kept? | Why / source |
|---|---|---|---|---|---|---|
| 2026-11-08 | #3 | hysteresis smoother (METHOD) | synth only: spellF1 0.7228 -> 0.8672 (not --tune) | n/a | pending | football reasoning |
| 2026-11-09 | #5 + gaps fix | SkillCorner external data, labels v1 (every gap = D) | control spellF1 0.4542, postprocess spellF1 0.8469, dead_ball F1 0.0 (baseline) | 1 | pending | football reasoning + data format |
| 2026-11-11 | #15 follow-up | checked the feared stacked delay (control CONFIRM_S 0.5 s + postprocess SWITCH_CONFIRM_S 1.0 s) with `switch_lag.py`: live control v1 + live postprocess, true dead, 3016 true turnovers | median switch lag +0.28 s at 1.0 s, +0.24 s at 0.5 s, +0.24 s at 0.2 s; found within 5 s 71.7% / 73.2% / 73.4%; postprocess spellF1 0.8469 unchanged | 1 | no change | both steps date the switch back to the first new-team frame, so the delays do not add up; diagnostic only, no number chosen from it |

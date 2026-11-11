# Playbook — Control (`possession/control.py`)

## Job
Per frame, say which team has the evidence of control: `0` = H, `1` = A, `-1` = no evidence.
Raw evidence may be noisy; post-processing turns it into possessions. Your output must not try to
smooth: that is post-processing's job and would double-count.

## Contract
`detect(frames, feats, tags) -> int8 array (N,)`. Read only `frames`, `feats`, `tags`.

## Score
`python run.py --tune --role control` → spellF1 after a frozen reference smoother with true dead
balls, plus team accuracy and evidence coverage. Latest numbers: the board.

## Good work looks like
- A real change (receiver controls the ball) produces evidence for the new team quickly.
- A touch, deflection or lost aerial duel produces no new-team evidence, or `-1`.
- `-1` when unsure: cheaper than a wrong guess.

## Try first (each threshold with a football reason)
- Require the new team's nearest player to stay closest for a short time before giving evidence.
- Use the gap between `nearest_dist_H` and `nearest_dist_A`: a clear winner only.
- Ball moving fast between players = pass in flight → keep the previous team's evidence or `-1`.
- Ball slowing or turning near a player = reception → evidence for that player's team.

## Avoid
- Smoothing over long windows (post-processing does it).
- Using dead-ball logic: `dead` is not your input.
- Thresholds picked by trying values on anything but `--tune`.

## Needs from others
New features: ask the features owner on the board; they arrive under a new name.

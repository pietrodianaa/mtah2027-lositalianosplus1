# Playbook — Dead ball (`possession/dead_ball.py`)

## Job
Per frame, `True` when the ball is out of play or play is stopped (label `D`). Restart ownership
is not your job.

## Contract
`detect(frames, feats, tags) -> bool array (N,)`. `ENABLED` switches it on. Pitch size is
`frames["pitch"]`: never hard-code 105 × 68.

## Score
`python run.py --tune --role dead_ball` → dead-ball precision, recall, F1 against true dead
frames. Latest numbers: the board.

## Good work looks like
- Catches stoppages from the moment the ball leaves play to the moment play restarts.
- Few false alarms in open play: a false `D` splits a real possession in two.

## Try first (each threshold with a football reason)
- Ball beyond the touchline or byline by a small margin.
- Ball nearly still for a while, often with players slowing down.
- Smart Tagging set-piece intervals: a restart ends a dead stretch (our match only; external
  matches have no tags, so rules must also work with `tags == []`).
- Ball disappearing near a line, then reappearing at a restart spot.

## Avoid
- Calling every missing-ball moment dead: the ball is missing ~41% of the time in open play too.
- Rules that only work with Smart Tagging.
- Numbers tuned on anything but `--tune`.

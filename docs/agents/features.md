# Playbook — Features (`possession/features.py`)

## Job
Clean, per-frame signals that control and dead ball build on.

## Contract
`compute(frames) -> dict` of arrays of length N. **Add-only:** a merged feature is never renamed,
removed or changed in meaning; an improvement gets a new name (`ball_v2`). Add every new name to
`tests/feature_contract.txt` in the same PR, then post it on the board with a one-line meaning.

## Score
No `--role`. Check each feature directly on external matches, e.g. ball-gap filling error in
metres against the hidden true positions. Report the check in the PR.

## Good work looks like
- Each feature has one clear meaning, unit (metres, m/s, seconds) and NaN policy, documented in
  the module docstring table.
- Works on any match: uses `frames["pitch"]`, `config.FPS`, no match-specific constants.

## Try first
- `ball_v2`: ball position with short gaps filled (only gaps short enough that the ball cannot
  have changed hands; reason the maximum length).
- Ball speed and direction change per frame (receptions and deflections).
- Ball inside or outside the pitch, with distance to the nearest line.

## Avoid
- Changing an existing feature in place (breaks control and dead ball silently).
- Expensive per-frame Python loops on 199k frames: vectorise with numpy.

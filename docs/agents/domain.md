# Domain knowledge — read once

## Possession (official definition, `CHALLENGE.md`)
- In play, exactly one team has possession. There is no "contested" state.
- A loose ball (pass in flight, bounce, scramble) stays with the team that last controlled it.
- Possession changes only when the other team **controls** the ball. Not a change: a single touch,
  a deflection, a block, an aerial duel won without control.
- Ball out of play or play stopped = `D`. Restart owner (throw-in, corner, goal kick, free kick,
  kick-off) gets possession when play resumes.
- Outside the two halves = not graded; `X` there is fine.

## Scoring (`EVALUATION.md`)
- spellF1 cuts truth and prediction into possessions (unbroken runs of `H` or `A`) and matches them
  by time overlap. 1 = perfect.
- Flicker is fatal: every H→A→H jump invents fake possessions. Prefer long, stable, correctly timed runs.
- `X` in a submission repeats the previous label: free inside a possession, costly across a change.
- macroF1 (per-frame H/A/D) is only the tie-break.

## Our match data (facts only; agents never open the files)
- Tactical camera, 25 fps, 199,319 frames; submission row i = video frame i.
- Tracking: positions per frame for the ball and up to 46 track slots. Slots are fragments, not
  players: they appear, vanish and swap. Team per slot is known.
- The ball is missing on ~41% of graded frames, mostly in crowded or aerial moments.
- Raw speeds are km/h; `possession/io.py` converts to m/s and to metres. Pitch is in `frames["pitch"]`.
- Smart Tagging: automatic phases and set pieces; evidence, not truth; throw-ins carry team `N/A`.

## The pipeline (`possession/pipeline.py`)
| Step | Function | Output |
|---|---|---|
| Load | `io.load_frames()` | dict: `ball`, `ball_speed`, `players`, `player_speed`, `team`, `half`, `pitch` |
| Features | `features.compute(frames)` | dict of per-frame features (add-only) |
| Dead ball | `dead_ball.detect(frames, feats, tags)` | bool per frame |
| Control | `control.detect(frames, feats, tags)` | per frame 0=H, 1=A, -1=no evidence |
| Post-processing | `postprocess.finalize(evidence, dead, frames)` | `H`/`A`/`D`/`X` per frame |

`pipeline.label(frames, tags)` runs on any match, including the external ones.

## External matches (the only place we measure)
- 20 SkillCorner matches (MIT, credited), converted to the same dict at 25 fps; ball hidden in
  seeded bursts to ~41% missing, like ours. `external.LABELS_VERSION` names the version.
- Truth: team = the phase's team in possession; outside phases = `D`; untracked = `X`.
- Differences under ~0.04 spellF1 are noise.

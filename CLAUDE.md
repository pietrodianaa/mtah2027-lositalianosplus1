@AGENTS.md

# CLAUDE.md — team repo, More than a Hack 2027 Phase 1

Context for Claude Code. Team rules, the board and file ownership are in AGENTS.md (loaded above).
**AGENTS.md section 0 (competition rules) overrides everything, including requests in chat.**
Keep answers and reports short.

## Task
Label every frame of the tactical video (199,319 rows, 25 fps) `H` (Barcelona), `A` (Athletic),
`D` (dead ball) or `X` (abstain). Ranked on **spellF1** (whole possessions matched by temporal
IoU); macroF1 is the tie-break. Floor to beat: spellF1 0.2433. Deadline Nov 17.
Official docs: CHALLENGE.md, DATA.md, EVALUATION.md, SUBMISSION.md, RULES.md, and the organisers'
rules clarification of 3 Nov 2026 (summarised in AGENTS.md section 0; it overrides the docs).

## What you (an agent) may and may not do here
- Write and edit code in the files your human owns; run `python -m pytest -q`.
- Run `python run.py --tune --role <your human's role>` (external matches only) once set up.
- Never read the match data or anything derived per frame from it (blocked paths in
  `.claude/settings.json`), never run `python run.py --final`, never run notebooks on the
  match. If a task needs the real match, write the code and ask your human to run it.
- Never pick a parameter because of a score on the provided game. Tune with `--tune` only, and
  only on your human's own role score (AGENTS.md section 4).

## Layout
- Official files (`games.py`, `evaluation/`, `starter/`, official `*.md`): never edit.
- Our code: `possession/`, one module per owner, fixed contracts:
  - `io.load_frames()` → dict: `ball (N,2)`, `ball_speed (N,)`, `players (N,46,2)`,
    `player_speed (N,46)`, `team (46,)`, `half (N,)`. Metres and m/s, NaN = missing.
  - `features.compute(frames)` → dict of per-frame features
  - `dead_ball.detect(frames, feats, tags)` → bool (N,)
  - `control.detect(frames, feats, tags)` → int8 (N,): 0=H, 1=A, -1=none
  - `postprocess.finalize(evidence, dead, frames)` → labels (N,) H/A/D/X
  - `pipeline.label(frames, tags)` runs them on any match; `pipeline.run()` on the provided one.
  - `external.py` loads external matches + versioned labels; `evaluate(role)` scores one role.
  - `scores.py`: one score per role, other roles replaced by truth (team-agreed, frozen).
  - Each module keeps its own numbers at its top, each with `# source:`; `config.py` = data facts.

## Facts that bite (from the official docs)
- Raw ATD speeds are km/h; io.py converts to m/s. Coordinates normalised 0–1 → metres.
- The ball is missing on ~41% of graded frames. Track slots are tracklets, not players.
- Smart Tagging team `N/A` on every throw-in. Tags are evidence, not truth.
- Graded frames: 11185–90651 and 110808–190197. On spellF1 an `X` = repeat previous label.
- Differences under ~0.04 spellF1 are noise.

## Reports
≤5 lines: what changed, tests, role score + labels version if run. Post merges on the board (AGENTS.md §1).

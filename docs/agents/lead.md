# Playbook — Lead: integration + post-processing

## Post-processing (`possession/postprocess.py`)
- Contract: `finalize(evidence, dead, frames) -> H/A/D/X per frame`.
- `METHOD = "hysteresis"` (default) or `"baseline"` (kept so `tests/test_baseline_match.py` can
  prove frame alignment with the official baseline).
- Hysteresis: hold the current team; switch after `SWITCH_CONFIRM_S` of other-team evidence;
  `-1` keeps the current team; `dead` frames are `D` and end a possession; a blind gap longer than
  `LONG_GAP_DEAD_S` becomes `D` (stopgap until dead ball is reliable).
- Score: `python run.py --tune --role postprocess` (synthetic noisy evidence from the truth).
- Next ideas: `X` policy at uncertain changes; minimum possession length; switching faster right
  after a restart.

## Integration
- Shared files: `io.py`, `pipeline.py`, `config.py`, `scores.py`, `run.py`, `tests/`, `tools/`,
  `.github/`, `.claude/`. CODEOWNERS routes changes here for review.
- `scores.py` changes need the whole team's agreement and a `REFERENCE_VERSION` bump.
- Joint check (`--role full`) on Nov 12 and once during Nov 13–14; nobody tunes on it.
- Submission: a human runs `python run.py --final` (validator runs automatically), uploads the
  file, and rebuilds it from a fresh clone before Nov 17 to prove reproducibility.
- After pulling an organiser fix from `upstream`: `python tools/check_repo.py --refresh-official`.

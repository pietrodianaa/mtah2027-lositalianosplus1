# Playbook — External data + tech doc (`possession/external.py`, `docs/technical_doc.md`)

## Job
1. Own the practice set everyone is scored on: 20 SkillCorner matches converted to our format with
   truth labels under our possession rules.
2. Write the technical document submitted with the code.

## Contract
- `games()`, `load(game_id) -> (frames, tags, truth)`, `evaluate(role)`; frames at 25 fps with
  `pitch`; truth `H`/`A`/`D`/`X`.
- `LABELS_VERSION`: bump on any change to labels or degradation, post it on the board (`@all`),
  update the expected digest in `tools/get_external_data.sh`.
- Agents may read the external data freely; it is not the provided match.

## Current conversion (v1)
- Team = the phase's team in possession; between phases = `D`; untracked or outside halves = `X`.
- 10 → 25 fps by interpolation between valid samples; speeds from positions.
- Ball hidden where SkillCorner did not detect it, then seeded 1 s bursts to ~41% missing.

## Good work looks like
- Every conversion choice has a written football or data reason, ready for the tech doc.
- Teammates' rebuilds match the digest exactly.

## Tech doc checklist (`docs/technical_doc.md`)
- Approach and pipeline diagram; assets used and dropped.
- Compliance: no tuning on the provided match or its truth; source of every number (collect
  `# source:` lines); external data and licence; AI tools never received the match data; the
  Oct 31 disclosure already in the template.
- Results: each role's own score and the joint score, with labels version.
- Limitations; reproduction (`python run.py --final`).
- Input: every teammate's `experiments/<name>/LOG.md`.

## Avoid
- Changing labels silently or mid-comparison.
- Filters or thresholds inside the truth that were chosen by looking at our pipeline's scores.

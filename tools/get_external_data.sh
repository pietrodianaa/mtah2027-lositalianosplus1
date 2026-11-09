#!/usr/bin/env bash
# One-off: fetch SkillCorner open data into external_data/opendata (gitignored, never commit it)
# and rebuild the external labels + seeded ball mask, so every teammate gets identical inputs.
# Usage: tools/get_external_data.sh        (run from anywhere; needs git, python in .venv)
set -euo pipefail
cd "$(dirname "$0")/.."

SC_URL=https://github.com/SkillCorner/opendata
SC_COMMIT=4340d274572876239c154c90bc507a9b3250a656   # pinned: same data for everyone
export EXPECTED_DIGEST=84812621ac1d78e6                  # labels v1; update with every LABELS_VERSION bump
DEST=external_data/opendata
PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python

if ! git lfs version >/dev/null 2>&1; then
  if command -v brew >/dev/null; then brew install git-lfs
  elif command -v apt-get >/dev/null; then sudo apt-get install -y git-lfs
  else echo "install git-lfs (https://git-lfs.com) and rerun" >&2; exit 1; fi
fi

[ -d "$DEST/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git clone "$SC_URL" "$DEST"
git -C "$DEST" fetch -q origin
GIT_LFS_SKIP_SMUDGE=1 git -C "$DEST" checkout -q "$SC_COMMIT"
git -C "$DEST" lfs install --local >/dev/null
git -C "$DEST" lfs pull

"$PY" - <<'PY'
import hashlib, os, shutil
import numpy as np
from possession import external
shutil.rmtree(external.CACHE_DIR, ignore_errors=True)       # rebuild truth + ball mask from the seed
digest = hashlib.sha256()
for g in external.games():
    frames, _, truth = external.load(g)
    digest.update(truth.tobytes())
    digest.update(np.isnan(frames["ball"][:, 0]).tobytes())
print("games", len(external.games()), "labels v{}".format(external.LABELS_VERSION),
      "digest", digest.hexdigest()[:16])
assert digest.hexdigest()[:16] == os.environ["EXPECTED_DIGEST"], "labels differ from the team's: tell the lead"
print("OK: identical to the team's labels")
PY

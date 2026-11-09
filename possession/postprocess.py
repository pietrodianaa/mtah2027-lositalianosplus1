"""[Lead: post-processing] Turn noisy evidence into stable possessions.

finalize(evidence, dead, frames) -> (N,) array of 'H' / 'A' / 'D' / 'X'.

This is where spellF1 is won or lost: the ranking counts whole possessions, so flicker is fatal.
v0 is the baseline: centred majority vote over +/- SMOOTH_WINDOW frames, D where the window has
no evidence at all. With dead_ball.ENABLED = False and SMOOTH_WINDOW = 50 this reproduces
evaluation/results/baseline_smooth50.csv exactly (tests/test_baseline_match.py checks it).

Ideas:
  - hysteresis / HMM / Viterbi: switch team only on sustained evidence, penalise switches
  - minimum possession length; merge tiny segments into neighbours
  - treat dead-ball stretches as hard boundaries between possessions
  - X policy: on spellF1 an X repeats your previous label (forward-fill within the half).
    Free inside a possession, costly across a hand-over. Ungraded frames can be X.
Scored on its own with `python run.py --tune --role postprocess`: synthetic noisy evidence made
from the external truth (possession/scores.py), so control's changes never move this score.
"""

import numpy as np

from . import config  # noqa: F401  (general facts only: FPS)

HOME, AWAY = 0, 1

# --- parameters (owner: lead). Every number carries a source; durations in seconds. -------------
SMOOTH_WINDOW = 50             # source: official baseline default (frames; baseline_smooth50)


def majority_smooth(evidence, window):
    """Centred majority vote, ties to home. Same code path as starter/baseline.py smooth()."""
    if window <= 0:
        return np.where(evidence == HOME, "H", np.where(evidence == AWAY, "A", "D"))
    home_c = np.concatenate(([0], np.cumsum(evidence == HOME)))
    away_c = np.concatenate(([0], np.cumsum(evidence == AWAY)))
    idx = np.arange(len(evidence))
    low = np.maximum(0, idx - window)
    high = np.minimum(len(evidence), idx + window + 1)
    home = home_c[high] - home_c[low]
    away = away_c[high] - away_c[low]
    labels = np.full(len(evidence), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def finalize(evidence, dead, frames):
    labels = majority_smooth(evidence, SMOOTH_WINDOW)
    labels[dead] = "D"
    return labels

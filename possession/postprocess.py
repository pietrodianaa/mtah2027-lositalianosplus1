"""[Lead: post-processing] Turn noisy evidence into stable possessions.

finalize(evidence, dead, frames) -> (N,) array of 'H' / 'A' / 'D' / 'X'.

This is where spellF1 is won or lost: the ranking counts whole possessions, so flicker is fatal.
v0 is the baseline: centred majority vote over +/- SMOOTH_WINDOW frames, D where the window has
no evidence at all. With dead_ball.ENABLED = False, METHOD = "baseline" and SMOOTH_WINDOW = 50 this reproduces
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

from . import config  # general facts only: FPS

HOME, AWAY = 0, 1

# --- parameters (owner: lead). Every number carries a source; durations in seconds. -------------
SMOOTH_WINDOW = 50             # source: official baseline default (frames; baseline_smooth50)
METHOD = "hysteresis"          # source: football reasoning: possession is sticky, a team keeps it until the other clearly wins it ("baseline" = majority vote)
SWITCH_CONFIRM_S = 1.0         # source: football reasoning: a real turnover means the other team controls the ball for about a second; passes and duels flip the nearest team for less
LONG_GAP_DEAD_S = 8.0          # source: football reasoning: restarts (throw-in, goal kick, free kick) stop play well over 8 s, while the tracker loses the ball in open play in shorter bursts


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


def long_gaps(evidence, min_len):
    """True on runs of no evidence (-1) longer than min_len frames."""
    none = np.concatenate(([0], (evidence < 0).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(none))
    mask = np.zeros(len(evidence), dtype=bool)
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start > min_len:
            mask[start:end] = True
    return mask


def hysteresis(evidence, dead, confirm):
    """Hold the current team; switch once the other team has `confirm` evidence frames without the
    current team's evidence in between. The switch is dated back to the challenger's first frame.
    No evidence keeps the current team. Dead frames are D and reset; the frames between a reset
    and the next evidence get that evidence's team."""
    labels = np.full(len(evidence), "D", dtype="<U1")
    names = "HA"
    cur, start, challenge_start, challenge_n = None, 0, 0, 0
    for i, e in enumerate(evidence):
        if dead[i]:
            cur, start = None, i + 1
            continue
        if cur is None:
            if e < 0:
                continue
            cur, challenge_n = e, 0
            labels[start:i] = names[e]                 # restart: backfill to the boundary
        elif e >= 0 and e != cur:
            if challenge_n == 0:
                challenge_start = i
            challenge_n += 1
            if challenge_n >= confirm:
                cur, challenge_n = e, 0
                labels[challenge_start:i] = names[e]   # date the turnover back
        elif e == cur:
            challenge_n = 0
        labels[i] = names[cur]
    return labels


def finalize(evidence, dead, frames):
    evidence = np.asarray(evidence)
    dead = np.asarray(dead, dtype=bool)
    if METHOD == "baseline":
        labels = majority_smooth(evidence, SMOOTH_WINDOW)
        labels[dead] = "D"
        return labels
    # ponytail: long gaps -> D is a stopgap until dead_ball.py is ready
    dead = dead | long_gaps(evidence, int(LONG_GAP_DEAD_S * config.FPS))
    return hysteresis(evidence, dead, max(int(SWITCH_CONFIRM_S * config.FPS), 1))

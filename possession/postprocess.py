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
RESTART_FIX_S = 2.0            # source: football reasoning: a restart gives the ball to one team, and its first pass (throw-in, free kick, goal kick) is usually still in flight or just received 2 s later; evidence for the other team that early means the first frame read the wrong team, so the whole stretch since the restart goes to the team that is confirmed
RESTART_CONFIRM_S = 0.6        # source: football reasoning: inside that window it takes a little longer than one pass or duel (under half a second) to overturn the restart team, but less than an open-play turnover because we already know a fresh possession has just started
MIN_SPELL_S = None             # source: football reasoning: off; a run of one team shorter than this between two runs of the other team is a blip (a touch, a block) and merges into them. Real possessions on the practice set are rarely under 1 s (2%), so 1.0 is the largest sensible value; with SWITCH_CONFIRM_S = 1.0 it changes nothing, it only bites with a shorter switch wait
LONG_GAP_DEAD_S = None         # source: football reasoning: off since dead_ball.py detects stoppages directly; a ball the tracker cannot see is still in play and stays with the last team (was 8.0 s as a stopgap; team decision Nov 12). A number of seconds turns it back on


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


def merge_short_spells(labels, min_len):
    """H/A runs shorter than min_len frames with the other team on both sides take that team.
    Runs that touch D (a restart or a stoppage) are kept: those boundaries come from dead ball."""
    labels = labels.copy()
    if min_len <= 1:
        return labels
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    for k in range(1, len(bounds) - 2):
        s, e = bounds[k], bounds[k + 1]
        left, right = labels[bounds[k] - 1], labels[e] if e < len(labels) else "D"
        if e - s < min_len and labels[s] in "HA" and left == right and left in "HA" and left != labels[s]:
            labels[s:e] = left
    return labels


def hysteresis(evidence, dead, confirm, restart_fix=0, restart_confirm=None):
    """Hold the current team; switch once the other team has `confirm` evidence frames without the
    current team's evidence in between. The switch is dated back to the challenger's first frame.
    No evidence keeps the current team. Dead frames are D and reset; the frames between a reset
    and the next evidence get that evidence's team."""
    labels = np.full(len(evidence), "D", dtype="<U1")
    names = "HA"
    if restart_confirm is None:
        restart_confirm = confirm
    cur, start, challenge_start, challenge_n = None, 0, 0, 0
    first = 0                                          # frame where the current spell's team was first set
    restart_open = False                               # still inside the window after a restart
    for i, e in enumerate(evidence):
        if dead[i]:
            cur, start = None, i + 1
            continue
        if cur is None:
            if e < 0:
                continue
            cur, challenge_n, first = e, 0, i
            restart_open = restart_fix > 0
            labels[start:i] = names[e]                 # restart: backfill to the boundary
        elif e >= 0 and e != cur:
            if challenge_n == 0:
                challenge_start = i
            challenge_n += 1
            if restart_open and i - first > restart_fix:
                restart_open = False
            need = restart_confirm if restart_open else confirm
            if challenge_n >= need:
                cur, challenge_n = e, 0
                back = start if restart_open else challenge_start
                labels[back:i] = names[e]              # date the turnover back (to the restart if early)
                restart_open = False
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
    if LONG_GAP_DEAD_S is not None:   # old stopgap from before dead_ball.py: long blind gaps -> D
        dead = dead | long_gaps(evidence, int(LONG_GAP_DEAD_S * config.FPS))
    confirm = max(int(SWITCH_CONFIRM_S * config.FPS), 1)
    restart_fix = int(RESTART_FIX_S * config.FPS) if RESTART_FIX_S else 0
    restart_confirm = max(int(RESTART_CONFIRM_S * config.FPS), 1) if RESTART_CONFIRM_S else confirm
    labels = hysteresis(evidence, dead, confirm, restart_fix, restart_confirm)
    if MIN_SPELL_S:
        labels = merge_short_spells(labels, int(MIN_SPELL_S * config.FPS))
    return labels

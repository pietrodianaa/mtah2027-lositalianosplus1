"""Synthetic sanity check for postprocess.METHOD: baseline vs hysteresis.

No match data. A seeded synthetic truth (possessions and dead-ball stretches with durations from
football reasoning) is scored with scores.postprocess, which adds the frozen synthetic noise.
Run:  python experiments/pietro/synth_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, postprocess, scores  # noqa: E402

SEED = 0                       # source: bookkeeping (any fixed seed)
HALF_S = 45 * 60               # source: football reasoning: one half of play
POSSESSION_MEAN_S = 12.0       # source: football reasoning: a typical possession lasts ~10-15 s
POSSESSION_MIN_S = 1.0         # source: football reasoning: shortest real possession is a touch and a pass
DEAD_SHARE = 0.25              # source: football reasoning: roughly one possession in four ends with the ball out or a foul
DEAD_RANGE_S = (8.0, 40.0)     # source: football reasoning: throw-ins take ~10 s, goal kicks and free kicks up to ~40 s


def synthetic_truth(rng):
    parts, team, total = [], rng.integers(2), 0
    while total < HALF_S * config.FPS:
        n = int(max(rng.exponential(POSSESSION_MEAN_S), POSSESSION_MIN_S) * config.FPS)
        parts.append(np.full(n, "HA"[team]))
        total += n
        if rng.random() < DEAD_SHARE:
            n = int(rng.uniform(*DEAD_RANGE_S) * config.FPS)
            parts.append(np.full(n, "D"))
            total += n
            team = rng.integers(2)
        else:
            team = 1 - team
    return np.concatenate(parts)


if __name__ == "__main__":
    truth = synthetic_truth(np.random.default_rng(SEED))
    for method in ("baseline", "hysteresis"):
        postprocess.METHOD = method
        r = scores.postprocess(postprocess.finalize, truth)
        print("{:<11} spellF1 {:.4f}  macroF1 {:.4f}  pred/true spells {}/{}".format(
            method, r["spellF1"], r["macroF1"], r["n_pred_spells"], r["n_true_spells"]))

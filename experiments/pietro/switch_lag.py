"""Aggregate check on EXTERNAL matches only: how late do turnovers land when live control evidence
goes through live postprocess (dead ball = truth)? Prints medians over all games, no frame tables.

Usage: python experiments/pietro/switch_lag.py [SWITCH_CONFIRM_S ...]
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from possession import config, control, external, features, postprocess  # noqa: E402

MATCH_WINDOW_S = 5.0   # source: football reasoning: a predicted switch more than 5 s from a true turnover is a different event, not a late one


def switches(labels):
    """Frames where the label goes directly H->A or A->H (no D in between)."""
    lab = np.asarray(labels)
    i = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"]) & np.isin(lab[:-1], ["H", "A"]))
    return i + 1


def main(confirms):
    cache = []
    for g in external.games():
        frames, tags, truth = external.load(g)
        ev = control.detect(frames, features.compute(frames), tags)
        cache.append((ev, truth))
    win = int(MATCH_WINDOW_S * config.FPS)
    for c in confirms:
        postprocess.SWITCH_CONFIRM_S = c
        lags, missed, n_true = [], 0, 0
        for ev, truth in cache:
            labels = postprocess.finalize(ev, truth == "D", None)
            t, p = switches(truth), switches(labels)
            n_true += len(t)
            for s in t:
                near = p[np.abs(p - s) <= win]
                if len(near) == 0:
                    missed += 1
                else:
                    lags.append(near[np.argmin(np.abs(near - s))] - s)
        lags = np.asarray(lags) / config.FPS
        print("SWITCH_CONFIRM_S={:.2f}  true turnovers {}  found within {:.0f}s {:.1%}  lag median {:+.2f}s  "
              "p25 {:+.2f}s  p75 {:+.2f}s".format(c, n_true, MATCH_WINDOW_S, 1 - missed / n_true,
                                                  np.median(lags), *np.percentile(lags, [25, 75])))


if __name__ == "__main__":
    main([float(x) for x in sys.argv[1:]] or [postprocess.SWITCH_CONFIRM_S])

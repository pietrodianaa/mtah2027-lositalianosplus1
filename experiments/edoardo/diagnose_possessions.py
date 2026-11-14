"""Scratch: the two possession-level errors of dead ball v2 (main). 20 external games only, never the provided match.

A. swallowed: true possessions (runs of H or A) at least 90% covered by our D. Duration, whether a
   true stoppage sits on both sides, and which rule covers them (rule 1 alone already covers it?).
B. missed same-team stops: true stoppages with the same team before and after that we do not touch
   at all (each one merges two possessions). Duration and where the ball is.
Totals only, no frame indices.
Run from the repo root:  python experiments/edoardo/diagnose_possessions.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features  # noqa: E402

SWALLOWED_SHARE = 0.9                              # source: football reasoning: a possession 90% under D is lost for matching
DURATION_EDGES_S = [0, 1, 2, 5, 10, 20, 40]        # source: football reasoning: quick restart < 5 s, throw-in ~15 s, corner ~34 s
NEAR_LINE_M = 2.0                                  # source: football reasoning: a restart after an out is taken on or next to a line
STILL_DRIFT_M = 1.0                                # source: football reasoning: same "ball still" scale as rule 2 (< 1 m in the stretch)

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bucket(value):
    for low, high in zip(DURATION_EDGES_S[:-1], DURATION_EDGES_S[1:]):
        if low <= value < high:
            return "{:>2}-{:<2} s".format(low, high)
    return ">={} s".format(DURATION_EDGES_S[-1])


labels = ["{:>2}-{:<2} s".format(a, b) for a, b in zip(DURATION_EDGES_S[:-1], DURATION_EDGES_S[1:])] + [">={} s".format(DURATION_EDGES_S[-1])]
sw = {b: {"n": 0, "between_stops": 0, "rule1_alone": 0} for b in labels}
ms = {b: {"n": 0, "past a line": 0, "near a line": 0, "mid-pitch, ball still": 0, "mid-pitch, ball moving": 0, "ball missing": 0} for b in labels}
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    dead_ball.STILL_ENABLED = True
    pred = dead_ball.detect(frames, feats, tags)
    dead_ball.STILL_ENABLED = False
    pred_rule1 = dead_ball.detect(frames, feats, tags)
    dead_ball.STILL_ENABLED = True
    graded = truth != "X"
    x, y = frames["ball"][:, 0], frames["ball"][:, 1]
    length, width = frames["pitch"]
    # A. swallowed possessions
    for team in ("H", "A"):
        for start, end in runs((truth == team) & graded):
            if pred[start:end].mean() < SWALLOWED_SHARE:
                continue
            b = bucket((end - start) / config.FPS)
            sw[b]["n"] += 1
            before = truth[start - 1] if start > 0 else "X"
            after = truth[end] if end < len(truth) else "X"
            sw[b]["between_stops"] += before == "D" and after == "D"
            sw[b]["rule1_alone"] += pred_rule1[start:end].mean() >= SWALLOWED_SHARE
    # B. missed same-team stops
    for start, end in runs((truth == "D") & graded):
        before = truth[start - 1] if start > 0 else "X"
        after = truth[end] if end < len(truth) else "X"
        if not (before in ("H", "A") and before == after) or pred[start:end].any():
            continue
        b = bucket((end - start) / config.FPS)
        ms[b]["n"] += 1
        seen = ~np.isnan(x[start:end])
        if not seen.any():
            ms[b]["ball missing"] += 1
            continue
        sx, sy = x[start:end][seen], y[start:end][seen]
        to_line = np.minimum(np.minimum(sx, length - sx), np.minimum(sy, width - sy))
        if (to_line < 0).any():
            ms[b]["past a line"] += 1
        elif (to_line < NEAR_LINE_M).any():
            ms[b]["near a line"] += 1
        elif np.hypot(sx.max() - sx.min(), sy.max() - sy.min()) < STILL_DRIFT_M:
            ms[b]["mid-pitch, ball still"] += 1
        else:
            ms[b]["mid-pitch, ball moving"] += 1

print("labels v{}, dead ball v2 as on main\n".format(external.LABELS_VERSION))
print("A. swallowed possessions: {}".format(sum(v["n"] for v in sw.values())))
print("duration    count  between two true stops  rule 1 alone covers it")
for b in labels:
    if sw[b]["n"]:
        print("{:>9}   {:5d}  {:22d}  {:22d}".format(b, sw[b]["n"], sw[b]["between_stops"], sw[b]["rule1_alone"]))
print("\nB. missed same-team stops: {}".format(sum(v["n"] for v in ms.values())))
print("duration    count  past line  near line  mid still  mid moving  ball missing")
for b in labels:
    m = ms[b]
    if m["n"]:
        print("{:>9}   {:5d}  {:9d}  {:9d}  {:9d}  {:10d}  {:12d}".format(
            b, m["n"], m["past a line"], m["near a line"], m["mid-pitch, ball still"], m["mid-pitch, ball moving"], m["ball missing"]))

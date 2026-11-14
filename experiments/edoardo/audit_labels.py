"""Scratch: what are the true D stretches of the external labels? 20 external games only, never the provided match.

External truth = every gap between two SkillCorner phases of play in the same half is D (external.py).
A real match has ~100 stoppages; labels v1 count ~145 per game. Are the extra ones tiny gaps
between two phases in open play? Totals and shares only, no frame indices.
Run from the repo root:  python experiments/edoardo/audit_labels.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, external  # noqa: E402

DURATION_EDGES_S = [0, 1, 2, 5, 10, 20, 40, 90]   # source: football reasoning: quick restart < 5 s, throw-in ~15 s, corner ~34 s, goal ~70 s
NEAR_LINE_M = 2.0                                  # source: football reasoning: a restart after an out is taken on or next to a line

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
count = {b: 0 for b in labels}
frames_in = {b: 0 for b in labels}
same_team = {b: 0 for b in labels}
where = {b: {"past a line": 0, "near a line": 0, "mid-pitch": 0, "ball missing": 0} for b in labels}
n_games, n_possessions, graded_frames = 0, 0, 0
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    n_games += 1
    graded = truth != "X"
    graded_frames += graded.sum()
    for team in ("H", "A"):
        n_possessions += len(runs((truth == team) & graded))
    x, y = frames["ball"][:, 0], frames["ball"][:, 1]
    length, width = frames["pitch"]
    for start, end in runs((truth == "D") & graded):
        b = bucket((end - start) / config.FPS)
        count[b] += 1
        frames_in[b] += end - start
        before = truth[start - 1] if start > 0 else "X"
        after = truth[end] if end < len(truth) else "X"
        same_team[b] += before in ("H", "A") and before == after
        seg_x, seg_y = x[start:end], y[start:end]
        seen = ~np.isnan(seg_x)
        if not seen.any():
            where[b]["ball missing"] += 1
            continue
        sx, sy = seg_x[seen], seg_y[seen]
        to_line = np.minimum(np.minimum(sx, length - sx), np.minimum(sy, width - sy))
        if (to_line < 0).any():
            where[b]["past a line"] += 1
        elif (to_line < NEAR_LINE_M).any():
            where[b]["near a line"] += 1
        else:
            where[b]["mid-pitch"] += 1

total = sum(count.values())
total_frames = sum(frames_in.values())
print("labels v{}  games {}  true D stretches {} ({:.1f} per game)  true possessions {:.1f} per game".format(
    external.LABELS_VERSION, n_games, total, total / n_games, n_possessions / n_games))
print("D frames: {:.1%} of graded frames\n".format(total_frames / graded_frames))
print("duration    stretches  per game  share of D frames  same team before/after   past line  near line  mid-pitch  missing")
for b in labels:
    c = count[b]
    if not c:
        continue
    w = where[b]
    print("{:>9}   {:9d}  {:8.1f}  {:17.1%}  {:21.0%}   {:9d}  {:9d}  {:9d}  {:7d}".format(
        b, c, c / n_games, frames_in[b] / total_frames, same_team[b] / c,
        w["past a line"], w["near a line"], w["mid-pitch"], w["ball missing"]))

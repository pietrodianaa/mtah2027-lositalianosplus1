"""Scratch: the true stops we still never touch (dead ball as on this branch). 20 external games only,
never the provided match. Before adding a rule (wall at free kicks, ...): where is the ball, do players
walk, how long are they, does the same team keep the ball. Totals and medians only, no frame indices.
Run from the repo root:  python experiments/edoardo/diagnose_missed.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features  # noqa: E402

NEAR_LINE_M = 6.0      # source: football reasoning: same as RESTART_ZONE_M, a restart after an out is within ~6 m of a line
BOX_DEPTH_M = 30.0     # source: football reasoning: free kicks with a wall are taken within ~30 m of goal
STILL_M = 1.0          # source: football reasoning: a placed ball moves < 1 m (same scale as rule 2)
WALK_MS = 2.0          # source: football reasoning: below jogging pace (same as UNSEEN_WALK_MS)
DUR_EDGES_S = [0, 2, 5, 10, 20, 40]   # source: football reasoning: phase gaps < 2 s, quick restarts < 5 s, throw-in ~15 s, corner ~34 s

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bucket(v):
    for lo, hi in zip(DUR_EDGES_S[:-1], DUR_EDGES_S[1:]):
        if lo <= v < hi:
            return "{:>2}-{:<2} s".format(lo, hi)
    return ">={} s".format(DUR_EDGES_S[-1])


places = ["ball missing", "near a line", "still, <30 m from goal", "still, elsewhere", "moving, <30 m from goal", "moving, elsewhere"]
buckets = [bucket(v) for v in DUR_EDGES_S[:-1]] + [bucket(DUR_EDGES_S[-1])]
table = {p: {b: 0 for b in buckets} for p in places}
frames_by_place = {p: 0 for p in places}
walk = {p: [] for p in places}
same = {p: 0 for p in places}
dead_ball.KICKOFF_ENABLED = False
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    pred = dead_ball.detect(frames, feats, tags)
    length, width = frames.get("pitch", config.PITCH)
    x, y = frames["ball"][:, 0], frames["ball"][:, 1]
    ps = frames["player_speed"]
    n_ps = (~np.isnan(ps)).sum(axis=1)
    mean_ps = np.where(np.isnan(ps), 0.0, ps).sum(axis=1) / np.maximum(n_ps, 1)
    for start, end in runs((truth == "D") & (truth != "X")):
        if pred[start:end].any():
            continue
        seen = ~np.isnan(x[start:end])
        if not seen.any():
            place = "ball missing"
        else:
            sx, sy = x[start:end][seen], y[start:end][seen]
            to_line = np.minimum(np.minimum(sx, length - sx), np.minimum(sy, width - sy)).min()
            to_goal = np.minimum(sx, length - sx).min()
            still = np.hypot(sx.max() - sx.min(), sy.max() - sy.min()) < STILL_M
            if to_line < NEAR_LINE_M and to_goal >= NEAR_LINE_M or to_line < 1.0:
                place = "near a line"
            else:
                place = ("still, " if still else "moving, ") + ("<30 m from goal" if to_goal < BOX_DEPTH_M else "elsewhere")
        table[place][bucket((end - start) / config.FPS)] += 1
        frames_by_place[place] += end - start
        sp = mean_ps[start:end][n_ps[start:end] > 0]
        if sp.size:
            walk[place].append((sp < WALK_MS).mean())
        before = truth[start - 1] if start > 0 else "X"
        after = truth[end] if end < len(truth) else "X"
        same[place] += before in ("H", "A") and before == after

tot = sum(frames_by_place.values())
print("labels v{}, untouched true stops, dead ball as on this branch (rule 4 off)\n".format(external.LABELS_VERSION))
print("{:<26} {:>6} {:>8} {:>10} {:>14}   {}".format("ball during the stop", "stops", "frames%", "same team", "walking (med)", "  ".join(buckets)))
for p in places:
    n = sum(table[p].values())
    if not n:
        continue
    print("{:<26} {:6d} {:7.1f}% {:10d} {:14.2f}   {}".format(
        p, n, 100 * frames_by_place[p] / max(tot, 1), same[p], np.median(walk[p]) if walk[p] else np.nan,
        "  ".join("{:>6d}".format(table[p][b]) for b in buckets)))

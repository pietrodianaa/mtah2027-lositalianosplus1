"""Scratch: where the missed dead frames are, for dead ball as on main. 20 external games only, never the provided match.

Splits every true dead frame (D in the external labels) into:
  covered       our D is on it
  untouched     inside a true stop we never mark at all
  late start    inside a touched stop, before our first D frame
  early end     inside a touched stop, after our last D frame
  hole          inside a touched stop, between two of our D frames
and our D outside true D into: right before a stop / right after a stop / elsewhere.
For touched stops it also gives the start lag and end gap in seconds, by the rule that fires first
and by where the ball is when the stop starts. Totals only, no frame indices.
Run from the repo root:  python experiments/edoardo/diagnose_coverage.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features  # noqa: E402

NEAR_LINE_M = 2.0      # source: football reasoning: a ball within 2 m of a line when play stops is an out or a restart on the line
LAG_EDGES_S = [0, 1, 2, 4, 8, 16]   # source: football reasoning: 1-2 s is the rule windows, 4-8 s is a throw-in set-up, 16 s a corner

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bucket(seconds):
    for low, high in zip(LAG_EDGES_S[:-1], LAG_EDGES_S[1:]):
        if low <= seconds < high:
            return "{:>2}-{:<2} s".format(low, high)
    return ">={} s".format(LAG_EDGES_S[-1])


def detect_with(still, walk, frames, feats, tags):
    dead_ball.STILL_ENABLED, dead_ball.WALK_ENABLED = still, walk
    out = dead_ball.detect(frames, feats, tags)
    dead_ball.STILL_ENABLED, dead_ball.WALK_ENABLED = True, True
    return out


fr = {"covered": 0, "untouched": 0, "late start": 0, "early end": 0, "hole": 0}
extra = {"right before a stop": 0, "right after a stop": 0, "elsewhere": 0}
groups = {}            # (group name) -> {"n", "start": [lags], "end": [gaps]}
n_stops = n_touched = 0
lag_buckets = [bucket(e) for e in LAG_EDGES_S[:-1]] + [bucket(LAG_EDGES_S[-1])]

for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    pred = dead_ball.detect(frames, feats, tags)
    r1 = detect_with(False, False, frames, feats, tags)
    r12 = detect_with(True, False, frames, feats, tags)
    graded = truth != "X"
    true_d = (truth == "D") & graded
    x, y = frames["ball"][:, 0], frames["ball"][:, 1]
    length, width = frames.get("pitch", config.PITCH)

    for start, end in runs(true_d):
        n_stops += 1
        inside = pred[start:end]
        if not inside.any():
            fr["untouched"] += end - start
            continue
        n_touched += 1
        hit = np.flatnonzero(inside)
        first, last = hit[0], hit[-1]
        fr["covered"] += int(inside.sum())
        fr["late start"] += int(first)
        fr["early end"] += int(end - start - 1 - last)
        fr["hole"] += int((~inside[first:last + 1]).sum())
        # which rule fires first, and where the ball is when the stop starts
        k = start + first
        rule = "rule 1 (out)" if r1[k] else "rule 2 (still)" if r12[k] else "rule 2b (walk)"
        seen = ~np.isnan(x[start:start + int(config.FPS)])
        if not seen.any():
            where = "ball missing at start"
        else:
            sx, sy = x[start:start + int(config.FPS)][seen], y[start:start + int(config.FPS)][seen]
            to_line = np.minimum(np.minimum(sx, length - sx), np.minimum(sy, width - sy)).min()
            where = "ball past a line" if to_line < 0 else "ball near a line" if to_line < NEAR_LINE_M else "ball mid-pitch"
        before = truth[start - 1] if start > 0 else "X"
        after = truth[end] if end < len(truth) else "X"
        same = "same team before/after" if before in ("H", "A") and before == after else "team changes"
        for name in ("all touched", rule, where, same):
            g = groups.setdefault(name, {"n": 0, "start": [], "end": []})
            g["n"] += 1
            g["start"].append(first / config.FPS)
            g["end"].append((end - start - 1 - last) / config.FPS)

    # our D outside true D
    for start, end in runs(pred & graded & ~true_d):
        if end < len(truth) and truth[end] == "D":
            extra["right before a stop"] += end - start
        elif start > 0 and truth[start - 1] == "D":
            extra["right after a stop"] += end - start
        else:
            extra["elsewhere"] += end - start

total = sum(fr.values())
print("labels v{}, dead ball as on main, {} games\n".format(external.LABELS_VERSION, len(external.games())))
print("True dead frames: {:.0f} min. Stops: {}, touched by our D: {}".format(total / config.FPS / 60, n_stops, n_touched))
for k, v in fr.items():
    print("  {:<11} {:5.1f}%  ({:.0f} min)".format(k, 100 * v / total, v / config.FPS / 60))
tot_extra = sum(extra.values())
print("\nOur D on non-dead frames: {:.1f} min".format(tot_extra / config.FPS / 60))
for k, v in extra.items():
    print("  {:<20} {:5.1f}%".format(k, 100 * v / max(tot_extra, 1)))

print("\nTouched stops: start lag and end gap (seconds), median / 75th pct, and start-lag histogram")
print("{:<24} {:>5}  {:>13}  {:>13}   {}".format("group", "n", "lag med/p75", "gap med/p75", "  ".join(lag_buckets)))
for name in ["all touched", "rule 1 (out)", "rule 2 (still)", "rule 2b (walk)",
             "ball past a line", "ball near a line", "ball mid-pitch", "ball missing at start",
             "same team before/after", "team changes"]:
    g = groups.get(name)
    if not g:
        continue
    s, e = np.array(g["start"]), np.array(g["end"])
    hist = [sum(bucket(v) == b for v in s) for b in lag_buckets]
    print("{:<24} {:5d}  {:5.1f} / {:5.1f}  {:5.1f} / {:5.1f}   {}".format(
        name, g["n"], np.median(s), np.percentile(s, 75), np.median(e), np.percentile(e, 75),
        "  ".join("{:>6d}".format(h) for h in hist)))

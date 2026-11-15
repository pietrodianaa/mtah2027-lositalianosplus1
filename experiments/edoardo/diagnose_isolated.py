"""Scratch: describe the isolated D runs (touch no true dead stretch), the ones that split possessions.

20 external games only, never the provided match. Prints only counts per bucket over all games.
Run from the repo root:  python experiments/edoardo/diagnose_isolated.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features  # noqa: E402

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bucket(value, edges):
    """Label of the bucket that holds value, e.g. '2-5'."""
    for low, high in zip(edges[:-1], edges[1:]):
        if low <= value < high:
            return "{}-{}".format(low, high)
    return ">={}".format(edges[-1])


iso = []      # one dict per isolated run
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    graded = truth != "X"
    true_dead = (truth == "D") & graded
    pred = dead_ball.detect(frames, feats, tags) & graded
    x, y = frames["ball"][:, 0], frames["ball"][:, 1]
    length, width = frames["pitch"]
    missing = np.isnan(x)
    true_idx = np.flatnonzero(true_dead)
    for start, end in runs(pred):
        if true_dead[start:end].any():
            continue
        seg_x, seg_y = x[start:end], y[start:end]
        with np.errstate(invalid="ignore"):
            past_side = np.nanmax(np.concatenate(([0.0], -seg_y, seg_y - width)))      # beyond a touchline
            past_end = np.nanmax(np.concatenate(([0.0], -seg_x, seg_x - length)))      # beyond a byline
        if true_idx.size:
            gap = min(abs(true_idx - start).min(), abs(true_idx - (end - 1)).min()) / config.FPS
        else:
            gap = np.inf
        with np.errstate(invalid="ignore"):
            out = ((seg_x < 0) | (seg_x > length) | (seg_y < 0) | (seg_y > width)).any()
        iso.append({"rule": "rule 1 (ball out)" if out else "rule 2 (ball still)",
                    "dur": (end - start) / config.FPS,
                    "past": max(past_side, past_end),
                    "line": "touchline" if past_side >= past_end else "byline",
                    "missing": missing[start:end].mean(),
                    "gap": gap})

print("isolated D runs:", len(iso), "in", len(external.games()), "games, labels v{}".format(external.LABELS_VERSION))
if not iso:
    sys.exit()
reports = [
    ("duration (s)", "dur", [2, 5, 10, 20, 40]),
    ("max distance past the line (m)", "past", [0, 0.5, 1, 2, 5]),
    ("share of the run with the ball missing", "missing", [0, 0.25, 0.5, 0.75, 1.0]),
    ("distance to the nearest true dead stretch (s)", "gap", [0, 2, 5, 10, 30, 60]),
]
for title, key, edges in reports:
    counts = {}
    for run in iso:
        label = bucket(run[key], edges)
        counts[label] = counts.get(label, 0) + 1
    order = ["{}-{}".format(a, b) for a, b in zip(edges[:-1], edges[1:])] + [">={}".format(edges[-1])]
    print("\n" + title + ":")
    for label in order:
        if label in counts:
            print("  {:>10}  {}".format(label, counts[label]))
lines = {}
for run in iso:
    lines[run["line"]] = lines.get(run["line"], 0) + 1
print("\nline:", ", ".join("{} {}".format(k, v) for k, v in sorted(lines.items())))
rules = {}
for run in iso:
    rules[run["rule"]] = rules.get(run["rule"], 0) + 1
print("rule:", ", ".join("{} {}".format(k, v) for k, v in sorted(rules.items())))
print("total frames in isolated runs:", int(sum(r["dur"] for r in iso) * config.FPS),
      "  mean duration {:.1f} s".format(np.mean([r["dur"] for r in iso])))

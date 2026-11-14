"""Scratch: are the D runs added by rule 1c that touch no true stop really open play? 20 external games
only, never the provided match. Compares the runs rule 1c adds (main -> main + rule 1c) that touch a
true stop with the ones that touch none (isolated), and both with true stops and open play:
duration, players' mean speed, share of the run with everyone walking, team on each side.
External labels: SkillCorner defines phases of play only while the ball is in play, so an isolated
run sits inside a phase. If its players move like in a true stop, the labels may be missing stops.
Totals and medians only, no frame indices.
Run from the repo root:  python experiments/edoardo/diagnose_unseen.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features  # noqa: E402

WALK_MS = 1.5          # source: football reasoning: everyone at walking pace or slower (rule 2b uses 1.0 smoothed)

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


groups = {k: {"n": 0, "dur": [], "speed": [], "walk": [], "same": 0}
          for k in ("1c run on a true stop", "1c run isolated", "true stop (all)", "open play (true H/A run)")}
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    dead_ball.UNSEEN_ENABLED = False
    base = dead_ball.detect(frames, feats, tags)
    dead_ball.UNSEEN_ENABLED = True
    full = dead_ball.detect(frames, feats, tags)
    graded = truth != "X"
    true_dead = (truth == "D") & graded
    ps = frames["player_speed"]
    tracked = ~np.isnan(ps)
    mean_speed = np.where(tracked, ps, 0.0).sum(axis=1) / np.maximum(tracked.sum(axis=1), 1)
    mean_speed[tracked.sum(axis=1) == 0] = np.nan

    def add(name, start, end):
        g = groups[name]
        g["n"] += 1
        g["dur"].append((end - start) / config.FPS)
        sp = mean_speed[start:end]
        sp = sp[~np.isnan(sp)]
        if sp.size:
            g["speed"].append(np.median(sp))
            g["walk"].append((sp < WALK_MS).mean())
        before = truth[start - 1] if start > 0 else "X"
        after = truth[end] if end < len(truth) else "X"
        g["same"] += before in ("H", "A") and before == after

    added = full & ~base & graded
    for start, end in runs(full & graded):
        if not added[start:end].any():
            continue
        name = "1c run on a true stop" if true_dead[start:end].any() else "1c run isolated"
        add(name, start, end)
    for start, end in runs(true_dead):
        add("true stop (all)", start, end)
    for team in ("H", "A"):
        for start, end in runs((truth == team) & graded):
            add("open play (true H/A run)", start, end)

print("labels v{}, rule 1c as on this branch, {} games\n".format(external.LABELS_VERSION, len(external.games())))
print("{:<26} {:>6}  {:>13}  {:>17}  {:>20}  {}".format(
    "group", "n", "dur med/p75 s", "players m/s (med)", "share walking (med)", "same team both sides"))
for name, g in groups.items():
    if not g["n"]:
        continue
    print("{:<26} {:6d}  {:5.1f} / {:5.1f}  {:17.2f}  {:20.2f}  {:.0%}".format(
        name, g["n"], np.median(g["dur"]), np.percentile(g["dur"], 75),
        np.median(g["speed"]) if g["speed"] else np.nan, np.median(g["walk"]) if g["walk"] else np.nan,
        g["same"] / g["n"]))

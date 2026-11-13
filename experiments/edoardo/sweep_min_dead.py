"""Scratch: longer minimum dead run vs isolated D runs. 20 external games only, never the provided match.

For each MIN_DEAD_S: mean precision / recall / dead_F1 (as `--tune`), isolated D runs (touch no
true dead stretch: they split a possession), how many of them are far (>= 10 s) from any true
dead stretch, and true dead stretches touched at least in part (each one we stop touching merges
two true possessions). Totals only.
Run from the repo root:  python experiments/edoardo/sweep_min_dead.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, scores  # noqa: E402

MIN_DEADS_S = [2.0, 3.0, 4.0, 5.0]   # source: football reasoning: throw-in ~15 s on average (Opta), quick ones still several seconds
FAR_S = 10.0                                   # source: football reasoning: 10 s from any stoppage = clearly open play

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")

games = []
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    tracked = ~np.isnan(frames["player_speed"])
    mean_speed = np.where(tracked, frames["player_speed"], 0.0).sum(axis=1) / np.maximum(tracked.sum(axis=1), 1)
    slim = {"ball": frames["ball"], "half": frames["half"], "pitch": frames["pitch"],
            "player_speed": mean_speed[:, None]}   # per-frame mean speed: same average for rule 2, far less memory
    games.append((slim, {"nearest_dist": feats["nearest_dist"]}, tags, truth))
print("loaded", len(games), "games, labels v{}".format(external.LABELS_VERSION))


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


print("min_dead_s   precision  recall  dead_F1   isolated  isolated_far  true_stops_touched")
for min_dead in MIN_DEADS_S:
    dead_ball.MIN_DEAD_S = min_dead
    rows, isolated, far, touched = [], 0, 0, 0
    for frames, feats, tags, truth in games:
        pred = dead_ball.detect(frames, feats, tags)
        rows.append(scores.dead_ball(pred, truth))
        graded = truth != "X"
        true_dead = (truth == "D") & graded
        true_idx = np.flatnonzero(true_dead)
        for start, end in runs(true_dead):
            touched += bool(pred[start:end].any())
        for start, end in runs(pred & graded):
            if true_dead[start:end].any():
                continue
            isolated += 1
            gap = min(abs(true_idx - start).min(), abs(true_idx - (end - 1)).min()) if true_idx.size else np.inf
            far += gap >= FAR_S * config.FPS
    p, r, f = [np.mean([row[k] for row in rows]) for k in ("dead_precision", "dead_recall", "dead_F1")]
    mark = "  <- main (v2)" if min_dead == 2.0 else ""
    print("{:10.0f}   {:9.4f}  {:6.4f}  {:7.4f}   {:8d}  {:12d}  {:18d}{}".format(min_dead, p, r, f, isolated, far, touched, mark), flush=True)

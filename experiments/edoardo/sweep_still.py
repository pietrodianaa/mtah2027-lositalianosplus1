"""Scratch: rule 2 (ball still) parameters on the 20 external games, never the provided match.

Per setting: mean precision / recall / dead_F1 (as `--tune`), isolated D runs (touch no true dead
stretch: they split a possession) and how many are far (>= 10 s) from any stoppage. Totals only.
First row = rule 2 off (v1.2 on main).
Run from the repo root:  python experiments/edoardo/sweep_still.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, scores  # noqa: E402

STILL_SPEEDS_MS = [0.5, 1.0]               # source: football reasoning: a placed ball barely moves
STILL_CLEARS_M = [1.0, 1.5, 3.0]           # source: football reasoning: keeper / player on the ball within ~1 m, taker steps a few metres away
STILL_PLAYERS_MS = [2.0, 2.5, 3.5, 99.0]   # source: football reasoning: players walk during set-up; 99 = condition off
STILL_MINS_S = [3.0, 5.0]                  # source: football reasoning: setting up a free kick takes seconds
FAR_S = 10.0                               # source: football reasoning: 10 s from any stoppage = clearly open play

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")

# load once, keep only what the rules use; player speeds reduced to their per-frame mean
# (one column), which gives detect() the same average (players would take GBs of memory)
games = []
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    tracked = ~np.isnan(frames["player_speed"])
    mean_speed = np.where(tracked, frames["player_speed"], 0.0).sum(axis=1) / np.maximum(tracked.sum(axis=1), 1)
    slim = {"ball": frames["ball"], "half": frames["half"], "pitch": frames["pitch"],
            "player_speed": mean_speed[:, None]}
    games.append((slim, {"nearest_dist": feats["nearest_dist"]}, tags, truth))
print("loaded", len(games), "games, labels v{}".format(external.LABELS_VERSION))


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def evaluate():
    rows, isolated, far = [], 0, 0
    for frames, feats, tags, truth in games:
        pred = dead_ball.detect(frames, feats, tags)
        rows.append(scores.dead_ball(pred, truth))
        graded = truth != "X"
        true_dead = (truth == "D") & graded
        true_idx = np.flatnonzero(true_dead)
        for start, end in runs(pred & graded):
            if true_dead[start:end].any():
                continue
            isolated += 1
            gap = min(abs(true_idx - start).min(), abs(true_idx - (end - 1)).min()) if true_idx.size else np.inf
            far += gap >= FAR_S * config.FPS
    p, r, f = [np.mean([row[k] for row in rows]) for k in ("dead_precision", "dead_recall", "dead_F1")]
    return p, r, f, isolated, far


print("speed  clear  players  min_s   precision  recall  dead_F1   isolated  far")
dead_ball.STILL_ENABLED = False
print("  off      -        -      -   {:9.4f}  {:6.4f}  {:7.4f}   {:8d}  {:3d}  <- v1.2".format(*evaluate()), flush=True)
dead_ball.STILL_ENABLED = True
for speed in STILL_SPEEDS_MS:
    for clear in STILL_CLEARS_M:
        for players in STILL_PLAYERS_MS:
            for min_s in STILL_MINS_S:
                dead_ball.STILL_SPEED_MS = speed
                dead_ball.STILL_CLEAR_M = clear
                dead_ball.STILL_PLAYERS_MS = players
                dead_ball.STILL_MIN_S = min_s
                print("{:5.1f}  {:5.1f}  {:7.1f}  {:5.0f}   {:9.4f}  {:6.4f}  {:7.4f}   {:8d}  {:3d}".format(
                    speed, clear, players, min_s, *evaluate()), flush=True)

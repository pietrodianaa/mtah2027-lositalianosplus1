"""Scratch: v2.2 (a duel at the ball before everyone walks = a foul), at possession level, v2.1 otherwise as on main.
20 external games only, never the provided match. First row = duel off (v2.1 on main).
Columns (totals over all games):
  F1 even / odd = dead_F1 on even / odd games separately (a real gain shows in both halves)
  isolated / far = our D runs touching no true stoppage (each splits a possession) / >= 10 s from any
  same_touched   = true stoppages with the SAME team before and after that we touch (only those
                   merge two possessions when missed; ~15% of all stoppages)
  swallowed      = true possessions at least 90% covered by our D (each one is lost)
Run from the repo root:  python experiments/edoardo/sweep_duel.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, scores  # noqa: E402

DUELS_M = [1.0, 1.5, 2.0]                # source: football reasoning: both players within ~1-2 m of the ball (tracking error ~0.5 m)
LOOKBACKS_S = [2.0, 4.0]                 # source: football reasoning: whistle ~1 s after the foul, players slow 1-2 s later
DUEL_WALKS_S = [4.0, 6.0]                # source: external v1: 8 s without a duel (v2.1); shorter after a duel
BACKDATES = [False, True]                # source: football reasoning: dead from the foul, not from the slowdown
FAR_S = 10.0                    # source: football reasoning: 10 s from any stoppage = clearly open play
SWALLOWED_SHARE = 0.9           # source: football reasoning: a possession 90% under D is lost for matching

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")

games = []
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    tracked = ~np.isnan(frames["player_speed"])
    mean_speed = np.where(tracked, frames["player_speed"], 0.0).sum(axis=1) / np.maximum(tracked.sum(axis=1), 1)
    mean_speed[tracked.sum(axis=1) == 0] = np.nan                 # no player tracked: unknown, as in detect()
    slim = {"ball": frames["ball"], "half": frames["half"], "pitch": frames["pitch"],
            "player_speed": mean_speed[:, None]}   # per-frame mean speed: same average for rule 2, far less memory
    games.append((slim, {k: feats[k] for k in ("nearest_dist", "nearest_dist_H", "nearest_dist_A")}, tags, truth))
print("loaded", len(games), "games, labels v{}".format(external.LABELS_VERSION))


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


same_total = 0
for frames, feats, tags, truth in games:
    for start, end in runs((truth == "D") & (truth != "X")):
        before = truth[start - 1] if start > 0 else "X"
        after = truth[end] if end < len(truth) else "X"
        same_total += before in ("H", "A") and before == after
print("true stoppages with the same team before and after:", same_total)

print("\nduel_m  look  walk  back  precision  recall  dead_F1  F1_even  F1_odd   isolated  far  same_touched  swallowed")
settings = [None] + [(m, l, w, b) for m in DUELS_M for l in LOOKBACKS_S for w in DUEL_WALKS_S for b in BACKDATES]
for setting in settings:
    dead_ball.DUEL_ENABLED = setting is not None
    if setting is not None:
        dead_ball.DUEL_M, dead_ball.DUEL_LOOKBACK_S, dead_ball.DUEL_WALK_MIN_S, dead_ball.DUEL_BACKDATE = setting
        name = "{:5.1f}  {:4.0f}  {:4.0f}  {:>4}".format(setting[0], setting[1], setting[2], "yes" if setting[3] else "no")
    else:
        name = "duel off (main)"
    rows, isolated, far, same_touched, swallowed = [], 0, 0, 0, 0
    for frames, feats, tags, truth in games:
        pred = dead_ball.detect(frames, feats, tags)
        rows.append(scores.dead_ball(pred, truth))
        graded = truth != "X"
        true_dead = (truth == "D") & graded
        true_idx = np.flatnonzero(true_dead)
        for start, end in runs(true_dead):
            before = truth[start - 1] if start > 0 else "X"
            after = truth[end] if end < len(truth) else "X"
            if before in ("H", "A") and before == after:
                same_touched += bool(pred[start:end].any())
        for team in ("H", "A"):
            for start, end in runs((truth == team) & graded):
                swallowed += pred[start:end].mean() >= SWALLOWED_SHARE
        for start, end in runs(pred & graded):
            if true_dead[start:end].any():
                continue
            isolated += 1
            gap = min(abs(true_idx - start).min(), abs(true_idx - (end - 1)).min()) if true_idx.size else np.inf
            far += gap >= FAR_S * config.FPS
    p, r, f = [np.mean([row[k] for row in rows]) for k in ("dead_precision", "dead_recall", "dead_F1")]
    f_even = np.mean([row["dead_F1"] for row in rows[0::2]])
    f_odd = np.mean([row["dead_F1"] for row in rows[1::2]])
    print("{:18} {:9.4f}  {:6.4f}  {:7.4f}  {:7.4f}  {:6.4f}   {:8d}  {:3d}  {:12d}  {:9d}".format(
        name, p, r, f, f_even, f_odd, isolated, far, same_touched, swallowed), flush=True)

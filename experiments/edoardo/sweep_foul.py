"""Scratch: v2.2 (Edoardo's foul: duel at the ball, one player stops, everyone walks, restart from the foul
spot), at possession level, v2.1 otherwise as on main. 20 external games only, never the provided match.

Needs every player's position, so it loads ONE game at a time and tries every setting on it before
moving on (memory stays at one game). First row = foul rule off (v2.1 on main).
Columns (totals over all games):
  F1 even / odd = dead_F1 on even / odd games separately (a real gain shows in both halves)
  isolated / far = our D runs touching no true stoppage (each splits a possession) / >= 10 s from any
  same_touched   = true stoppages with the same team before and after that we touch
  swallowed      = true possessions at least 90% covered by our D (each one is lost)
Run from the repo root:  python experiments/edoardo/sweep_foul.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, scores  # noqa: E402

SPOTS_M = [None, 2.0, 5.0, 10.0, 20.0]   # source: football reasoning (Edoardo): free kick from the foul spot, 1-2 m in practice; wider for the ball drifting before the whistle; None = end when players speed up
DUELS_M = [1.5, 2.5]                     # source: football reasoning: both players within ~1.5-2.5 m of the ball (tracking error ~0.5 m)
STOPS_MS = [0.5, 1.0]                    # source: football reasoning: a fouled player stands still or walks
WALKS_S = [4.0, 6.0]                     # source: football reasoning: after a confirmed foul a shorter walk than rule 2b (8 s) is enough
FAR_S = 10.0                             # source: football reasoning: 10 s from any stoppage = clearly open play
SWALLOWED_SHARE = 0.9                    # source: football reasoning: a possession 90% under D is lost for matching

games = external.games()
if not games:
    sys.exit("No external games: run tools/get_external_data.sh first.")

settings = [None] + [(sp, du, st, wk) for sp in SPOTS_M for du in DUELS_M for st in STOPS_MS for wk in WALKS_S]
acc = {k: {"rows": [], "isolated": 0, "far": 0, "same_touched": 0, "swallowed": 0} for k in range(len(settings))}


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


same_total = 0
for g, game_id in enumerate(games):
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    graded = truth != "X"
    true_dead = (truth == "D") & graded
    true_idx = np.flatnonzero(true_dead)
    same_stops = []
    for start, end in runs(true_dead):
        before = truth[start - 1] if start > 0 else "X"
        after = truth[end] if end < len(truth) else "X"
        if before in ("H", "A") and before == after:
            same_stops.append((start, end))
    same_total += len(same_stops)
    possessions = [r for team in ("H", "A") for r in runs((truth == team) & graded)]
    for k, setting in enumerate(settings):
        dead_ball.FOUL_ENABLED = setting is not None
        if setting is not None:
            dead_ball.FOUL_SPOT_M, dead_ball.FOUL_DUEL_M, dead_ball.FOUL_STOP_MS, dead_ball.FOUL_WALK_MIN_S = setting
        pred = dead_ball.detect(frames, feats, tags)
        a = acc[k]
        a["rows"].append(scores.dead_ball(pred, truth))
        a["same_touched"] += sum(bool(pred[s:e].any()) for s, e in same_stops)
        a["swallowed"] += sum(pred[s:e].mean() >= SWALLOWED_SHARE for s, e in possessions)
        for start, end in runs(pred & graded):
            if true_dead[start:end].any():
                continue
            a["isolated"] += 1
            gap = min(abs(true_idx - start).min(), abs(true_idx - (end - 1)).min()) if true_idx.size else np.inf
            a["far"] += gap >= FAR_S * config.FPS
    print("game {}/{} done".format(g + 1, len(games)), flush=True)
    del frames, feats

print("\nlabels v{}, true stoppages with the same team before and after: {}".format(external.LABELS_VERSION, same_total))
print("spot_m  duel_m  stop  walk   precision  recall  dead_F1  F1_even  F1_odd   isolated  far  same_touched  swallowed")
for k, setting in enumerate(settings):
    a = acc[k]
    rows = a["rows"]
    p, r, f = [np.mean([row[key] for row in rows]) for key in ("dead_precision", "dead_recall", "dead_F1")]
    f_even = np.mean([row["dead_F1"] for row in rows[0::2]])
    f_odd = np.mean([row["dead_F1"] for row in rows[1::2]])
    if setting is None:
        name = "foul off (main)          "
    else:
        sp, du, st, wk = setting
        name = "{:>6}  {:6.1f}  {:4.1f}  {:4.0f}".format("end" if sp is None else "{:.0f}".format(sp), du, st, wk)
    print("{}   {:9.4f}  {:6.4f}  {:7.4f}  {:7.4f}  {:6.4f}   {:8d}  {:3d}  {:12d}  {:9d}".format(
        name, p, r, f, f_even, f_odd, a["isolated"], a["far"], a["same_touched"], a["swallowed"]))

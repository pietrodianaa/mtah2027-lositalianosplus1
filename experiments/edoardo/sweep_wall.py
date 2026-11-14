"""Scratch: rule 5, wall at a free kick (Law 13), at possession level, the rest as on branch 10-walk-zone.
20 external games only, never the provided match. Loads one game at a time (the wall needs every
player's position and speed). First row = rule 5 off. Columns as in sweep_unseen.py:
  F1 even / odd = dead_F1 on even / odd games; isolated / far = our D runs touching no true stop / >= 10 s
  from any; same_touched = same-team true stops we touch; swallowed = true possessions >= 90% under our D;
  spell_proxy = official spellF1 with teams from the truth and D from us (others perfect)
Run from the repo root:  python experiments/edoardo/sweep_wall.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, scores  # noqa: E402

WALL_NS = [3, 4]                          # source: football reasoning: a wall has 3 or more players
WALL_RANGES_M = [(7.0, 12.0), (5.0, 14.0)]  # source: football reasoning: 9.15 m (Law 13) +- creeping and tracking error, tight / loose
WALL_MINS_S = [1.0, 3.0]                  # source: football reasoning: a wall holds its shape for seconds before the kick
FAR_S = 10.0                    # source: football reasoning: 10 s from any stoppage = clearly open play
SWALLOWED_SHARE = 0.9           # source: football reasoning: a possession 90% under D is lost for matching

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def spell_proxy(pred, truth):
    lab = truth.copy()
    for start, end in runs((truth == "D") & ~pred):
        prev = lab[start - 1] if start > 0 else "X"
        nxt = truth[end] if end < len(truth) else "X"
        lab[start:end] = prev if prev in ("H", "A") else nxt if nxt in ("H", "A") else "D"
    lab[pred & (truth != "X")] = "D"
    return scores._evaluate(truth, lab)["spellF1"]


settings = [None] + [(k, r, m) for k in WALL_NS for r in WALL_RANGES_M for m in WALL_MINS_S]
acc = {i: {"rows": [], "isolated": 0, "far": 0, "same": 0, "swallowed": 0, "spell": []} for i in range(len(settings))}
for g, game_id in enumerate(external.games()):
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    graded = truth != "X"
    true_dead = (truth == "D") & graded
    true_idx = np.flatnonzero(true_dead)
    for i, st in enumerate(settings):
        dead_ball.WALL_ENABLED = st is not None
        if st is not None:
            dead_ball.WALL_N, (dead_ball.WALL_MIN_M, dead_ball.WALL_MAX_M), dead_ball.WALL_MIN_S = st
        pred = dead_ball.detect(frames, feats, tags)
        a = acc[i]
        a["rows"].append(scores.dead_ball(pred, truth))
        a["spell"].append(spell_proxy(pred, truth))
        for start, end in runs(true_dead):
            before = truth[start - 1] if start > 0 else "X"
            after = truth[end] if end < len(truth) else "X"
            if before in ("H", "A") and before == after:
                a["same"] += bool(pred[start:end].any())
        for team in ("H", "A"):
            for start, end in runs((truth == team) & graded):
                a["swallowed"] += pred[start:end].mean() >= SWALLOWED_SHARE
        for start, end in runs(pred & graded):
            if true_dead[start:end].any():
                continue
            a["isolated"] += 1
            gap = min(abs(true_idx - start).min(), abs(true_idx - (end - 1)).min()) if true_idx.size else np.inf
            a["far"] += gap >= FAR_S * config.FPS
    print("game {}/{} done".format(g + 1, len(external.games())), flush=True)
    del frames, feats

print("\nlabels v{}".format(external.LABELS_VERSION))
print("n  range_m     min_s   precision  recall  dead_F1  F1_even  F1_odd   isolated  far  same_touched  swallowed  spell_proxy")
for i, st in enumerate(settings):
    a = acc[i]
    name = "rule 5 off (v3.1 + 2c)" if st is None else "{}  {:4.1f}-{:<5.1f}  {:5.1f}".format(st[0], st[1][0], st[1][1], st[2])
    p, r, f = [np.mean([row[k] for row in a["rows"]]) for k in ("dead_precision", "dead_recall", "dead_F1")]
    fe = np.mean([row["dead_F1"] for row in a["rows"][0::2]])
    fo = np.mean([row["dead_F1"] for row in a["rows"][1::2]])
    print("{:22} {:9.4f}  {:6.4f}  {:7.4f}  {:7.4f}  {:6.4f}   {:8d}  {:3d}  {:12d}  {:9d}  {:11.4f}".format(
        name, p, r, f, fe, fo, a["isolated"], a["far"], a["same"], a["swallowed"], np.mean(a["spell"])))

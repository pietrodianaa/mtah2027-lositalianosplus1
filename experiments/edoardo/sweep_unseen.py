"""Scratch: rule 1c, outs the tracker does not see (v3), at possession level, v2.1 otherwise as on main.
20 external games only, never the provided match. Targets the stops we never touch
(diagnose_coverage.py: 34.9% of true dead frames). First row = rule 1c off (main).
Columns (totals over all games):
  F1 even / odd = dead_F1 on even / odd games separately (a real gain shows in both halves)
  isolated / far = our D runs touching no true stoppage (each splits a possession) / >= 10 s from any
  same_touched   = true stoppages with the SAME team before and after that we touch
  swallowed      = true possessions at least 90% covered by our D (each one is lost)
  spell_proxy    = official spellF1 with teams from the truth and D from us: a true dead stretch we
                   miss keeps the team before it (the team after if our D sits right before), so it
                   weighs splits, swallowed and merged possessions together; possessions = predicted per game
Run from the repo root:  python experiments/edoardo/sweep_unseen.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, scores  # noqa: E402

LAST_M = [1.5, 3.0, 6.0]                 # source: football reasoning: last seen on the line / within a stride / within a few metres of it
GAP_S = [1.5, 3.0, 5.0]                  # source: football reasoning: a quick throw-in takes a couple of seconds, fetching the ball ~5 s or more
HEADING = [True, False]                  # source: football reasoning: a ball going out moves towards the line it crosses
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
    slim = {"ball": frames["ball"], "half": frames["half"], "pitch": frames.get("pitch", config.PITCH),
            "player_speed": mean_speed[:, None]}   # per-frame mean speed: same average for rule 2, far less memory
    games.append((slim, {"nearest_dist": feats["nearest_dist"], "ball_v2": feats["ball_v2"]}, tags, truth))
print("loaded", len(games), "games, labels v{}".format(external.LABELS_VERSION))


def spell_proxy(pred, truth):
    """Official spellF1 when everything but dead ball is perfect (others = truth)."""
    lab = truth.copy()
    for start, end in runs((truth == "D") & ~pred):
        prev = lab[start - 1] if start > 0 else "X"
        nxt = truth[end] if end < len(truth) else "X"
        lab[start:end] = prev if prev in ("H", "A") else nxt if nxt in ("H", "A") else "D"
    lab[pred & (truth != "X")] = "D"
    out = scores._evaluate(truth, lab)
    return out["spellF1"], out["n_pred_spells"]


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

print("\nlast_m  gap_s  head   precision  recall  dead_F1  F1_even  F1_odd   isolated  far  same_touched  swallowed  spell_proxy  possessions")
for ref_name, make in (("no dead ball at all", lambda t: np.zeros(len(t), dtype=bool)),
                       ("perfect dead ball", lambda t: (t == "D"))):
    res = [spell_proxy(make(truth), truth) for _, _, _, truth in games]
    print("{:18} spell_proxy {:.4f}, possessions {:.1f}".format(ref_name, np.mean([r[0] for r in res]), np.mean([r[1] for r in res])))
settings = [(None, None, None)] + [(m, g, h) for h in HEADING for m in LAST_M for g in GAP_S]
for last_m, gap_s, heading in settings:
    dead_ball.UNSEEN_ENABLED = last_m is not None
    if last_m is not None:
        dead_ball.UNSEEN_LAST_M, dead_ball.UNSEEN_GAP_S, dead_ball.UNSEEN_HEADING = last_m, gap_s, heading
        name = "{:6.1f}  {:5.1f}  {:5}".format(last_m, gap_s, "yes" if heading else "no")
    else:
        name = "rule 1c off (main)"
    rows, isolated, far, same_touched, swallowed, spells, n_spells = [], 0, 0, 0, 0, [], []
    for frames, feats, tags, truth in games:
        pred = dead_ball.detect(frames, feats, tags)
        rows.append(scores.dead_ball(pred, truth))
        sp, ns = spell_proxy(pred, truth)
        spells.append(sp)
        n_spells.append(ns)
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
    print("{:18} {:9.4f}  {:6.4f}  {:7.4f}  {:7.4f}  {:6.4f}   {:8d}  {:3d}  {:12d}  {:9d}  {:11.4f}  {:11.1f}".format(
        name, p, r, f, f_even, f_odd, isolated, far, same_touched, swallowed, np.mean(spells), np.mean(n_spells)), flush=True)

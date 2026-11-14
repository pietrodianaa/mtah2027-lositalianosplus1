"""Scratch: is rule 1c's spellF1 gain robust to how a missed stop gets filled? Three fills of the true
dead frames we miss: the team before the stop (as in sweep_unseen.py), the team after it, or split
at the middle. Teams otherwise from the truth (others perfect)., at possession level, v2.1 otherwise as on main.
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
Run from the repo root:  python experiments/edoardo/check_proxy_fill.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, scores  # noqa: E402

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


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def spell_proxy(pred, truth, fill):
    lab = truth.copy()
    for start, end in runs((truth == "D") & ~pred):
        prev = lab[start - 1] if start > 0 else "X"
        nxt = truth[end] if end < len(truth) else "X"
        prev = prev if prev in ("H", "A") else nxt
        nxt = nxt if nxt in ("H", "A") else prev
        if prev not in ("H", "A"):
            lab[start:end] = "D"
        elif fill == "before":
            lab[start:end] = prev
        elif fill == "after":
            lab[start:end] = nxt
        else:
            mid = (start + end) // 2
            lab[start:mid], lab[mid:end] = prev, nxt
    lab[pred & (truth != "X")] = "D"
    return scores._evaluate(truth, lab)["spellF1"]


settings = [("main (rule 1c off)", None, None, None), ("v3: 6 m, 2.0 m/s, 0.5", 6.0, 2.0, 0.5),
            ("6 m, no walk filter", 6.0, 2.0, 0.0), ("3 m, 1.5 m/s, 0.7", 3.0, 1.5, 0.7)]
print("\n{:<24} {:>12} {:>12} {:>12}".format("setting", "fill before", "fill after", "fill split"))
for name, last_m, walk_ms, share in settings:
    dead_ball.UNSEEN_ENABLED = last_m is not None
    if last_m is not None:
        dead_ball.UNSEEN_LAST_M, dead_ball.UNSEEN_WALK_MS, dead_ball.UNSEEN_WALK_SHARE = last_m, walk_ms, share
    res = {f: [] for f in ("before", "after", "split")}
    for frames, feats, tags, truth in games:
        pred = dead_ball.detect(frames, feats, tags)
        for f in res:
            res[f].append(spell_proxy(pred, truth, f))
    print("{:<24} {:12.4f} {:12.4f} {:12.4f}".format(name, *[np.mean(res[f]) for f in ("before", "after", "split")]), flush=True)

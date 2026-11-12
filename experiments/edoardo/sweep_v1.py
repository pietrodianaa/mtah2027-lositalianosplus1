"""Scratch: try every combination of the dead ball v1 parameters on the 20 external games.

External SkillCorner games only, never the provided match. Prints only mean scores per setting,
averaged over games exactly like `python run.py --tune --role dead_ball`.
Run from the repo root:  python experiments/edoardo/sweep_v1.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import dead_ball, external, scores  # noqa: E402

MARGINS_M = [0.0, 0.1, 0.25, 0.5, 1.0]  # source: external v1: run 1 rose as the margin fell (1.5 -> 0.5 m), so look lower
MAX_FILLS_S = [0.0, 5.0, 10.0, 20.0, 40.0]  # source: external v1: run 1 flat from 20 to 90 s; 0 = no fill, does filling help at all?
MIN_DEADS_S = [1.0, 2.0]              # source: external v1: run 1 moved < 0.01 between 1 and 3 s

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")

# load once, keep only what v1 uses (players would take GBs of memory)
games = []
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    games.append(({"ball": frames["ball"], "half": frames["half"], "pitch": frames["pitch"]}, tags, truth))
print("loaded", len(games), "games, labels v{}".format(external.LABELS_VERSION))

print("margin_m  max_fill_s  min_dead_s   precision  recall  dead_F1")
for margin in MARGINS_M:
    for max_fill in MAX_FILLS_S:
        for min_dead in MIN_DEADS_S:
            dead_ball.MARGIN_M = margin
            dead_ball.MAX_FILL_S = max_fill
            dead_ball.MIN_DEAD_S = min_dead
            rows = [scores.dead_ball(dead_ball.detect(frames, {}, tags), truth) for frames, tags, truth in games]
            precision = np.mean([r["dead_precision"] for r in rows])
            recall = np.mean([r["dead_recall"] for r in rows])
            f1 = np.mean([r["dead_F1"] for r in rows])
            mark = "  <- v1" if (margin, max_fill, min_dead) == (1.0, 40.0, 2.0) else ""
            print("{:8.1f}  {:10.0f}  {:10.0f}   {:9.4f}  {:6.4f}  {:7.4f}{}".format(
                margin, max_fill, min_dead, precision, recall, f1, mark), flush=True)

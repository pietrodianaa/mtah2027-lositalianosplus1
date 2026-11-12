"""Scratch: hold D after an out until the restart; try the restart parameters on the 20 external games.

External SkillCorner games only, never the provided match. Prints only mean scores per setting,
averaged over games exactly like `python run.py --tune --role dead_ball`. First row = v1.1 (no hold).
Run from the repo root:  python experiments/edoardo/sweep_hold.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import dead_ball, external, scores  # noqa: E402

RESTART_SPEEDS_MS = [4.0, 6.0, 9.0]   # source: football reasoning: around a ball thrown or kicked at the restart
RESTART_ZONES_M = [6.0, 10.0, 20.0]   # source: football reasoning: goal area 5.5 m deep, plus buffer
MAX_FILLS_S = [20.0, 40.0, 60.0]      # source: football reasoning: throw-in ~15 s, goal kick ~29 s, corner ~34 s (Opta)

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")

# load once, keep only what the rule uses (players would take GBs of memory)
games = []
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    games.append(({"ball": frames["ball"], "half": frames["half"], "pitch": frames["pitch"]}, tags, truth))
print("loaded", len(games), "games, labels v{}".format(external.LABELS_VERSION))


def mean_scores():
    rows = [scores.dead_ball(dead_ball.detect(frames, {}, tags), truth) for frames, tags, truth in games]
    return [np.mean([r[key] for r in rows]) for key in ("dead_precision", "dead_recall", "dead_F1")]


print("hold  speed_ms  zone_m  max_fill_s   precision  recall  dead_F1")
dead_ball.HOLD_UNTIL_RESTART = False
print("  no         -       -  {:10.0f}   {:9.4f}  {:6.4f}  {:7.4f}  <- v1.1".format(
    dead_ball.MAX_FILL_S, *mean_scores()), flush=True)
dead_ball.HOLD_UNTIL_RESTART = True
for speed in RESTART_SPEEDS_MS:
    for zone in RESTART_ZONES_M:
        for max_fill in MAX_FILLS_S:
            dead_ball.RESTART_SPEED_MS = speed
            dead_ball.RESTART_ZONE_M = zone
            dead_ball.MAX_FILL_S = max_fill
            print(" yes  {:8.1f}  {:6.1f}  {:10.0f}   {:9.4f}  {:6.4f}  {:7.4f}".format(
                speed, zone, max_fill, *mean_scores()), flush=True)

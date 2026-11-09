"""Aggregate checks of the SkillCorner conversion (external data only, never the match).

1. How often SkillCorner's per-frame possession.group disagrees with the phase's team in
   possession (the truth uses the phase team).
2. Share of between-phase gaps that touch a restart event in dynamic_events (an event with
   game_interruption_before starting in the gap or within TOUCH_S after it, or one with
   game_interruption_after ending within TOUCH_S before it or in the gap).

Run:  .venv/bin/python experiments/pietro/external_check.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import external  # noqa: E402

TOUCH_S = 1.0                  # source: football reasoning: a restart event is logged within about a second of the stoppage
touch = int(TOUCH_S * external.SC_FPS)
SHORT_S = 2.0                  # source: football reasoning: report split only, quick restarts vs longer stoppages
short = int(SHORT_S * external.SC_FPS)
agree = disagree = 0
gaps = {"short": [0, 0], "long": [0, 0]}          # [touching a restart, total]
for g in external.games():
    game = external.read_game(g)
    for first, last, team in game["phases"]:
        grp = game["group"][first:last + 1]
        if team is not None:
            agree += int((grp == team).sum())
            disagree += int(((grp != team) & (grp != external.NONE)).sum())
    events = list(csv.DictReader(open(external._path(g, "dynamic_events.csv"))))
    before = [int(e["frame_start"]) for e in events if e["game_interruption_before"]]
    after = [int(e["frame_end"]) for e in events if e["game_interruption_after"]]
    period = game["period"]
    for (_, prev_last, _), (next_first, _, _) in zip(game["phases"], game["phases"][1:]):
        n = next_first - prev_last - 1
        if n <= 0 or period[prev_last] != period[next_first]:
            continue
        hit = any(prev_last < f <= next_first + touch for f in before) or \
            any(prev_last - touch <= f < next_first for f in after)
        key = "short" if n < short else "long"
        gaps[key][0] += hit
        gaps[key][1] += 1
print("games", len(external.games()))
print("group vs phase team: disagree {:.3f} of frames with a group ({} / {})".format(
    disagree / max(agree + disagree, 1), disagree, agree + disagree))
for key, (hit, total) in gaps.items():
    print("{} gaps (< or >= {} s): {}/{} touch a restart = {:.3f}".format(
        key, SHORT_S, hit, total, hit / max(total, 1)))

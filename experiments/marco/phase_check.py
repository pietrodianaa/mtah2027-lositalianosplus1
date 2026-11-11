"""Aggregate checks of the labels-v1 truth rules, from SkillCorner events only (no tracking needed).

External data only, never the match. Prints totals over the 20 games, no frame-level tables.

1. Gaps between phases of play: does any SkillCorner player_possession overlap one?
   (v1 labels every gap D, so a possession inside a gap would mean ball in play labelled D.)
2. player_possession events that sit entirely inside a phase of the other team: how long are
   they? (Short ones are touches or deflections, which our rules say do not change possession.)
3. Restarts: is the phase team at the start of a restart possession the team taking it?

Run:  python experiments/marco/phase_check.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import external  # noqa: E402

LONG_S = 2.0                   # source: football reasoning: a 2 s spell on the ball is control, not a touch (report split only)
long_frames = int(LONG_S * external.SC_FPS)
RESTARTS = ("throw_in", "goal_kick", "free_kick", "corner")   # source: data format (game_interruption_before values)

n_gaps = gaps_with_poss = 0
n_poss = poss_in_gap = 0
other = []                     # durations (10 fps frames) of possessions wholly inside the other team's phase
restart_same = restart_other = 0
games = sorted(os.listdir(external.MATCHES_DIR), key=int)
for g in games:
    phases = sorted(csv.DictReader(open(external._path(g, "phases_of_play.csv"))),
                    key=lambda p: int(p["frame_start"]))
    poss = [e for e in csv.DictReader(open(external._path(g, "dynamic_events.csv")))
            if e["event_type"] == "player_possession"]
    n = max([int(p["frame_end"]) for p in phases] + [int(e["frame_end"]) for e in poss]) + 1
    team = np.zeros(n, dtype=np.int64)             # 0 = outside every phase
    for p in phases:
        team[int(p["frame_start"]):int(p["frame_end"]) + 1] = int(p["team_in_possession_id"])
    gap = np.zeros(n, dtype=bool)
    for a, b in zip(phases, phases[1:]):
        if a["period"] != b["period"]:
            continue
        first, stop = int(a["frame_end"]) + 1, int(b["frame_start"])
        if stop > first:
            n_gaps += 1
            gap[first:stop] = True
            gaps_with_poss += any(int(e["frame_start"]) < stop and int(e["frame_end"]) >= first
                                  for e in poss)
    for e in poss:
        first, last, tid = int(e["frame_start"]), int(e["frame_end"]), int(e["team_id"])
        seg = team[first:last + 1]
        n_poss += 1
        poss_in_gap += bool(gap[first:last + 1].any())
        if (seg != 0).all() and not (seg == tid).any():
            other.append(last - first + 1)
        if e["game_interruption_before"].startswith(RESTARTS) and team[first] != 0:
            restart_same += team[first] == tid
            restart_other += team[first] != tid

other = np.array(other)
print("games", len(games))
print("gaps between phases (same period): {}, containing a player_possession: {}".format(n_gaps, gaps_with_poss))
print("player_possessions: {}, overlapping a gap: {}".format(n_poss, poss_in_gap))
print("possessions wholly inside the other team's phase: {} ({:.3f}); median {:.1f} s; >= {} s: {} ({:.0f} s in total)".format(
    len(other), len(other) / n_poss, np.median(other) / external.SC_FPS, LONG_S,
    int((other >= long_frames).sum()), other[other >= long_frames].sum() / external.SC_FPS))
print("restart possessions: phase team = taker {}, other team {} ({:.3f})".format(
    restart_same, restart_other, restart_other / max(restart_same + restart_other, 1)))

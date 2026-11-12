"""Scratch: where do the dead-ball errors come from? 20 external games only, never the provided match.

Prints only totals over all games (no frame indices):
  - false D frames by ball state (past a line / missing / visible inside the pitch);
  - predicted D runs that touch no true dead stretch ("isolated": they split a possession)
    vs false frames attached to a true dead stretch ("boundary": they only move an edge);
  - true dead stretches touched at least in part (recall by stretch, not by frame).
Runs v1.1 (HOLD_UNTIL_RESTART off) and v1.2 (on) side by side.
Run from the repo root:  python experiments/edoardo/diagnose_dead.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import dead_ball, external  # noqa: E402

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")

games = []
for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    games.append(({"ball": frames["ball"], "half": frames["half"], "pitch": frames["pitch"]}, tags, truth))
print("loaded", len(games), "games, labels v{}".format(external.LABELS_VERSION))


def runs(mask):
    """(start, end) of every run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


for hold in (False, True):
    dead_ball.HOLD_UNTIL_RESTART = hold
    t = {"pred": 0, "false": 0, "false_past_line": 0, "false_missing": 0, "false_inside": 0,
         "runs": 0, "isolated_runs": 0, "isolated_frames": 0, "true_stretches": 0, "touched": 0}
    for frames, tags, truth in games:
        graded = truth != "X"
        true_dead = (truth == "D") & graded
        pred = dead_ball.detect(frames, {}, tags) & graded
        false = pred & ~true_dead
        x, y = frames["ball"][:, 0], frames["ball"][:, 1]
        length, width = frames["pitch"]
        missing = np.isnan(x)
        with np.errstate(invalid="ignore"):
            past_line = ~missing & ((x < 0) | (x > length) | (y < 0) | (y > width))
        t["pred"] += pred.sum()
        t["false"] += false.sum()
        t["false_past_line"] += (false & past_line).sum()
        t["false_missing"] += (false & missing).sum()
        t["false_inside"] += (false & ~missing & ~past_line).sum()
        for start, end in runs(pred):
            t["runs"] += 1
            if not true_dead[start:end].any():
                t["isolated_runs"] += 1
                t["isolated_frames"] += end - start
        for start, end in runs(true_dead):
            t["true_stretches"] += 1
            t["touched"] += bool(pred[start:end].any())
    f = max(t["false"], 1)
    print("\n== {} ==".format("v1.2 (hold until restart)" if hold else "v1.1 (no hold)"))
    print("predicted D frames: {}  false: {} ({:.1%} of predicted)".format(t["pred"], t["false"], t["false"] / max(t["pred"], 1)))
    print("false D by ball state: past a line {:.1%}, missing {:.1%}, visible inside {:.1%}".format(
        t["false_past_line"] / f, t["false_missing"] / f, t["false_inside"] / f))
    print("predicted D runs: {}  isolated (touch no true dead stretch): {} ({:.1f} per game)".format(
        t["runs"], t["isolated_runs"], t["isolated_runs"] / len(games)))
    print("false D frames in isolated runs: {:.1%}  at the edge of a true dead stretch: {:.1%}".format(
        t["isolated_frames"] / f, 1 - t["isolated_frames"] / f))
    print("true dead stretches: {} ({:.1f} per game)  touched at least in part: {:.1%}".format(
        t["true_stretches"], t["true_stretches"] / len(games), t["touched"] / max(t["true_stretches"], 1)))

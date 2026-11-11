"""Scratch bench for control ideas on the EXTERNAL matches only (never the provided game).

    python experiments/damiano/control_lab.py

Loads the 20 SkillCorner games once and scores candidate detect() functions with the same
scores.control() that `run.py --tune --role control` uses. Numbers chosen here still have to be
confirmed with `python run.py --tune --role control` before they go into possession/control.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import external, features, scores  # noqa: E402

GAMES = None


def games():
    global GAMES
    if GAMES is None:
        GAMES = []
        for g in external.games():
            frames, tags, truth = external.load(g)
            GAMES.append((frames, features.compute(frames), tags, truth))
    return GAMES


def score(detect, subset=None):
    rows = []
    for i, (frames, feats, tags, truth) in enumerate(games()):
        if subset is not None and i % 2 != subset:
            continue
        rows.append(scores.control(detect(frames, feats, tags), truth))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def show(name, detect):
    a, b = score(detect, 0), score(detect, 1)
    allr = {k: (a[k] + b[k]) / 2 for k in a}
    print("{:<40} spellF1 {:.4f} (even {:.4f} odd {:.4f}) acc {:.3f} cov {:.3f} spells {:.0f}".format(
        name, allr["spellF1"], a["spellF1"], b["spellF1"], allr["team_accuracy"],
        allr["evidence_coverage"], allr["n_pred_spells"]))

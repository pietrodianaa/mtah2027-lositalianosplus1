"""[External data + doc] The external matches we may tune on, and per-role scores on them.

Rules (organisers, 3 Nov 2026): never tune on the ground truth, the public 5 minutes, the video
or the provided game. External data is optional: football reasoning is the default source of
every number.

Dataset: SkillCorner open data (github.com/SkillCorner/opendata), MIT license, free: 10 A-League
2024/25 matches, broadcast tracking of players and ball at 10 fps with per-frame possession info.
Clone it into external_data/ (gitignored); never commit it. Credit SkillCorner in the tech doc.

Contract (what the owner implements)
------------------------------------
games()          -> list of game ids found locally (empty until set up)
load(game_id)    -> (frames, tags, truth)
                    frames: same dict as io.load_frames(), RESAMPLED TO 25 FPS (metres, m/s, NaN)
                    tags:   [] (SkillCorner has no Smart Tagging) or a list shaped like io.load_tags()
                    truth:  (N,) 'H' / 'A' / 'D' / 'X', derived BY CODE from SkillCorner's
                            possession info under OUR rules: loose ball stays with the last team,
                            out of play = D. Home = the team listed first.
LABELS_VERSION   bump it (and post on the board) whenever the conversion changes: every score is
                 recorded with the version it was computed on, so old and new are never compared.

Conversion decisions to write in the tech doc: resampling 10 -> 25 fps; possession mapping; how
ball positions are hidden to mimic our ~41% missing ball (do it once, seeded, saved with the
version, so every role sees the same degraded input).

evaluate(role) is already written (lead): it runs only the role asked for, with the other roles
replaced by the truth (possession/scores.py), and averages over games.
"""

import os

import numpy as np

from . import control, dead_ball, features, postprocess, scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = os.path.join(ROOT, "external_data")
LABELS_VERSION = 0             # source: bookkeeping (0 = not set up yet; bump on every change)

ROLES = ("dead_ball", "control", "postprocess", "full")


def games():
    return []      # implement: list the games found under external_data/


def load(game_id):
    raise NotImplementedError("external data not set up yet: see the module docstring")


def _score_one(role, frames, tags, truth):
    if role == "postprocess":
        return scores.postprocess(postprocess.finalize, truth, frames)
    feats = features.compute(frames)
    if role == "dead_ball":
        return scores.dead_ball(dead_ball.detect(frames, feats, tags), truth)
    if role == "control":
        return scores.control(control.detect(frames, feats, tags), truth)
    from . import pipeline
    return scores.full(pipeline.label(frames, tags), truth)


def evaluate(role, game_ids=None):
    if role not in ROLES:
        raise ValueError("role must be one of {}".format(ROLES))
    rows = [_score_one(role, *load(g)) for g in (game_ids or games())]
    result = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    result.update({"role": role, "n_games": len(rows), "labels_version": LABELS_VERSION,
                   "reference_version": scores.REFERENCE_VERSION})
    return result

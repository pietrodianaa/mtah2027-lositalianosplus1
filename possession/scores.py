"""[Lead, frozen once agreed] One score per role, each independent of the other roles' work.

Principle: every role is scored with the OTHER roles replaced by the true labels (or by a fixed
reference), so a merge by one person can never move another person's score.

    role          scored on                                         replaced by truth / fixed
    -----------   -----------------------------------------------   ---------------------------
    dead_ball     D precision / recall / F1 vs true dead frames     nothing else is involved
    control       spellF1 of its evidence after a FIXED reference   dead ball = true D frames;
                  smoother (a frozen copy of the official baseline   smoother = frozen reference
                  vote), plus team accuracy on frames with evidence  (never the live postprocess)
    postprocess   spellF1 of finalize() on SYNTHETIC evidence made   control = noisy truth;
                  from the truth with fixed, seeded noise            dead ball = true D frames
    full          the real pipeline end to end                      nothing: the joint check,
                                                                     run by the lead at milestones
Features have no score of their own here: they are checked by their owner (e.g. ball-gap filling
error in metres on external data) and protected by the add-only contract (tests/).

All spellF1 / macroF1 numbers come from the official evaluation/possession_metrics.py.
Truth arrays are 'H' / 'A' / 'D' / 'X' at 25 fps (external.py resamples to 25 fps).
Changing anything in this file changes everyone's numbers: only by team agreement, announced on
the board, with REFERENCE_VERSION bumped.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

REFERENCE_VERSION = 1          # source: bookkeeping (bump when this file changes)

# Frozen reference smoother for scoring control.   source: official baseline default (±50 frames)
REFERENCE_SMOOTH_WINDOW = 50

# Synthetic evidence for scoring post-processing.   source: football reasoning + DATA.md
NOISE_SEED = 7
NOISE_MISSING_SHARE = 0.41     # source: DATA.md (ball missing on ~41% of graded frames)
NOISE_MISSING_BURST_S = 1.0    # source: football reasoning (tracker loses the ball in short bursts)
NOISE_FLIP_SHARE = 0.10        # source: football reasoning (wrong nearest team, duels and passes)
NOISE_FLIP_BURST_S = 0.4       # source: football reasoning (a pass or duel lasts under half a second)
FPS = 25                       # source: data format

HOME, AWAY, NONE = 0, 1, -1


def _graded(truth):
    return np.asarray(truth) != "X"


def _evaluate(truth, labels):
    result = pm.evaluate(pm.label_codes(np.asarray(truth)), pm.label_codes(np.asarray(labels)))
    return {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "n_pred_spells": result["n_pred_spells"], "n_true_spells": result["n_gt_spells"]}


def _reference_smooth(evidence):
    """Frozen copy of the official baseline vote. Never import postprocess here."""
    evidence = np.asarray(evidence)
    window = REFERENCE_SMOOTH_WINDOW
    home_c = np.concatenate(([0], np.cumsum(evidence == HOME)))
    away_c = np.concatenate(([0], np.cumsum(evidence == AWAY)))
    idx = np.arange(len(evidence))
    low, high = np.maximum(0, idx - window), np.minimum(len(evidence), idx + window + 1)
    home, away = home_c[high] - home_c[low], away_c[high] - away_c[low]
    labels = np.full(len(evidence), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def dead_ball(pred_dead, truth):
    truth = np.asarray(truth)
    graded = _graded(truth)
    true_dead = (truth == "D") & graded
    pred = np.asarray(pred_dead, dtype=bool) & graded
    tp = int((pred & true_dead).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(true_dead.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"dead_precision": precision, "dead_recall": recall, "dead_F1": f1}


def control(evidence, truth):
    truth = np.asarray(truth)
    evidence = np.asarray(evidence)
    labels = _reference_smooth(evidence)
    labels[truth == "D"] = "D"                               # dead ball replaced by truth
    alive = np.isin(truth, ["H", "A"])
    has = alive & (evidence != NONE)
    team_truth = np.where(truth == "H", HOME, AWAY)
    out = _evaluate(truth, labels)
    out["team_accuracy"] = float((evidence[has] == team_truth[has]).mean()) if has.any() else 0.0
    out["evidence_coverage"] = float(has.sum() / max(alive.sum(), 1))
    return out


def synthetic_evidence(truth):
    """Truth turned into noisy control evidence with fixed, seeded noise (no live control code)."""
    truth = np.asarray(truth)
    rng = np.random.default_rng(NOISE_SEED)
    evidence = np.full(len(truth), NONE, dtype=np.int8)
    evidence[truth == "H"], evidence[truth == "A"] = HOME, AWAY

    def bursts(share, burst_s):
        mask = np.zeros(len(truth), dtype=bool)
        burst = max(int(burst_s * FPS), 1)
        n = int(share * len(truth) / burst)
        for start in rng.integers(0, max(len(truth) - burst, 1), size=n):
            mask[start:start + burst] = True
        return mask

    flip = bursts(NOISE_FLIP_SHARE, NOISE_FLIP_BURST_S) & (evidence != NONE)
    evidence[flip] = 1 - evidence[flip]
    evidence[bursts(NOISE_MISSING_SHARE, NOISE_MISSING_BURST_S)] = NONE
    return evidence


def postprocess(finalize, truth, frames=None):
    truth = np.asarray(truth)
    labels = finalize(synthetic_evidence(truth), truth == "D", frames)   # others = truth
    return _evaluate(truth, labels)


def full(labels, truth):
    return _evaluate(truth, labels)

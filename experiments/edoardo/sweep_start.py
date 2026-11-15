"""Scratch: EARLIER STARTS of the dead runs main already finds, scored on the joint pipeline (live
control + live postprocess). 20 external games only, never the provided match. Means only.
Why (decomposition on main, labels v1): joint 0.5333; true start on the stops we touch 0.5646,
true end 0.5421; untouched stops added 0.6009. The start is where touched stops lose.
  out     rule 1 backdate to the last touch, longer / stricter (OUT_BACKDATE_S, OUT_TOUCH_M)
  walk    every dead run grows backwards while the players already walk (sweep_extend.grow_walk)
  1c      rule 1c with the ball last seen farther from a line (UNSEEN_LAST_M)
Run from the repo root:  python experiments/edoardo/sweep_start.py
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession import control, dead_ball, external, features, postprocess, scores  # noqa: E402
from sweep_extend import grow_walk  # noqa: E402

# name, dead_ball overrides, walk-back (m/s, cap s) or None.   source of every value: football
# reasoning: a ball rolls or flies out for 1-3 s after the last touch; players slow from jogging
# (2 m/s) to walking (1-1.5 m/s) over a few seconds after a whistle; a long clearance can leave
# the pitch from anywhere
VARIANTS = [("main", {}, None)]
VARIANTS += [("out {} m {} s".format(m, s), {"OUT_TOUCH_M": m, "OUT_BACKDATE_S": s}, None)
             for m in (2.5, 1.5) for s in (2.0, 3.0)]
VARIANTS += [("walk back {} m/s cap {} s".format(ms, cap), {}, (ms, cap))
             for ms in (1.25, 1.5, 1.75) for cap in (2.0, 4.0, 8.0)]
VARIANTS += [("1c last 15 m", {"UNSEEN_LAST_M": 15.0}, None),
             ("1c last 15 m + walk back 1.5/4", {"UNSEEN_LAST_M": 15.0}, (1.5, 4.0)),
             ("1c last 15 m + walk back 1.25/4", {"UNSEEN_LAST_M": 15.0}, (1.25, 4.0)),
             ("1c any + walk back 1.5/4", {"UNSEEN_LAST_M": 999.0}, (1.5, 4.0))]
DEFAULTS = {k: getattr(dead_ball, k) for _, p, _ in VARIANTS for k in p}


def one_game(game_id):
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    evidence = control.detect(frames, feats, tags)
    synth = scores.synthetic_evidence(truth)
    rows = []
    for name, params, walk in VARIANTS:
        for k, v in DEFAULTS.items():
            setattr(dead_ball, k, params.get(k, v))
        dead = dead_ball.detect(frames, feats, tags)
        if walk:
            dead = grow_walk(dead, frames, walk[0], walk[1], fwd=False)
        d = scores.dead_ball(dead, truth)
        real = scores._evaluate(truth, postprocess.finalize(synth, dead, frames))
        full = scores._evaluate(truth, postprocess.finalize(evidence, dead, frames))
        rows.append((d["dead_F1"], d["dead_precision"], d["dead_recall"], real["spellF1"],
                     full["spellF1"], full["macroF1"], full["n_pred_spells"]))
    return rows


if __name__ == "__main__":
    games = external.games()
    if not games:
        sys.exit("No external games: run tools/get_external_data.sh first.")
    with Pool(4) as pool:
        per_game = np.array(pool.map(one_game, games))
    print("labels v{} reference v{} games {}".format(external.LABELS_VERSION, scores.REFERENCE_VERSION, len(games)))
    print("{:34s} {:>6s} {:>6s} {:>6s} | {:>6s} | {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>5s}".format(
        "variant", "F1", "P", "R", "real.", "JOINT", "even", "odd", "macro", "spells", "wins"))
    base = per_game[:, 0, 4]
    for v, (name, _, _) in enumerate(VARIANTS):
        r = per_game[:, v]
        m = r.mean(axis=0)
        print("{:34s} {:6.4f} {:6.4f} {:6.4f} | {:6.4f} | {:6.4f} {:6.4f} {:6.4f} {:6.4f} {:6.1f} {:>5s}".format(
            name, m[0], m[1], m[2], m[3], m[4], r[0::2, 4].mean(), r[1::2, 4].mean(), m[5], m[6],
            "{}/{}".format(int((r[:, 4] > base).sum()), len(games))))

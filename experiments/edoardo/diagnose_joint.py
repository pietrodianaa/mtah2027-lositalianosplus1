"""Scratch: where the JOINT score (live control + live postprocess) loses because of dead ball.
20 external games only, never the provided match. Means only.
Main's dead runs are swapped for versions repaired with the truth, one repair at a time, to see
which kind of error costs possessions: false D, stops never touched (by length), or the start /
end of the stops we do touch. Diagnostic only: no number is chosen from it.
Run from the repo root:  python experiments/edoardo/diagnose_joint.py
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, control, dead_ball, external, features, postprocess, scores  # noqa: E402

LENGTH_BINS_S = [(0, 3), (3, 6), (6, 12), (12, 10 ** 6)]   # source: football reasoning: quick restart / throw-in / free kick or goal kick / long stop


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def one_game(game_id):
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    evidence = control.detect(frames, feats, tags)
    dead = dead_ball.detect(frames, feats, tags)
    true_dead = truth == "D"
    stops = runs(true_dead)
    n = len(truth)
    touched, untouched, start_fix, end_fix = (np.zeros(n, dtype=bool) for _ in range(4))
    by_len = {b: np.zeros(n, dtype=bool) for b in LENGTH_BINS_S}
    counts = {b: 0 for b in LENGTH_BINS_S}
    for s, e in stops:
        hit = np.flatnonzero(dead[s:e])
        if len(hit):
            touched[s:e] = True
            start_fix[s:s + hit[0]] = True
            end_fix[s + hit[-1]:e] = True
        else:
            untouched[s:e] = True
            for lo, hi in LENGTH_BINS_S:
                if lo * config.FPS <= e - s < hi * config.FPS:
                    by_len[(lo, hi)][s:e] = True
                    counts[(lo, hi)] += 1
    cases = [("main", dead, 0),
             ("no dead ball at all", np.zeros(n, dtype=bool), 0),
             ("true dead ball", true_dead, 0),
             ("main, false D removed", dead & true_dead, 0),
             ("main + touched stops completed", dead | touched, 0),
             ("  only their start", dead | start_fix, 0),
             ("  only their end", dead | end_fix, 0),
             ("main + untouched stops added", dead | untouched, len(runs(untouched)))]
    cases += [("  only those of {}-{} s".format(lo, "inf" if hi > 999 else hi), dead | by_len[(lo, hi)],
               counts[(lo, hi)]) for lo, hi in LENGTH_BINS_S]
    out = []
    for name, d, count in cases:
        r = scores._evaluate(truth, postprocess.finalize(evidence, d, frames))
        out.append((name, r["spellF1"], r["n_pred_spells"], r["n_true_spells"], count))
    return out


if __name__ == "__main__":
    games = external.games()
    if not games:
        sys.exit("No external games: run tools/get_external_data.sh first.")
    with Pool(4) as pool:
        res = pool.map(one_game, games)
    print("labels v{} reference v{} games {}".format(external.LABELS_VERSION, scores.REFERENCE_VERSION, len(games)))
    print("{:34s} {:>6s} {:>6s} {:>6s} {:>7s} {:>10s}".format("dead ball used", "JOINT", "even", "odd", "spells", "stops/game"))
    for i, (name, *_) in enumerate(res[0]):
        v = np.array([r[i][1:] for r in res], dtype=float)
        print("{:34s} {:6.4f} {:6.4f} {:6.4f} {:7.1f} {:>10s}".format(
            name, v[:, 0].mean(), v[0::2, 0].mean(), v[1::2, 0].mean(), v[:, 1].mean(),
            "{:.1f}".format(v[:, 3].mean()) if v[:, 3].any() else ""))
    print("true possessions per game {:.1f}".format(np.mean([r[0][3] for r in res])))

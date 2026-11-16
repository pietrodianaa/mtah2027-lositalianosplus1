"""Extra per-game arrays for turnover_lab.py: own-goal side of each team per frame (from where its
deepest player, the keeper, stands over the half). External practice set only.
Run once: PYTHONPATH=. python experiments/marco/turnover_lab_cache.py -> external_data/marco_lab/"""
import os
import warnings
from multiprocessing import Pool

import numpy as np

from possession import external

OUT = os.path.join(external.EXTERNAL_DIR, "marco_lab")


def one(g):
    path = os.path.join(OUT, "{}.npz".format(g))
    if os.path.isfile(path):
        return g
    frames, _, _ = external.load(g)
    x, team, half = frames["players"][:, :, 0], frames["team"], np.asarray(frames["half"])
    goal_sign = np.zeros((len(x), 2), dtype=np.int8)       # -1: own goal at x<0, +1: at x>0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = [np.nanmin(x[:, team == s], axis=1) for s in (0, 1)]
        hi = [np.nanmax(x[:, team == s], axis=1) for s in (0, 1)]
        for h in np.unique(half[half > 0]):
            m = half == h
            # the team whose deepest player is further toward x<0 defends that goal (its keeper)
            left = 0 if np.nanmean(lo[0][m]) < np.nanmean(lo[1][m]) else 1
            goal_sign[m, left], goal_sign[m, 1 - left] = -1, 1
    np.savez_compressed(path, goal_sign=goal_sign)
    return g


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with Pool(4) as pool:
        for g in pool.imap_unordered(one, external.games()):
            print("cached", g, flush=True)

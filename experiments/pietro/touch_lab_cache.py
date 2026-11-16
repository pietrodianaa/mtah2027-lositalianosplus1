"""Cache per external game what the touch-filter lab needs, so each control variant costs seconds:
features, live dead ball, team-area side, ball speed, truth. External practice set only.
Run once: PYTHONPATH=. python experiments/pietro/touch_lab_cache.py
Cache goes to external_data/touch_lab/ (gitignored with the rest of external_data/)."""
import os
from multiprocessing import Pool

import numpy as np

from possession import control, dead_ball, external, features

OUT = os.path.join(external.EXTERNAL_DIR, "touch_lab")
KEEP = ["ball_visible", "nearest_dist_H", "nearest_dist_A", "nearest_team",
        "nearest_dist_H_v2", "nearest_dist_A_v2", "ball_v2"]


def one(g):
    path = os.path.join(OUT, "{}.npz".format(g))
    if os.path.isfile(path):
        return g
    frames, tags, truth = external.load(g)
    feats = features.compute(frames)
    dead = dead_ball.detect(frames, feats, tags)
    area = control._team_area_side(frames)
    np.savez_compressed(path, truth=truth, dead=dead, area=area, ball=frames["ball"],
                        half=np.asarray(frames.get("half")),
                        **{k: feats[k] for k in KEEP})
    return g


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with Pool(3) as pool:
        for g in pool.imap_unordered(one, external.games()):
            print("cached", g, flush=True)

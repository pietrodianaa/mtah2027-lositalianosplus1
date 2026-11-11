"""ball_v2 check on external matches (SkillCorner, labels v1); never the provided match.

The cached external ball has extra seeded 1 s bursts hidden (to ~41% missing, like our match).
Here the true positions under those bursts are known, so the filled positions can be measured.
For each maximum gap length: share of hidden-but-known graded frames that ball_v2 fills,
gap-filling error in metres (median, 90th percentile), and how often the nearest team to the
filled ball differs from the nearest team to the true ball (what control would see).
The cut-off itself (features.BALL_GAP_MAX_S) comes from football reasoning, not from this table.

Run:  .venv/bin/python experiments/pietro/ball_v2_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, external, features  # noqa: E402

GAPS_S = (0.2, 0.5, 1.0, 1.5, 2.0, 3.0)   # source: report only, cut-offs to compare around the 1 s chosen by football reasoning

rows = {g: {"hidden": 0, "filled": 0, "err": [], "flip": 0} for g in GAPS_S}
for game_id in external.games():
    game = external.read_game(game_id)
    n10 = len(game["tracked"])
    src = np.arange(int((n10 - 1) * config.FPS / external.SC_FPS) + 1) * external.SC_FPS / config.FPS
    true_ball = external._resample(game["ball"], src)
    frames, _, truth = external.load(game_id)
    hidden = np.isnan(frames["ball"][:, 0]) & ~np.isnan(true_ball[:, 0]) & (truth != "X")
    players, team = frames["players"], frames["team"]

    def nearest_team(ball):
        d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
        dh = features._nanmin_or_nan(d[:, team == 0], axis=1)
        da = features._nanmin_or_nan(d[:, team == 1], axis=1)
        return np.where(dh < da, 0, 1)

    true_team = nearest_team(true_ball)
    for g in GAPS_S:
        out, filled = features.fill_ball_gaps(frames["ball"], frames["half"],
                                              int(round(g * config.FPS)),
                                              features.BALL_MAX_SPEED_MS / config.FPS)
        m = filled & hidden
        r = rows[g]
        r["hidden"] += int(hidden.sum())
        r["filled"] += int(m.sum())
        r["err"].append(np.linalg.norm(out[m] - true_ball[m], axis=1))
        r["flip"] += int((nearest_team(out)[m] != true_team[m]).sum())

print("labels v{}, {} games".format(external.LABELS_VERSION, len(external.games())))
print("max_gap_s  filled_share  err_median_m  err_p90_m  nearest_team_flip")
for g in GAPS_S:
    r = rows[g]
    err = np.concatenate(r["err"])
    if not len(err):
        print("{:>9}  {:>12.3f}  (nothing filled: hidden bursts are longer)".format(g, 0.0))
        continue
    print("{:>9}  {:>12.3f}  {:>12.2f}  {:>9.2f}  {:>17.3f}".format(
        g, r["filled"] / r["hidden"], np.median(err), np.percentile(err, 90),
        r["flip"] / max(r["filled"], 1)))

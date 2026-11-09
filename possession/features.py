"""[Features] Per-frame features computed from the frame dict.

compute(frames) -> dict of per-frame arrays, every one of length N_FRAMES.

Other roles READ these features; only the features owner changes this file.
ADD-ONLY CONTRACT: a feature, once merged, is never renamed, removed or changed in meaning. A better
version gets a NEW name (e.g. ball_v2). tests/feature_contract.txt lists the frozen names and the
tests fail if one disappears. Add the new name there in the same PR, and post it on the board.

    ball_visible     (N,) bool     the tracker sees the ball
    dist             (N, 46)       metres from each track slot to the ball (NaN if either missing)
    nearest_slot     (N,) int      slot closest to the ball, -1 if none
    nearest_team     (N,) int8     0 = H, 1 = A, -1 if no ball or no players
    nearest_dist     (N,) float    distance of that player to the ball, metres
    nearest_dist_H   (N,) float    closest home player to the ball, metres
    nearest_dist_A   (N,) float    closest away player to the ball, metres
    n_players        (N,) int      tracked objects on the frame

Ideas that belong here: interpolating short ball gaps, ball acceleration and direction changes,
ball inside/outside the pitch, cleaning duplicate tracks.
"""

import numpy as np

NOBODY = -1

# --- parameters (owner: features). Every number carries a source; durations in seconds. --------


def _nanmin_or_nan(values, axis):
    out = np.full(values.shape[0], np.nan)
    has = ~np.all(np.isnan(values), axis=axis)
    out[has] = np.nanmin(values[has], axis=axis)
    return out


def compute(frames):
    ball, players, team = frames["ball"], frames["players"], frames["team"]
    n = len(ball)

    ball_visible = ~np.isnan(ball[:, 0])
    # Squared distance computed exactly like starter/baseline.py, so v0 reproduces it bit-for-bit.
    dx = players[:, :, 0] - ball[:, None, 0]
    dy = players[:, :, 1] - ball[:, None, 1]
    sq = dx * dx + dy * dy                          # NaN where player or ball missing

    has_any = ~np.all(np.isnan(sq), axis=1)
    nearest_slot = np.full(n, NOBODY, dtype=np.int64)
    nearest_slot[has_any] = np.nanargmin(sq[has_any], axis=1)   # first minimum, like baseline
    nearest_team = np.full(n, NOBODY, dtype=np.int8)
    nearest_team[has_any] = team[nearest_slot[has_any]]

    dist = np.sqrt(sq)
    nearest_dist = np.full(n, np.nan)
    nearest_dist[has_any] = dist[has_any, nearest_slot[has_any]]

    return {
        "ball_visible": ball_visible,
        "dist": dist,
        "nearest_slot": nearest_slot,
        "nearest_team": nearest_team,
        "nearest_dist": nearest_dist,
        "nearest_dist_H": _nanmin_or_nan(dist[:, team == 0], axis=1),
        "nearest_dist_A": _nanmin_or_nan(dist[:, team == 1], axis=1),
        "n_players": np.sum(~np.isnan(players[:, :, 0]), axis=1),
    }

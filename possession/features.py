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

  added in #12 (ball_v2: short ball gaps filled):
    ball_v2          (N, 2) float  ball position in metres; like `ball`, but a gap of at most
                                   BALL_GAP_MAX_S between two seen positions in the same half is
                                   filled by a straight line between them. NaN where still unknown.
    ball_v2_filled   (N,) bool     ball_v2 is an interpolated (not seen) position on this frame
    nearest_dist_H_v2 (N,) float   like nearest_dist_H, measured to ball_v2 (metres, NaN if unknown)
    nearest_dist_A_v2 (N,) float   like nearest_dist_A, measured to ball_v2 (metres, NaN if unknown)

Ideas that belong here: ball acceleration and direction changes,
ball inside/outside the pitch, cleaning duplicate tracks.
"""

import numpy as np

from . import config  # general facts only: FPS

NOBODY = -1

# --- parameters (owner: features). Every number carries a source; durations in seconds. --------
BALL_GAP_MAX_S = 1.0           # source: football reasoning: in one second the ball makes at most one straight flight (a 10-20 m pass); the other team needs a reception plus a touch (~0.5 s) to control it, so a straight line between the two seen ends is a fair guess and cannot hide a full exchange
BALL_MAX_SPEED_MS = 35.0       # source: football reasoning: the hardest shots reach ~35 m/s; a gap whose two ends imply a faster straight flight is a tracker jump, so it stays unfilled


def _nanmin_or_nan(values, axis):
    out = np.full(values.shape[0], np.nan)
    has = ~np.all(np.isnan(values), axis=axis)
    out[has] = np.nanmin(values[has], axis=axis)
    return out


def fill_ball_gaps(ball, half, max_frames, max_step_m):
    """Fill runs of missing ball (NaN) of at most max_frames frames by linear interpolation
    between the last and next seen positions, when both exist, lie in the same half and are at
    most max_step_m apart per frame. Returns (filled positions, filled mask). Vectorised."""
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.minimum.accumulate(np.where(seen, idx, n)[::-1])[::-1]
    ok = ~seen & (prev >= 0) & (nxt < n) & (nxt - prev - 1 <= max_frames)
    p, q = prev[ok], nxt[ok]
    if half is not None:
        same = half[p] == half[q]
        ok[np.flatnonzero(ok)[~same]] = False
        p, q = prev[ok], nxt[ok]
    step = np.linalg.norm(ball[q] - ball[p], axis=1) / (q - p)
    fast = step > max_step_m
    ok[np.flatnonzero(ok)[fast]] = False
    p, q = prev[ok], nxt[ok]
    w = ((idx[ok] - p) / (q - p))[:, None]
    out = ball.astype(float, copy=True)
    out[ok] = ball[p] * (1 - w) + ball[q] * w
    return out, ok


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

    ball_v2, ball_v2_filled = fill_ball_gaps(
        ball, frames.get("half"), int(round(BALL_GAP_MAX_S * config.FPS)),
        BALL_MAX_SPEED_MS / config.FPS)
    dist_v2 = np.hypot(players[:, :, 0] - ball_v2[:, None, 0],
                       players[:, :, 1] - ball_v2[:, None, 1])

    return {
        "ball_visible": ball_visible,
        "dist": dist,
        "nearest_slot": nearest_slot,
        "nearest_team": nearest_team,
        "nearest_dist": nearest_dist,
        "nearest_dist_H": _nanmin_or_nan(dist[:, team == 0], axis=1),
        "nearest_dist_A": _nanmin_or_nan(dist[:, team == 1], axis=1),
        "n_players": np.sum(~np.isnan(players[:, :, 0]), axis=1),
        "ball_v2": ball_v2,
        "ball_v2_filled": ball_v2_filled,
        "nearest_dist_H_v2": _nanmin_or_nan(dist_v2[:, team == 0], axis=1),
        "nearest_dist_A_v2": _nanmin_or_nan(dist_v2[:, team == 1], axis=1),
    }

"""[Control / possession changes] Per-frame evidence of which team controls the ball.

detect(frames, feats, tags) -> (N,) int8 evidence: 0 = H, 1 = A, -1 = no evidence.

This is RAW evidence, allowed to be noisy: postprocess turns it into stable possessions.

The real problem (CHALLENGE.md): possession only changes when the other team has the ball UNDER
CONTROL. These are NOT changes: a single touch, a deflected pass, an aerial duel won without
control. A loose ball stays with the team that last had it.

v1 (issue #9), three steps:
  1. Contact: the ball is visible, the nearest player is within CONTROL_RADIUS_M of it and is
     clearly closer than the nearest opponent (by CLEAR_MARGIN_M). Otherwise no contact.
  2. Control: a contact counts only if the same team keeps it for CONFIRM_S without a break.
     A touch, deflection or duel is shorter, so it gives no evidence for the new team.
  3. Loose ball: after the last confirmed control, the evidence stays with that team for up to
     HOLD_S (pass in flight, bounce, ball briefly lost by the tracker). After that: -1.

v2 (Damiano's football idea): where v1 still has no evidence (ball unseen, no clear contact), the
team that covers more of the pitch is taken as the one in possession. A team with the ball spreads
out in length and width to give passing options; the team without it compacts. Measured as the
area between the 10th and 90th percentile of its players' x and y, smoothed over TEAM_AREA_SMOOTH_S;
it decides only when one team's area is clearly larger (log ratio > TEAM_AREA_MIN_LOG_RATIO).
"""

import warnings

import numpy as np

from . import config
from .features import NOBODY

# --- parameters (owner: control). Every number carries a source; durations in seconds. ---------
METHOD = "v2"            # source: issue #9 ("baseline" = v0 nearest team, kept so tests/test_baseline_match.py can prove frame alignment with the official baseline)
CONTROL_RADIUS_M = 2.5   # source: football reasoning: a player controlling the ball has it within one stride (~2 m) plus ~0.5 m tracking error; external v1: flat between 2.5 and 3 m
CLEAR_MARGIN_M = 0.5     # source: football reasoning: the controller must be clearly nearer than the nearest opponent, half a metre is above tracking noise (a duel is closer than that); external v1: flat between 0.25 and 1 m
CONFIRM_S = 0.5          # source: football reasoning: a touch or deflection keeps the ball near a player for a few frames, a reception plus first touch takes about half a second; external v1: flat between 0.4 and 0.6 s
HOLD_S = 5.0             # source: football reasoning: a pass in flight or a loose ball rarely lasts more than a few seconds before someone controls it; external v1: flat between 5 and 8 s
TEAM_AREA_MIN_LOG_RATIO = 0.2   # source: football reasoning: a team in possession is clearly more spread (about 20% more area); a smaller difference is a team in transition; external v1: flat between 0 and 0.2
TEAM_AREA_SMOOTH_S = 1.0        # source: football reasoning: a team's shape changes over seconds, not frames, so one second removes tracking jitter without lagging a turnover; external v1: flat between 0 and 3 s
TEAM_AREA_LOW_Q, TEAM_AREA_HIGH_Q = 10, 90   # source: football reasoning: drop the keeper and the furthest outlier so the area is the outfield block, not one player


def _contact(feats):
    """Step 1: team in clear, close contact with the visible ball, NOBODY otherwise."""
    d_home, d_away = feats["nearest_dist_H"], feats["nearest_dist_A"]
    nearest = np.fmin(d_home, d_away)
    gap = np.abs(d_home - d_away)            # NaN if a team is missing: no clear winner then
    close = (nearest < CONTROL_RADIUS_M) & (gap > CLEAR_MARGIN_M) & feats["ball_visible"]
    team = np.where(d_home < d_away, 0, 1)     # 0 = H, 1 = A (players only, never a referee)
    return np.where(close, team, NOBODY).astype(np.int8)


def _confirmed(contact, min_frames):
    """Step 2: keep only runs of the same team at least min_frames long."""
    n = len(contact)
    if n == 0:
        return contact.copy()
    starts = np.flatnonzero(np.r_[True, contact[1:] != contact[:-1]])
    lengths = np.diff(np.r_[starts, n])
    keep = (contact[starts] != NOBODY) & (lengths >= min_frames)
    return np.where(np.repeat(keep, lengths), contact, NOBODY).astype(np.int8)


def _hold(evidence, max_frames):
    """Step 3: carry the last confirmed team forward for at most max_frames."""
    idx = np.arange(len(evidence))
    last = np.maximum.accumulate(np.where(evidence != NOBODY, idx, -1))
    held = np.where(last >= 0, evidence[np.maximum(last, 0)], NOBODY).astype(np.int8)
    held[(last < 0) | (idx - last > max_frames)] = NOBODY
    return held


def _team_area_side(frames):
    """v2: team that spreads over more pitch (0 = H, 1 = A), NOBODY when unclear or off-play."""
    players, team = frames["players"], frames["team"]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # frames with no player of a team: NaN, handled below
        areas = []
        for side in (0, 1):
            x, y = players[:, team == side, 0], players[:, team == side, 1]
            if x.shape[1] == 0:
                return np.full(len(players), NOBODY, dtype=np.int8)
            lo_x, hi_x = np.nanpercentile(x, [TEAM_AREA_LOW_Q, TEAM_AREA_HIGH_Q], axis=1)
            lo_y, hi_y = np.nanpercentile(y, [TEAM_AREA_LOW_Q, TEAM_AREA_HIGH_Q], axis=1)
            areas.append((hi_x - lo_x) * (hi_y - lo_y))
        ratio = np.log(areas[0] / areas[1])
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    window = max(int(round(TEAM_AREA_SMOOTH_S * config.FPS)), 1)
    ratio = np.convolve(ratio, np.ones(window) / window, mode="same")
    side = np.where(ratio > TEAM_AREA_MIN_LOG_RATIO, 0,
                    np.where(ratio < -TEAM_AREA_MIN_LOG_RATIO, 1, NOBODY)).astype(np.int8)
    if "half" in frames:
        side[np.asarray(frames["half"]) == 0] = NOBODY
    return side


def _baseline(feats):
    """v0, the official baseline rule: nearest team to the visible ball."""
    evidence = feats["nearest_team"].copy()
    evidence[~feats["ball_visible"]] = NOBODY
    return evidence


def detect(frames, feats, tags):
    if METHOD == "baseline":
        return _baseline(feats)
    confirm = max(int(round(CONFIRM_S * config.FPS)), 1)
    hold = int(round(HOLD_S * config.FPS))
    evidence = _hold(_confirmed(_contact(feats), confirm), hold)
    if METHOD == "v2":
        gap = evidence == NOBODY
        evidence[gap] = _team_area_side(frames)[gap]
    return evidence

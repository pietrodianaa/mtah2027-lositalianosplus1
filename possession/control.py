"""[Control / possession changes] Per-frame evidence of which team controls the ball.

detect(frames, feats, tags) -> (N,) int8 evidence: 0 = H, 1 = A, -1 = no evidence.

This is RAW evidence, allowed to be noisy: postprocess turns it into stable possessions. v0 is the
baseline rule: whoever is nearest to the visible ball, no evidence when the ball is not visible.

The real problem (CHALLENGE.md): possession only changes when the other team has the ball UNDER
CONTROL. These are NOT changes: a single touch, a deflected pass, an aerial duel won without
control. A loose ball stays with the team that last had it. Ideas:
  - require the new team to be nearest for K consecutive frames, or within R metres
  - ball slowing down / changing direction near a player = reception or touch
  - ball moving fast between players = pass in flight, keep the previous owner
  - compare home vs away nearest distance (feats nearest_dist_H / nearest_dist_A)
  - output -1 when unsure: it is cheaper than a wrong guess (see postprocess)
"""

from .features import NOBODY

# --- parameters (owner: control). Every number carries a source; durations in seconds. ---------


def detect(frames, feats, tags):
    evidence = feats["nearest_team"].copy()
    evidence[~feats["ball_visible"]] = NOBODY
    return evidence

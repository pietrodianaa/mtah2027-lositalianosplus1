"""[Dead ball] Decide on which frames the ball is out of play.

detect(frames, feats, tags) -> (N,) bool, True = dead ball (label D).

v0 returns all False: the baseline has no dead-ball notion beyond "the tracker lost the ball",
which postprocess handles. That is why its D class is weak, and why this module is an opening.

Signals worth trying (each from football reasoning; check with `python run.py --tune --role dead_ball`):
  - ball outside the pitch lines (x < 0, x > 105.3, y < 0, y > 68), with a margin
  - ball still (speed ~0) for a while, often with players also slowing down
  - Smart Tagging 'SET PIECES' intervals: every set piece is a restart, so a dead stretch ends
    near its start. Type tag: Throw-in, Corner Kick, Goal Kick, Free Kick, Kick Off.
  - players regrouping (kick-off shape after goals, wall at free kicks)
  - the ball disappearing near a touchline / byline
Remember: the ground truth says D only while the ball is dead. Restart ownership is NOT asked for.
"""

import numpy as np

from . import config  # noqa: F401  (general facts only: FPS, PITCH)

# --- parameters (owner: dead ball). Every number carries a source; durations in seconds. -------
ENABLED = False                # source: official baseline default (no dead-ball detection)


def detect(frames, feats, tags):
    dead = np.zeros(len(frames["ball"]), dtype=bool)
    if not ENABLED:
        return dead
    # implement here.
    return dead

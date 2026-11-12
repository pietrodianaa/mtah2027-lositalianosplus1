"""[Dead ball] Decide on which frames the ball is out of play.

detect(frames, feats, tags) -> (N,) bool, True = dead ball (label D).

v1 (#10, tuned in v1.1): ball out over a touchline or byline.
  1. A stretch where the visible ball is beyond a line is dead if, somewhere in it, the ball is
     clearly out (more than MARGIN_M past the line). The whole stretch counts, so the dead run
     starts when the ball crosses the line, not when it passes the margin.
  2. After it goes out the tracker usually loses the ball (stands, ball boy, off camera): stay
     dead while the ball is missing, up to MAX_FILL_S, never across halves. The run ends as soon
     as the ball is seen again inside the pitch.
  3. Dead runs shorter than MIN_DEAD_S are tracking glitches, not stoppages: dropped.
Works with tags == [] (tags are not used yet).

Signals still to try (each from football reasoning; check with `python run.py --tune --role dead_ball`):
  - ball still (speed ~0) for a while, often with players also slowing down; never the keeper
    holding the ball (ball stays with him), but yes a free kick in the box (taker steps away)
  - Smart Tagging 'SET PIECES' intervals: every set piece is a restart, so a dead stretch ends
    near its start. Type tag: Throw-in, Corner Kick, Goal Kick, Free Kick, Kick Off.
  - players regrouping (kick-off shape after goals, wall at free kicks)
Remember: the ground truth says D only while the ball is dead. Restart ownership is NOT asked for.
"""

import numpy as np

from . import config

# --- parameters (owner: dead ball). Every number carries a source; durations in seconds. -------
ENABLED = True                 # source: football reasoning: v1 out-of-bounds rule (#10)
MARGIN_M = 0.25                # source: external v1: sweep flat from 0 to 0.25 m (F1 0.503-0.486, P 0.926), 1 m 0.414; football reasoning: whole ball past the line = centre >= 0.11 m past it, 0.25 m keeps a buffer for tracking noise
MAX_FILL_S = 40.0              # source: external v1: sweep 0 s F1 0.09, 10 s 0.46, flat from 20 to 90 s; football reasoning: throw-in ~15 s, goal kick ~29 s, corner ~34 s (Opta)
MIN_DEAD_S = 2.0               # source: football reasoning: a real stoppage lasts seconds, shorter runs are glitches; external v1: 1-3 s within 0.01


def detect(frames, feats, tags):
    n = len(frames["ball"])
    dead = np.zeros(n, dtype=bool)
    if not ENABLED:
        return dead

    length, width = frames.get("pitch", config.PITCH)       # synthetic test frames have no pitch
    x = frames["ball"][:, 0]
    y = frames["ball"][:, 1]
    half = frames["half"]
    visible = ~np.isnan(x)
    with np.errstate(invalid="ignore"):
        past_line = visible & ((x < 0) | (x > length) | (y < 0) | (y > width))
        past_margin = visible & ((x < -MARGIN_M) | (x > length + MARGIN_M)
                                 | (y < -MARGIN_M) | (y > width + MARGIN_M))
    max_fill = int(MAX_FILL_S * config.FPS)
    min_dead = int(MIN_DEAD_S * config.FPS)

    # 1 + 2: stretches past a line that are clearly out (or continue a dead run), then the gap after
    i = 0
    while i < n:
        if not past_line[i]:
            i += 1
            continue
        j = i
        while j < n and past_line[j]:
            j += 1
        if past_margin[i:j].any() or (i > 0 and dead[i - 1]):
            dead[i:j] = True
            k = j
            while k < n and k - j < max_fill and not visible[k] and half[k] == half[j - 1]:
                dead[k] = True
                k += 1
        i = j

    # 3: drop dead runs too short to be a stoppage
    i = 0
    while i < n:
        if not dead[i]:
            i += 1
            continue
        j = i
        while j < n and dead[j]:
            j += 1
        if j - i < min_dead:
            dead[i:j] = False
        i = j
    return dead

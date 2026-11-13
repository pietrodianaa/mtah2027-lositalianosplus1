"""[Dead ball] Decide on which frames the ball is out of play.

detect(frames, feats, tags) -> (N,) bool, True = dead ball (label D).

v1.2 (#10): ball out over a touchline or byline, dead until the restart.
  1. A stretch where the visible ball is beyond a line is dead if, somewhere in it, the ball is
     clearly out (more than MARGIN_M past the line). The whole stretch counts, so the dead run
     starts when the ball crosses the line, not when it passes the margin.
  2. After it goes out the tracker usually loses the ball (stands, ball boy, off camera): stay
     dead up to MAX_FILL_S, never across halves. With HOLD_UNTIL_RESTART the run ends at the
     restart: the ball moving fast again (throw-in, kick) or seen more than RESTART_ZONE_M inside
     every line; a ball seen again near a line but still (placed for the restart) stays dead.
     Without it (v1.1) the run ends as soon as the ball is seen again.
Rule 2 (#10): ball still after a whistle (fouls, offsides). The ball barely moves for
STILL_WINDOW_S, nobody is within STILL_CLEAR_M of it (the taker steps away for the run-up; this
excludes a keeper holding the ball or a player with it at his feet) and players walk on average:
dead if that lasts at least STILL_MIN_S (backdated by the window). Short tracker gaps where the
ball reappears on the same spot are bridged. Ends when the ball moves or a player reaches it.
Rule 2b (#10): everyone walks. The players' average speed (smoothed over WALK_SMOOTH_S) stays below
WALK_SPEED_MS for at least WALK_MIN_S and the ball is passed on at most WALK_BALL_FAST_SHARE of the
seen frames: dead. Catches fouls where the ball is carried to the spot or rolls away; the run ends
when the players speed up again, so a pass back to the taker does not end it.
v2.2 (#10, Edoardo's idea): a foul before the walk. An H and an A player both at the ball and one of
them, running, stops right after: if everyone then walks (shorter walk than rule 2b), dead until the
ball is kicked from near the foul spot (Law 13: the free kick is taken where the foul happened).
  3. Dead runs shorter than MIN_DEAD_S are tracking glitches, not stoppages: dropped (all rules).
Works with tags == [] (tags are not used yet).

Signals still to try (each from football reasoning; check with `python run.py --tune --role dead_ball`):
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
HOLD_UNTIL_RESTART = True      # source: football reasoning: after an out play stays dead until the restart, even with the ball seen again inside (ball boy, ball placed); external v1: F1 0.486 -> 0.545
SPEED_WINDOW_S = 0.2           # source: football reasoning: a thrown or kicked ball covers > 1 m in 0.2 s, well above ~0.5 m tracking error
RESTART_SPEED_MS = 4.0         # source: external v1: sweep 4/6/9 m/s, F1 flat (0.545-0.558), precision 0.916/0.899/0.873, so 4 (fewest false D); football reasoning: throws and kicks start faster than a ball carried or placed
RESTART_ZONE_M = 6.0           # source: football reasoning: every restart after an out is near a line (touchline, corner arc, goal area 5.5 m deep); external v1: 6/10/20 m flat, 6 m best precision
# rule 2, ball still (fouls, offsides): ball placed, nobody on it, players walking
STILL_ENABLED = True           # source: football reasoning: after a whistle the ball is placed and play waits; external v1: F1 0.545 -> 0.590 with bridge + backdate, precision up, far isolated D runs unchanged (9)
STILL_WINDOW_S = 1.0           # source: football reasoning: 1 s tells a placed ball from tracking jitter (~0.5 m); external v1: 0.5 s no better, more isolated runs with a 2 s bridge
STILL_SPEED_MS = 1.0           # source: football reasoning: a placed ball moves < 1 m in 1 s; external v1: 0.5 vs 1.0 within 0.004, 1.5 adds +0.003 but one more far isolated run
STILL_CLEAR_M = 1.5            # source: football reasoning: taker steps away; keeper holding the ball within ~1 m (2024-25: keepers held it long); external v1: 1.0/1.5/3.0 m F1 0.559/0.557/0.551, 1.5 for keeper safety
STILL_PLAYERS_MS = 2.5         # source: football reasoning: players walk while a free kick is set up; external v1: no effect (2.0 to off identical), kept as a safeguard
STILL_MIN_S = 3.0              # source: football reasoning: setting up even a quick free kick takes a few seconds; external v1: 3 vs 5 s within 0.01
STILL_BRIDGE_S = 2.0           # source: football reasoning: a ball unseen for a moment that reappears on the same spot has not moved (tracker loses it ~41% of frames); external v1: 0/1/2 s F1 0.560/0.582/0.590
STILL_BRIDGE_M = 1.0           # source: football reasoning: same spot = within 1 m, about twice the tracking error
STILL_BACKDATE = True          # source: football reasoning: a ball still now vs STILL_WINDOW_S ago was already still during that window; external v1: +0.003 to +0.006, no extra isolated runs
# rule 2b, everyone walks (fouls where the ball is carried or rolls away): players slow, ball rarely passed
WALK_ENABLED = True            # source: football reasoning: after a whistle every player walks while the free kick is set up; external v1: F1 0.590 -> 0.663, gain on even and odd games
WALK_SMOOTH_S = 1.0            # source: football reasoning: average the players' speed over 1 s so a single stride or tracking jitter does not count
WALK_SPEED_MS = 1.0            # source: football reasoning: slow walking ~1 m/s; external v1: 1.5 or 2 m/s flood open play (isolated D runs up to 349), 1.0 keeps precision ~0.93
WALK_MIN_S = 8.0               # source: football reasoning: setting up a free kick takes several seconds; external v1: 4/6/8 s isolated runs 56/34/29 (main 27), 8 s for transfer to full-pitch tracking
WALK_BALL_FAST_MS = 4.0        # source: football reasoning: a pass moves faster than a ball carried by hand or on foot (same scale as RESTART_SPEED_MS)
WALK_BALL_FAST_SHARE = 0.15    # source: football reasoning: during a stoppage the ball is passed at most once or twice; external v1: 0.05/0.15 same isolated runs, 0.30 more
# v2.2, a foul (Edoardo): an H and an A player at the ball, one of them stops, then everyone walks;
# the free kick is taken where the foul happened (Law 13), so only a kick from that spot ends the stoppage
FOUL_ENABLED = True            # source: football reasoning (Edoardo): a foul is two opponents at the ball and one of them stopping right away
FOUL_DUEL_M = 1.5              # source: football reasoning: in a duel both players are within ~1.5 m of the ball (tracking error ~0.5 m)
FOUL_MOVING_MS = 2.0           # source: football reasoning: the player was running into the duel (faster than a jog)
FOUL_STOP_MS = 1.0             # source: football reasoning: after a foul one of the two stops or falls (slower than walking)
FOUL_STOP_WITHIN_S = 2.0       # source: football reasoning: a fouled player stops or falls within a second or two, and before the others start walking
FOUL_LOOKBACK_S = 4.0          # source: football reasoning: whistle ~1 s after the foul, players slow down 1-2 s later
FOUL_WALK_SPEED_MS = 1.0       # source: football reasoning: same walking pace as rule 2b
FOUL_WALK_MIN_S = 4.0          # source: football reasoning: after a confirmed foul a shorter walk is enough (rule 2b alone: 8 s)
FOUL_PASS_SHARE = 0.3          # source: football reasoning (Edoardo): after a foul the ball may roll away and be passed back to the taker, more ball movement than rule 2b allows
FOUL_PLACED_MS = 1.5           # source: football reasoning: the free kick is taken with a stationary ball (Law 13); 1.5 m/s over 0.2 s tolerates tracking jitter
FOUL_SPOT_M = None             # source: football reasoning: free kick from the foul spot (Law 13), in practice with some tolerance; None = end when players speed up
FOUL_SPOT_GRACE_S = 5.0        # source: football reasoning: players start moving into position a few seconds before the kick


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

    # restart = the ball moves fast again, or is seen far from every line (own speed from
    # positions, so the external and the real tracker are treated the same way)
    w = int(SPEED_WINDOW_S * config.FPS)
    speed = np.full(n, np.nan)
    speed[w:] = np.linalg.norm(frames["ball"][w:] - frames["ball"][:-w], axis=1) / SPEED_WINDOW_S
    with np.errstate(invalid="ignore"):
        to_line = np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))
        restarted = visible & ((speed > RESTART_SPEED_MS) | (to_line > RESTART_ZONE_M))

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
            while k < n and k - j < max_fill and half[k] == half[j - 1]:
                if visible[k] and (restarted[k] or not HOLD_UNTIL_RESTART):
                    break
                dead[k] = True
                k += 1
        i = j

    # rule 2: ball still with nobody on it and players walking, for at least STILL_MIN_S
    if STILL_ENABLED:
        ws = int(STILL_WINDOW_S * config.FPS)
        with np.errstate(invalid="ignore"):
            clear = feats["nearest_dist"] > STILL_CLEAR_M
        # bridge short gaps: ball unseen for <= STILL_BRIDGE_S, back within STILL_BRIDGE_M of where
        # it vanished = it did not move; hold its position and keep the nearest-player check as before
        ball = frames["ball"].copy()
        max_gap = int(STILL_BRIDGE_S * config.FPS)
        i = 1
        while i < n:
            if visible[i]:
                i += 1
                continue
            j = i
            while j < n and not visible[j]:
                j += 1
            if j < n and visible[i - 1] and j - i <= max_gap \
                    and np.linalg.norm(ball[j] - ball[i - 1]) < STILL_BRIDGE_M:
                ball[i:j] = ball[i - 1]
                clear[i:j] = clear[i - 1]
            i = j
        drift = np.full(n, np.nan)
        drift[ws:] = np.linalg.norm(ball[ws:] - ball[:-ws], axis=1) / STILL_WINDOW_S
        player_speed = frames["player_speed"]
        tracked = ~np.isnan(player_speed)
        mean_speed = np.where(tracked, player_speed, 0.0).sum(axis=1) / np.maximum(tracked.sum(axis=1), 1)
        with np.errstate(invalid="ignore"):
            still = (drift < STILL_SPEED_MS) & clear & (mean_speed < STILL_PLAYERS_MS)
        min_still = int(STILL_MIN_S * config.FPS)
        i = 0
        while i < n:
            if not still[i]:
                i += 1
                continue
            j = i
            while j < n and still[j]:
                j += 1
            if j - i >= min_still:
                dead[max(i - ws, 0) if STILL_BACKDATE else i:j] = True
            i = j

    # rule 2b: everyone walks for at least WALK_MIN_S and the ball is rarely passed; the run ends
    # when the players speed up again (the restart), so a pass back to the taker does not end it
    if WALK_ENABLED:
        player_speed = frames["player_speed"]
        tracked = ~np.isnan(player_speed)
        n_tracked = tracked.sum(axis=1)
        mean_speed = np.where(tracked, player_speed, 0.0).sum(axis=1) / np.maximum(n_tracked, 1)
        k = max(int(WALK_SMOOTH_S * config.FPS), 1)
        smooth = np.convolve(mean_speed, np.ones(k) / k, mode="same")
        walking = (smooth < WALK_SPEED_MS) & (n_tracked > 0)
        with np.errstate(invalid="ignore"):
            passing = speed > WALK_BALL_FAST_MS
        min_walk = int(WALK_MIN_S * config.FPS)
        i = 0
        while i < n:
            if not walking[i]:
                i += 1
                continue
            j = i
            while j < n and walking[j]:
                j += 1
            seen = visible[i:j]
            share = passing[i:j][seen].mean() if seen.any() else 0.0
            if j - i >= min_walk and share <= WALK_BALL_FAST_SHARE:
                dead[i:j] = True
            i = j

    # v2.2: a foul before the walk. Duel = nearest H and nearest A player both within FOUL_DUEL_M of the
    # ball; foul = one of the two was running and stops within FOUL_STOP_WITHIN_S. If everyone then
    # walks (FOUL_WALK_SPEED_MS for FOUL_WALK_MIN_S, starting at most FOUL_LOOKBACK_S after the foul),
    # dead from the walk start until the ball is kicked from within FOUL_SPOT_M of the foul spot.
    if WALK_ENABLED and FOUL_ENABLED:
        players = frames["players"]
        home = frames["team"] == 0
        away = frames["team"] == 1
        dist = np.linalg.norm(players - frames["ball"][:, None, :], axis=2)
        dist = np.where(np.isnan(dist), np.inf, dist)
        slot_h = np.flatnonzero(home)[dist[:, home].argmin(axis=1)]
        slot_a = np.flatnonzero(away)[dist[:, away].argmin(axis=1)]
        duel = (dist[:, home].min(axis=1) < FOUL_DUEL_M) & (dist[:, away].min(axis=1) < FOUL_DUEL_M)
        # per-player speed averaged over 0.4 s (an untracked player can be neither moving nor stopped)
        ps = frames["player_speed"]
        ok = ~np.isnan(ps)
        q = max(int(0.4 * config.FPS), 1)
        csum = np.cumsum(np.vstack((np.zeros((1, ps.shape[1])), np.where(ok, ps, 0.0))), axis=0)
        ccnt = np.cumsum(np.vstack((np.zeros((1, ps.shape[1])), ok.astype(float))), axis=0)
        low = np.maximum(np.arange(n) - q + 1, 0)
        cnt = ccnt[np.arange(n) + 1] - ccnt[low]
        pspeed = np.where(cnt > 0, (csum[np.arange(n) + 1] - csum[low]) / np.maximum(cnt, 1), np.nan)
        walking_foul = (smooth < FOUL_WALK_SPEED_MS) & (n_tracked > 0)
        foul_look = int(FOUL_LOOKBACK_S * config.FPS)
        before = int(1.0 * config.FPS)
        within = int(FOUL_STOP_WITHIN_S * config.FPS)
        foul_walk = int(FOUL_WALK_MIN_S * config.FPS)
        grace = int(FOUL_SPOT_GRACE_S * config.FPS)
        i = 0
        while i < n:
            if not walking_foul[i]:
                i += 1
                continue
            j = i
            while j < n and walking_foul[j]:
                j += 1
            seen = visible[i:j]
            share = passing[i:j][seen].mean() if seen.any() else 0.0
            spot = None
            if j - i >= foul_walk and share <= FOUL_PASS_SHARE:
                # latest duel before the walk where one of the two was running and then stops
                for t in range(i, max(i - foul_look, before) - 1, -1):
                    if not duel[t]:
                        continue
                    for slot in (slot_h[t], slot_a[t]):
                        was = pspeed[t - before:t + 1, slot]
                        after = pspeed[t:min(t + within, n), slot]
                        others = mean_speed[t:min(t + within, n)]
                        with np.errstate(invalid="ignore"):
                            # he was running and stops while the others are still moving (before the whistle)
                            stops = (after < FOUL_STOP_MS) & (others >= FOUL_WALK_SPEED_MS)
                            if np.nanmax(was, initial=0.0) > FOUL_MOVING_MS and stops.any():
                                spot = frames["ball"][t]
                    if spot is not None:
                        break
            if spot is not None:
                end = j
                if FOUL_SPOT_M is not None:
                    # the restart: a stationary ball (Law 13) starts moving fast from near the foul spot
                    for k in range(i + 2 * w, min(j + grace, n)):
                        if visible[k] and speed[k] > RESTART_SPEED_MS and speed[k - w] < FOUL_PLACED_MS \
                                and np.linalg.norm(frames["ball"][k - w] - spot) < FOUL_SPOT_M:
                            end = k
                            break
                dead[i:end] = True
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

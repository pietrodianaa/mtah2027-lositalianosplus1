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
Rule 1c (#10): outs the tracker does not see. The ball (ball_v2: gaps up to 1 s already filled) is
last seen within UNSEEN_LAST_M of a line, optionally heading to it, then stays unseen for at least
UNSEEN_GAP_S and comes back at a restart spot (within RESTART_ZONE_M of a line, or within
UNSEEN_CENTRE_M of the centre spot for a kick-off), and players walk (mean speed below
UNSEEN_WALK_MS) on at least UNSEEN_WALK_SHARE of the gap: the gap is dead, then held until the
restart as in rule 1.
Rule 4 (#10): kick-off shape (after a goal, at the start of a half). Law 8: at a kick-off every
player is in his own half and the ball is on the centre mark. Both teams split by the halfway line
(at least KICKOFF_SHARE of each team's tracked players on its side, KICKOFF_TOL_M of slack, at
least KICKOFF_MIN_PLAYERS per team tracked), players walking, ball unseen or within
KICKOFF_BALL_M of the centre spot, for at least KICKOFF_MIN_S: dead. Ends at the kick-off.
Rule 2 (#10): ball still after a whistle (fouls, offsides). The ball barely moves for
STILL_WINDOW_S, nobody is within STILL_CLEAR_M of it (the taker steps away for the run-up; this
excludes a keeper holding the ball or a player with it at his feet) and players walk on average:
dead if that lasts at least STILL_MIN_S (backdated by the window). Short tracker gaps where the
ball reappears on the same spot are bridged. Ends when the ball moves or a player reaches it.
Rule 2b (#10): everyone walks. The players' average speed (smoothed over WALK_SMOOTH_S) stays below
WALK_SPEED_MS for at least WALK_MIN_S and the ball is passed on at most WALK_BALL_FAST_SHARE of the
seen frames: dead. Catches fouls where the ball is carried to the spot or rolls away; the run ends
when the players speed up again, so a pass back to the taker does not end it.
Rule 5 (#10): wall at a free kick (Law 13: opponents at least 9.15 m from the ball). At least
WALL_N players of one team standing (< WALL_STAND_MS) side by side (each within WALL_GAP_M of
another), WALL_MIN_M to WALL_MAX_M from the ball, for at least WALL_MIN_S: dead, held until the
ball is kicked (faster than RESTART_SPEED_MS), at most MAX_FILL_S.
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
# rule 1c, outs the tracker does not see: ball lost next to a line, back at a restart spot
UNSEEN_ENABLED = True          # source: football reasoning: after an out the broadcast camera often loses the ball (stands, ball boy) before it is seen past the line; external v1 (diagnose_coverage.py): 34.9% of dead frames are in stops we never touch
UNSEEN_LAST_M = 6.0            # source: football reasoning: a ball lost within a few metres of a line, back at a restart spot after seconds, went out; external v1 (sweep_unseen.py): 1.5/3/6 m spell_proxy 0.845/0.850/0.853 at 3 s
UNSEEN_HEADING = False         # source: football reasoning: a ball going out moves towards the line, but the last seen frames are often a deflection or a bounce; external v1: the heading check lowers spell_proxy at every setting (0.848 vs 0.853 at 6 m, 3 s)
UNSEEN_GAP_S = 3.0             # source: football reasoning: a ball fetched and placed for a throw-in takes several seconds; short losses are tracker noise (ball_v2 fills up to 1 s); external v1: 1.5 s floods isolated D runs (263 at 6 m), 3 vs 5 s spell_proxy 0.853/0.852
UNSEEN_WALK_MS = 2.0           # source: football reasoning: below jogging pace = play has stopped (rule 2b uses 1.0 m/s smoothed); external v1 (sweep_unseen.py round 2): 1.5 vs 2.0 m/s at 6 m, share 0.5: spell_proxy 0.849 vs 0.852
UNSEEN_WALK_SHARE = 0.5        # source: football reasoning: during a real stop players walk most of the time, on a wing in open play they run (wide wingers hug the touchline); external v1 (diagnose_unseen.py): walking share 1.00 on 1c runs at true stops, 0.21 on isolated ones; sweep round 2: 0/0.3/0.5/0.7 at 6 m, 2 m/s spell_proxy 0.853/0.853/0.852/0.851, isolated 132/115/107/103
UNSEEN_CENTRE_M = 3.0          # source: football reasoning: a ball back on the centre spot is a kick-off (after a goal: the ball went over the goal line unseen)
# rule 4, kick-off shape: both teams in their own half, ball on the centre spot, players waiting
KICKOFF_ENABLED = True         # source: football reasoning: Law 8, before a kick-off every player is in his own half and the ball is on the centre mark; external v1 (sweep_kickoff.py, 9 settings): F1 up on even and odd games at every setting, no swallowed possession, spell_proxy never lower
KICKOFF_SHARE = 0.9            # source: football reasoning: all players in their own half, minus one tracking error per ~10 players; external v1: 0.9 vs 1.0 F1 0.7476 vs 0.7467 at 3 m, 4 s
KICKOFF_TOL_M = 1.0            # source: football reasoning: players stand on the halfway line, ~0.5 m tracking error
KICKOFF_MIN_PLAYERS = 6        # source: football reasoning: with fewer players tracked per team a split by the halfway line can happen by chance
KICKOFF_BALL_M = 3.0           # source: football reasoning: ball on the centre mark, ~0.5 m tracking error plus the taker touching it
KICKOFF_WALK_MS = 2.0          # source: football reasoning: players wait for the whistle, below jogging pace (same as UNSEEN_WALK_MS)
KICKOFF_MIN_S = 4.0            # source: football reasoning: teams take several seconds to line up for a kick-off; external v1: 2 vs 4 s F1 0.7485 vs 0.7476, isolated D runs 110 vs 108 (v3: 107), 4 s for precision
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

# rule 5, wall at a free kick: players of one team standing side by side ~9.15 m from the ball
WALL_ENABLED = True            # source: football reasoning: Law 13, at a free kick near goal the defenders build a wall 9.15 m from the ball; open play never shows players standing shoulder to shoulder
WALL_N = 3                     # source: football reasoning: a wall has at least 3 players (2 can be two defenders marking side by side)
WALL_GAP_M = 1.5               # source: football reasoning: players in a wall stand shoulder to shoulder (< 1 m apart), plus ~0.5 m tracking error
WALL_MIN_M = 7.0               # source: football reasoning: 9.15 m (Law 13) minus walls creeping forward and tracking error
WALL_MAX_M = 12.0              # source: football reasoning: 9.15 m plus a wall set a little deeper and tracking error
WALL_STAND_MS = 1.0            # source: football reasoning: players in a wall stand still while the kick is prepared
WALL_MIN_S = 2.0               # source: football reasoning: a wall holds its shape for several seconds before the kick


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

    # rule 1c: ball lost next to a line (heading to it), back at a restart spot after a long gap
    if UNSEEN_ENABLED:
        ball2 = feats["ball_v2"]
        vis2 = ~np.isnan(ball2[:, 0])
        with np.errstate(invalid="ignore"):
            to_line2 = np.minimum(np.minimum(ball2[:, 0], length - ball2[:, 0]),
                                  np.minimum(ball2[:, 1], width - ball2[:, 1]))
            to_centre2 = np.hypot(ball2[:, 0] - length / 2, ball2[:, 1] - width / 2)
        min_gap = int(UNSEEN_GAP_S * config.FPS)
        ps = frames["player_speed"]
        n_ps = (~np.isnan(ps)).sum(axis=1)
        mean_ps = np.where(np.isnan(ps), 0.0, ps).sum(axis=1) / np.maximum(n_ps, 1)
        slow = (mean_ps < UNSEEN_WALK_MS) & (n_ps > 0)
        i = 1
        while i < n:
            if vis2[i]:
                i += 1
                continue
            j = i
            while j < n and not vis2[j]:
                j += 1
            last = i - 1
            ok = (vis2[last] and j < n and j - i >= min_gap and half[last] == half[j]
                  and to_line2[last] <= UNSEEN_LAST_M
                  and (to_line2[j] <= RESTART_ZONE_M or to_centre2[j] <= UNSEEN_CENTRE_M))
            if ok and UNSEEN_WALK_SHARE > 0:
                seen_ps = n_ps[i:j] > 0
                ok = seen_ps.any() and slow[i:j][seen_ps].mean() >= UNSEEN_WALK_SHARE
            if ok and UNSEEN_HEADING:
                before = last - w
                ok = before >= 0 and vis2[before] and to_line2[last] < to_line2[before]
            if ok:
                dead[i:j] = True
                k = j
                while k < n and k - j < max_fill and half[k] == half[j - 1]:
                    if visible[k] and restarted[k]:
                        break
                    dead[k] = True
                    k += 1
            i = j

    # rule 4: kick-off shape, both teams split by the halfway line, ball on the centre spot
    if KICKOFF_ENABLED:
        px = frames["players"][:, :, 0]
        team = np.asarray(frames["team"])
        mid = length / 2
        with np.errstate(invalid="ignore"):
            left = px < mid + KICKOFF_TOL_M
            right = px > mid - KICKOFF_TOL_M
        seen_p = ~np.isnan(px)
        split = np.zeros(n, dtype=bool)
        for home_side, away_side in ((left, right), (right, left)):
            ok_side = np.ones(n, dtype=bool)
            for t, side in ((0, home_side), (1, away_side)):
                cols = team == t
                n_t = seen_p[:, cols].sum(axis=1)
                on_side = (side[:, cols] & seen_p[:, cols]).sum(axis=1)
                ok_side &= (n_t >= KICKOFF_MIN_PLAYERS) & (on_side >= KICKOFF_SHARE * n_t)
            split |= ok_side
        ps = frames["player_speed"]
        n_ps = (~np.isnan(ps)).sum(axis=1)
        mean_ps = np.where(np.isnan(ps), 0.0, ps).sum(axis=1) / np.maximum(n_ps, 1)
        with np.errstate(invalid="ignore"):
            on_spot = np.hypot(x - mid, y - width / 2) <= KICKOFF_BALL_M
        waiting = split & (n_ps > 0) & (mean_ps < KICKOFF_WALK_MS) & (~visible | on_spot)
        min_ko = int(KICKOFF_MIN_S * config.FPS)
        i = 0
        while i < n:
            if not waiting[i]:
                i += 1
                continue
            j = i
            while j < n and waiting[j]:
                j += 1
            if j - i >= min_ko:
                dead[i:j] = True
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

    # rule 5: a wall, players of one team standing side by side ~9.15 m from the ball
    if WALL_ENABLED:
        players = frames["players"]
        team = np.asarray(frames["team"])
        pspeed = frames["player_speed"]
        ball2 = feats["ball_v2"]
        wall = np.zeros(n, dtype=bool)
        with np.errstate(invalid="ignore"):
            dist_b = np.linalg.norm(players - ball2[:, None, :], axis=2)
            cand = (dist_b >= WALL_MIN_M) & (dist_b <= WALL_MAX_M) & (pspeed < WALL_STAND_MS)
        for t in (0, 1):
            cand_t = cand & (team == t)[None, :]
            for f in np.flatnonzero(cand_t.sum(axis=1) >= WALL_N):
                pos = players[f, cand_t[f]]
                gap = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
                np.fill_diagonal(gap, np.inf)
                if (gap.min(axis=1) <= WALL_GAP_M).sum() >= WALL_N:
                    wall[f] = True
        min_wall = int(WALL_MIN_S * config.FPS)
        i = 0
        while i < n:
            if not wall[i]:
                i += 1
                continue
            j = i
            while j < n and wall[j]:
                j += 1
            if j - i >= min_wall:
                dead[i:j] = True
                k = j
                while k < n and k - j < max_fill and half[k] == half[j - 1]:
                    if visible[k] and speed[k] > RESTART_SPEED_MS:
                        break
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

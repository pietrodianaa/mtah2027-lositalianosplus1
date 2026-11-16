"""Touch-filter lab (issue: fake possessions from touches, deflections, blocks, aerial duels).
Prototype control variants WITHOUT editing possession/control.py (Damiano's file), scored on the
external practice set only: control role score (frozen reference smoother, true dead) and the joint
check (live dead ball + live postprocess), from the cache built by touch_lab_cache.py.
Run: PYTHONPATH=. python experiments/pietro/touch_lab.py [variant ...]"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from possession import config, control, external, postprocess, scores
from possession.features import NOBODY

CACHE = os.path.join(external.EXTERNAL_DIR, "touch_lab")
FPS = config.FPS


def frames_of(s):
    return int(round(s * FPS))


def runs(x):
    n = len(x)
    starts = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    return starts, np.diff(np.r_[starts, n])


def ball_speed(ball):
    """m/s from the visible ball, NaN where either frame is hidden; 0.2 s centred difference."""
    k = 2
    v = np.full(len(ball), np.nan)
    d = np.hypot(ball[2 * k:, 0] - ball[:-2 * k, 0], ball[2 * k:, 1] - ball[:-2 * k, 1])
    v[k:-k] = d * FPS / (2 * k)
    return v


# ---------------------------------------------------------------- control variants (lab copies)
def contact(c, radius=control.CONTROL_RADIUS_M, margin=control.CLEAR_MARGIN_M, v2=False):
    dh = c["nearest_dist_H_v2" if v2 else "nearest_dist_H"]
    da = c["nearest_dist_A_v2" if v2 else "nearest_dist_A"]
    vis = (~np.isnan(c["ball_v2"][:, 0])) if v2 else c["ball_visible"]
    nearest, gap = np.fmin(dh, da), np.abs(dh - da)
    close = (nearest < radius) & (gap > margin) & vis
    return np.where(close, np.where(dh < da, 0, 1), NOBODY).astype(np.int8)


def confirmed(ct, confirm_s=control.CONFIRM_S, switch_s=None, slow_ms=None, speed=None):
    """v1 confirm, plus two touch filters:
    switch_s: a contact run by the team OTHER than the last confirmed controller needs switch_s
              (a take-over must last longer than a touch; same-team re-contacts keep confirm_s).
    slow_ms:  a take-over run must contain a frame with ball speed under slow_ms (the receiver
              slows the ball down; a deflection or block leaves it fast)."""
    starts, lengths = runs(ct)
    keep = np.zeros(len(starts), dtype=bool)
    owner = NOBODY
    base = max(frames_of(confirm_s), 1)
    for k, (s, n) in enumerate(zip(starts, lengths)):
        t = ct[s]
        if t == NOBODY:
            continue
        challenger = owner != NOBODY and t != owner
        need = max(frames_of(switch_s), 1) if (challenger and switch_s) else base
        ok = n >= need
        if ok and challenger and slow_ms is not None:
            sp = speed[s:s + n]
            ok = bool(np.any(sp[~np.isnan(sp)] < slow_ms)) if np.any(~np.isnan(sp)) else True
        if ok:
            keep[k] = True
            owner = t
    return np.where(np.repeat(keep, lengths), ct, NOBODY).astype(np.int8)


def area_fill(ev, c, mode="v2", area_switch_s=None):
    """v2 fills every no-evidence frame with the team-area side.
    mode 'agree': the area may only fill with the last confirmed controller's team (never switches
                  on shape alone); after a long blind spell (> area_switch_s) it may switch."""
    area = c["area"]
    gap = ev == NOBODY
    out = ev.copy()
    if mode == "v2":
        out[gap] = area[gap]
        return out
    idx = np.arange(len(ev))
    last = np.maximum.accumulate(np.where(~gap, idx, -1))
    last_team = np.where(last >= 0, ev[np.maximum(last, 0)], NOBODY)
    since = idx - last
    ok = gap & ((last_team == NOBODY) | (area == last_team) |
                ((area_switch_s is not None) & (since > frames_of(area_switch_s or 0))))
    out[ok] = area[ok]
    return out


def area_persist(ev, c, persist_s):
    """Team-area side only counts once it has held for persist_s (shape after a turnover settles
    over seconds; a brief spread during a counter-press is not a turnover)."""
    a = c["area"].copy()
    starts, lengths = runs(a)
    short = np.repeat(lengths < frames_of(persist_s), lengths)
    a[short] = NOBODY
    gap = ev == NOBODY
    out = ev.copy()
    out[gap] = a[gap]
    return out


def evidence(c, p):
    if p.get("oracle"):                       # DIAGNOSTIC ceiling only: v2 coverage, true team
        ev = evidence(c, {})
        t = c["truth"]
        tt = np.where(t == "H", 0, np.where(t == "A", 1, NOBODY)).astype(np.int8)
        m = (ev != NOBODY) & (tt != NOBODY)
        ev[m] = tt[m]
        return ev
    sp = ball_speed(c["ball_v2"] if p.get("v2") else c["ball"])
    ct = contact(c, p.get("radius", control.CONTROL_RADIUS_M),
                 p.get("margin", control.CLEAR_MARGIN_M), p.get("v2", False))
    if p.get("fast_ms"):                      # pass in flight going past a player: no contact
        ct[(sp > p["fast_ms"])] = NOBODY
    ev = confirmed(ct, p.get("confirm", control.CONFIRM_S), p.get("switch"), p.get("slow"), sp)
    ev = control._hold(ev, frames_of(p.get("hold", control.HOLD_S)))
    if p.get("area_persist"):
        return area_persist(ev, c, p["area_persist"])
    return area_fill(ev, c, p.get("area", "v2"), p.get("area_switch"))


VARIANTS = {
    "base": {},
    "switch0.8": {"switch": 0.8},
    "switch1.0": {"switch": 1.0},
    "switch1.5": {"switch": 1.5},
    "slow5": {"slow": 5.0},
    "slow7": {"slow": 7.0},
    "margin1": {"margin": 1.0},
    "margin1.5": {"margin": 1.5},
    "fast8": {"fast_ms": 8.0},
    "agree": {"area": "agree"},
    "agree10": {"area": "agree", "area_switch": 10.0},
    "persist2": {"area_persist": 2.0},
    "persist4": {"area_persist": 4.0},
    "fast8slow7": {"fast_ms": 8.0, "slow": 7.0},
    "fast10slow7": {"fast_ms": 10.0, "slow": 7.0},
    "fast8slow5": {"fast_ms": 8.0, "slow": 5.0},
    "fast6": {"fast_ms": 6.0},
    "fast10": {"fast_ms": 10.0},
    "slow9": {"slow": 9.0},
    "fast8slow7v2": {"fast_ms": 8.0, "slow": 7.0, "v2": True},
    "fast8slow7sw0.8": {"fast_ms": 8.0, "slow": 7.0, "switch": 0.8},
    "oracle_team": {"oracle": "team"},
    "hold1": {"hold": 1.0},
    "hold2": {"hold": 2.0},
    "hold3": {"hold": 3.0},
    "hold2fast8slow7": {"hold": 2.0, "fast_ms": 8.0, "slow": 7.0},
    "hold3fast8slow7": {"hold": 3.0, "fast_ms": 8.0, "slow": 7.0},
}


def score_game(args):
    g, names = args
    c = dict(np.load(os.path.join(CACHE, "{}.npz".format(g))))
    out = {}
    for name in names:
        ev = evidence(c, VARIANTS[name])
        r_c = scores.control(ev, c["truth"])
        labels = postprocess.finalize(ev, c["dead"], None)
        r_f = scores.full(labels, c["truth"])
        out[name] = (r_c["spellF1"], r_f["spellF1"], r_f["n_pred_spells"], r_c["team_accuracy"],
                     r_f["macroF1"])
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or list(VARIANTS)
    t = time.time()
    games = external.games()
    with Pool(4) as pool:
        rows = pool.map(score_game, [(g, names) for g in games])
    print("labels v{}, {} games, {:.0f}s".format(external.LABELS_VERSION, len(rows), time.time() - t))
    print("{:12s} {:>8s} {:>8s} {:>7s} {:>7s} {:>8s}  even/odd full".format(
        "variant", "control", "full", "spells", "teamacc", "macroF1"))
    for name in names:
        m = np.array([r[name] for r in rows])
        ev, od = m[0::2, 1].mean(), m[1::2, 1].mean()
        print("{:12s} {:8.4f} {:8.4f} {:7.1f} {:7.4f} {:8.4f}  {:.4f}/{:.4f}".format(
            name, *m.mean(axis=0), ev, od), flush=True)

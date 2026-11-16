"""Turnover lab: turnovers missed while the 5 s control hold keeps the old team (lab only;
possession/control.py untouched). External practice set only (labels v1), from Pietro's touch_lab
cache. Scores each variant on the control role (frozen reference smoother, true dead) and the
joint check (live dead ball + live postprocess).

Idea: between the last confirmed control of team T and the next confirmed control of the OTHER
team T', v1 holds T for up to HOLD_S. The turnover happened somewhere in that stretch; the variants
move the hand-over earlier (backdate T' evidence into the stretch) at a football-motivated point.

Run: PYTHONPATH=. python experiments/marco/turnover_lab.py [diag | variant ...]"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pietro"))
import touch_lab as L  # noqa: E402
from possession import control, external, postprocess, scores  # noqa: E402
from possession.features import NOBODY  # noqa: E402

FPS = L.FPS
EXTRA = os.path.join(external.EXTERNAL_DIR, "marco_lab")
f = L.frames_of


def runs_of(x):
    s, n = L.runs(x)
    return [(a, a + b, x[a]) for a, b in zip(s, n)]


def hidden_runs(vis, lo, hi):
    """(start, end) of hidden-ball runs inside [lo, hi)."""
    h = ~vis[lo:hi]
    if not h.any():
        return []
    e = np.flatnonzero(np.diff(np.r_[0, h.astype(np.int8), 0]))
    return [(lo + a, lo + b) for a, b in zip(e[::2], e[1::2])]


def split_point(c, e, s, old, new, mode):
    """Frame where the hand-over from `old` to `new` is placed inside the stretch [e, s)
    (e = first frame after old's last confirmed control, s = first frame of new's)."""
    if mode == "start":
        return e
    if mode == "mid":
        return (e + s) // 2
    vis = c["ball_visible"]
    if mode == "lastnear":
        # just after the last visible frame where old's player was nearest to the ball
        nt = c["nearest_team"][e:s]
        idx = np.flatnonzero(vis[e:s] & (nt == old))
        return e + idx[-1] + 1 if len(idx) else e
    if mode in ("gapstart", "gapmid"):
        # the ball vanished with old and reappeared with new: hand-over inside the last hidden run
        hr = hidden_runs(vis, e, s)
        if not hr:
            return None                       # never hidden: not this lab's case, keep v1
        a, b = hr[-1]
        return a if mode == "gapstart" else (a + b) // 2
    raise ValueError(mode)


def evidence(c, p):
    sp = L.ball_speed(c["ball"])
    ct = L.contact(c)
    if p.get("fast_ms"):
        ct[sp > p["fast_ms"]] = NOBODY
    conf = L.confirmed(ct, control.CONFIRM_S, None, p.get("slow"), sp)
    hold = f(p.get("hold", control.HOLD_S))
    ev = control._hold(conf, hold)
    mode = p.get("split")
    back = f(p.get("back", control.HOLD_S))
    hidden_only = p.get("hidden_only", False)
    vis = c["ball_visible"]
    if mode or p.get("hidden_hold") is not None:
        rr = [r for r in runs_of(conf) if r[2] != NOBODY]
        nxt_other = {}                        # run index -> start of the next run of another team
        for i in range(len(rr)):
            j = next((j for j in range(i + 1, len(rr)) if rr[j][2] != rr[i][2]), None)
            nxt_other[i] = rr[j][0] if j is not None else len(conf)
        for i, ((_, e, old), (s, s_end, new)) in enumerate(zip(rr[:-1], rr[1:])):
            if s - e <= 0:
                continue
            if p.get("hidden_hold") is not None:
                # hold ends once the ball has been hidden for hidden_hold s since old's control
                hid = np.cumsum(~vis[e:s])
                stop = np.flatnonzero(hid > f(p["hidden_hold"]))
                if len(stop):
                    ev[e + stop[0]:min(s, e + hold)] = NOBODY
            if not mode or new == old:
                continue
            if hidden_only and vis[e:s].all():
                continue
            if p.get("min_new") and s_end - s < f(p["min_new"]):
                continue                      # new team's control too short to be a real take-over
            if p.get("persist") and nxt_other[i + 1] - s < f(p["persist"]):
                continue                      # old team back within `persist` s: not a turnover
            k = split_point(c, e, s, old, new, mode)
            if k is None:
                continue
            k = max(k, s - back)
            ev[k:s] = new                     # new team's evidence from the hand-over on
    if p.get("retreat_ms"):
        ev = retreat(c, conf, ev, p)
    return L.area_fill(ev, c, "v2")


def retreat(c, conf, ev, p):
    """Coordinator's hint: during a hold of team T, a visible ball travelling fast toward T's OWN
    goal (>= retreat_ms over 1 s) means the other team is attacking: the hold ends there ('end':
    no evidence, team area decides) or the other team gets the evidence ('switch')."""
    gs = np.load(os.path.join(EXTRA, "{}.npz".format(c["_game"])))["goal_sign"]
    w = f(1.0)                                # source: football reasoning: one pass flight
    ball = c["ball"]
    dx = np.full(len(ball), np.nan)
    dx[w:] = ball[w:, 0] - ball[:-w, 0]
    held = (conf == NOBODY) & (ev != NOBODY)
    t = np.where(held, ev, 0)
    g = gs[np.arange(len(ev)), t]
    toward = dx * g                          # metres moved toward the held team's own goal in 1 s
    trig = held & (toward > p["retreat_ms"] * 1.0)
    if p.get("high"):                        # only when it started in T's attacking half
        x0 = np.full(len(ball), np.nan)
        x0[w:] = ball[:-w, 0]
        trig &= (x0 * g < 0)
    out = ev.copy()
    hold = f(control.HOLD_S)
    for k in np.flatnonzero(trig & ~np.r_[False, trig[:-1]]):
        team = ev[k]
        if out[k] != team:
            continue
        e = k
        while e < len(ev) and e - k < hold and conf[e] == NOBODY and ev[e] == team:
            e += 1
        out[k - w if p.get("from_start") else k:e] = NOBODY if p["retreat"] == "end" else 1 - team
    return out


V = {
    "base": {},
    # hand-over placed at a football point inside an old->new stretch, backdated at most 5 s (the hold)
    "start": {"split": "start"},
    "mid": {"split": "mid"},
    "lastnear": {"split": "lastnear"},
    "gapstart": {"split": "gapstart"},
    "gapmid": {"split": "gapmid"},
    # same, but backdated at most 1 s (one pass flight) or 2 s (pass + reception)
    "mid_b1": {"split": "mid", "back": 1.0},
    "lastnear_b1": {"split": "lastnear", "back": 1.0},
    "lastnear_b2": {"split": "lastnear", "back": 2.0},
    "gapstart_b2": {"split": "gapstart", "back": 2.0},
    # only stretches where the ball was hidden at some point
    "lastnear_hid": {"split": "lastnear", "hidden_only": True},
    "mid_hid": {"split": "mid", "hidden_only": True},
    # hold ends after 1 s / 2 s of hidden ball (team area takes over), visible loose balls keep 5 s
    "hidhold1": {"hidden_hold": 1.0},
    "hidhold2": {"hidden_hold": 2.0},
    # backdate only take-overs that look real: the new team's control lasts >= 1 s (the postprocess
    # switch wait), or the old team does not control again within 2 s / 3 s (a counter-press window)
    "mid_n1": {"split": "mid", "min_new": 1.0},
    "lastnear_n1": {"split": "lastnear", "min_new": 1.0},
    "lastnear_b1_n1": {"split": "lastnear", "back": 1.0, "min_new": 1.0},
    "gapmid_n1": {"split": "gapmid", "min_new": 1.0},
    "mid_p3": {"split": "mid", "persist": 3.0},
    "lastnear_p3": {"split": "lastnear", "persist": 3.0},
    "lastnear_b1_p3": {"split": "lastnear", "back": 1.0, "persist": 3.0},
    "lastnear_p2": {"split": "lastnear", "persist": 2.0},
    "gapmid_p3": {"split": "gapmid", "persist": 3.0},
    "start_p3": {"split": "start", "persist": 3.0},
    # coordinator hint: ball moving fast toward the held team's own goal ends / flips the hold;
    # 8 m/s = Pietro's pass-in-flight speed, 15 m/s = a clearance or long counter pass
    "ret8_end": {"retreat_ms": 8.0, "retreat": "end"},
    "ret8_sw": {"retreat_ms": 8.0, "retreat": "switch"},
    "ret15_end": {"retreat_ms": 15.0, "retreat": "end"},
    "ret15_sw": {"retreat_ms": 15.0, "retreat": "switch"},
    "ret8_hi_sw": {"retreat_ms": 8.0, "retreat": "switch", "high": True},
    "ret8_hi_end": {"retreat_ms": 8.0, "retreat": "end", "high": True},
    # combined with Pietro's best touch filter (fast 8 + slow 7 m/s)
    "f8s7": {"fast_ms": 8.0, "slow": 7.0},
    "lastnear_f8s7": {"split": "lastnear", "fast_ms": 8.0, "slow": 7.0},
    "lastnear_b1_n1_f8s7": {"split": "lastnear", "back": 1.0, "min_new": 1.0, "fast_ms": 8.0, "slow": 7.0},
    "best_combo": {"split": "lastnear", "back": 1.0, "min_new": 1.0, "fast_ms": 8.0, "slow": 7.0,
                   "retreat_ms": 8.0, "retreat": "end"},
    "mid_f8s7": {"split": "mid", "fast_ms": 8.0, "slow": 7.0},
}


def score_game(args):
    g, names = args
    c = dict(np.load(os.path.join(L.CACHE, "{}.npz".format(g))))
    c["_game"] = g
    out = {}
    for name in names:
        ev = evidence(c, V[name])
        r_c = scores.control(ev, c["truth"])
        r_f = scores.full(postprocess.finalize(ev, c["dead"], None), c["truth"])
        out[name] = (r_c["spellF1"], r_f["spellF1"], r_f["n_pred_spells"], r_c["team_accuracy"])
    return out


def diag_game(g):
    """Wrong hold frames: is the next confirmed control the true team (late switch) or not; was
    the ball hidden; length of the stretch."""
    c = dict(np.load(os.path.join(L.CACHE, "{}.npz".format(g))))
    t = c["truth"]
    tt = np.where(t == "H", 0, np.where(t == "A", 1, NOBODY))
    conf = L.confirmed(L.contact(c))
    held = control._hold(conf, f(control.HOLD_S))
    hold = (conf == NOBODY) & (held != NOBODY) & (tt != NOBODY)
    wrong = hold & (held != tt)
    idx = np.arange(len(t))
    nxt = np.minimum.accumulate(np.where(conf != NOBODY, idx, len(t))[::-1])[::-1]
    nxt_team = np.where(nxt < len(t), conf[np.minimum(nxt, len(t) - 1)], NOBODY)
    late = wrong & (nxt_team == tt)
    stretch_hidden = np.zeros(len(t), bool)     # stretch [e,s) contains hidden frames
    rr = [r for r in runs_of(conf) if r[2] != NOBODY]
    for (_, e, _), (s, _, _) in zip(rr[:-1], rr[1:]):
        if s > e and (~c["ball_visible"][e:s]).any():
            stretch_hidden[e:s] = True
    vis = c["ball_visible"]
    return np.array([hold.sum(), wrong.sum(), late.sum(), (wrong & ~vis).sum(),
                     (wrong & stretch_hidden).sum(), (tt != NOBODY).sum(),
                     (hold & ~vis).sum(), (hold & ~vis & (held != tt)).sum()])


if __name__ == "__main__":
    games = external.games()
    t0 = time.time()
    if sys.argv[1:] == ["diag"]:
        with Pool(4) as pool:
            a = np.sum(pool.map(diag_game, games), axis=0)
        hold, wrong, late, wh, ws, live, hh, hhw = a
        print("hold frames {:.3f} of live, wrong {:.3f} of hold".format(hold / live, wrong / hold))
        print("of wrong hold frames: next confirmed = true team (late switch) {:.3f}; ball hidden "
              "that frame {:.3f}; stretch has hidden frames {:.3f}".format(late / wrong, wh / wrong, ws / wrong))
        print("hold frames with hidden ball: {:.3f} of hold, wrong {:.3f}; visible: wrong {:.3f}".format(
            hh / hold, hhw / hh, (wrong - hhw) / (hold - hh)))
        sys.exit()
    names = sys.argv[1:] or list(V)
    with Pool(4) as pool:
        rows = pool.map(score_game, [(g, names) for g in games])
    print("labels v{}, {} games, {:.0f}s".format(external.LABELS_VERSION, len(rows), time.time() - t0))
    print("{:14s} {:>8s} {:>8s} {:>7s} {:>7s}  even/odd full".format("variant", "control", "full", "spells", "teamacc"))
    for name in names:
        m = np.array([r[name] for r in rows])
        print("{:14s} {:8.4f} {:8.4f} {:7.1f} {:7.4f}  {:.4f}/{:.4f}".format(
            name, *m.mean(axis=0), m[0::2, 1].mean(), m[1::2, 1].mean()), flush=True)

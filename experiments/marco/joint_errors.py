"""Joint-pipeline error analysis (diagnostic only, no number chosen here).
Oracle swaps on control evidence -> postprocess.finalize -> scores.full, plus error slices.
External practice set only (SkillCorner, labels v1), from Pietro's touch_lab cache.
Prints aggregate numbers only. Run from repo root:
    PYTHONPATH=.:experiments/pietro python experiments/marco/joint_errors.py
"""
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
NEAR_S = 3.0      # source: football reasoning: a turnover / restart settles within ~3 s
HIDDEN_LONG_S = 1.0  # source: football reasoning: a ball hidden longer than one pass flight


def team_of(truth):
    return np.where(truth == "H", 0, np.where(truth == "A", 1, NOBODY)).astype(np.int8)


def spells(lab):
    """List of (start, end_excl, team) for H/A runs."""
    st, ln = L.runs(lab)
    return [(s, s + n, lab[s]) for s, n in zip(st, ln) if lab[s] in ("H", "A")]


def spell_errors(pred, truth):
    """Own simple matching (not the official one): a true spell is 'found' when one same-team
    predicted spell covers >= 50% of it and is not more than twice as long; 'split' when
    same-team cover >= 50% but by >= 2 predicted spells; else 'missed'. A predicted spell is
    'extra' when < 50% of it lies on same-team truth."""
    g = truth != "X"
    ts, ps = spells(np.where(g, truth, "X")), spells(np.where(g, pred, "X"))
    out = dict(true=len(ts), pred=len(ps), found=0, split=0, merged=0, missed=0, extra=0,
               missed_short=0)
    for s, e, t in ts:
        seg = pred[s:e]
        same = seg == t
        cover = same.mean()
        if cover < 0.5:
            out["missed"] += 1
            out["missed_short"] += int(e - s < 5 * FPS)
            continue
        st, ln = L.runs(seg)
        n_same = sum(1 for a in st if seg[a] == t)
        # predicted spell(s) touching this true spell: does one extend far beyond it?
        if n_same >= 2:
            out["split"] += 1
        else:
            a = s
            while a > 0 and pred[a - 1] == t:
                a -= 1
            b = e
            while b < len(pred) and pred[b] == t:
                b += 1
            if (b - a) > 2 * (e - s):
                out["merged"] += 1
            else:
                out["found"] += 1
    for s, e, t in ps:
        if (truth[s:e] == t).mean() < 0.5:
            out["extra"] += 1
    return out


def scenarios(c):
    t = c["truth"]
    tt = team_of(t)
    ev = L.evidence(c, {})
    ev_or = L.evidence(c, {"oracle": "team"})
    perfect = tt.copy()                                     # true team on every live frame
    live_dead, true_dead = c["dead"].astype(bool), t == "D"
    # partial oracles (sub-ceilings): fix one evidence source or one dead-ball error type
    conf = L.confirmed(L.contact(c))
    held = control._hold(conf, L.frames_of(control.HOLD_S))
    src = np.where(conf != NOBODY, 0, np.where(held != NOBODY, 1, np.where(ev != NOBODY, 2, 3)))
    hidden = np.isnan(c["ball"][:, 0])
    def fix(k, where=True):
        e = ev.copy()
        m = (src == k) & (tt != NOBODY) & (ev != NOBODY) & where
        e[m] = tt[m]
        return e
    no_false_d = live_dead & ~((tt != NOBODY) & (t != "X"))
    dst, dln = L.runs(true_dead.astype(np.int8))
    touched = live_dead.copy()
    found_d = live_dead.copy()
    for s, n in zip(dst, dln):
        if true_dead[s]:
            if live_dead[s:s + n].any():
                found_d[s:s + n] = True                     # complete stops we touched
    all_stops = live_dead | true_dead
    return ev, {
        "base (live C, live D)": (ev, live_dead),
        "1a D: drop false D": (ev, no_false_d),
        "1b D: complete touched": (ev, found_d),
        "1c D: add all missed": (ev, all_stops),
        "2a C: oracle contact": (fix(0), live_dead),
        "2b C: oracle hold": (fix(1), live_dead),
        "2c C: oracle area": (fix(2), live_dead),
        "2b1 hold, ball visible": (fix(1, ~hidden), live_dead),
        "2b2 hold, ball hidden": (fix(1, hidden), live_dead),
        "1 true D": (ev, true_dead),
        "2 oracle C": (ev_or, live_dead),
        "3 oracle C + true D": (ev_or, true_dead),
        "4 perfect C, live D": (perfect, live_dead),
        "5 perfect C + true D": (perfect, true_dead),
    }


def slices(c, ev, pred):
    t = c["truth"]
    tt = team_of(t)
    g = t != "X"
    n_g = g.sum()
    live = tt != NOBODY
    pt = team_of(pred)
    wrong = live & (pt != NOBODY) & (pt != tt)
    false_d = live & (pred == "D")
    missed_d = (t == "D") & (pred != "D") & g
    hidden = np.isnan(c["ball"][:, 0])
    # hidden stretch length so far
    idx = np.arange(len(t))
    last_vis = np.maximum.accumulate(np.where(~hidden, idx, -1))
    hid_for = idx - last_vis
    conf = L.confirmed(L.contact(c))
    held = control._hold(conf, L.frames_of(control.HOLD_S))
    src = np.where(conf != NOBODY, "contact", np.where(held != NOBODY, "hold",
                   np.where(ev != NOBODY, "area", "none")))
    # true events
    prev = np.r_[NOBODY, tt[:-1]]
    turn = live & (prev != NOBODY) & (prev != tt)          # open-play turnover frame
    restart = live & (np.concatenate((["D"], t[:-1])) == "D")           # first live frame after D
    def near(ev_mask, s):                                   # within s seconds after an event
        last = np.maximum.accumulate(np.where(ev_mask, idx, -10 ** 9))
        return (idx - last) <= int(s * FPS)
    def before(ev_mask, s):                                 # within s seconds before an event
        nxt = np.minimum.accumulate(np.where(ev_mask, idx, 10 ** 9)[::-1])[::-1]
        return (nxt - idx) <= int(s * FPS)
    after_turn, after_rest = near(turn, NEAR_S), near(restart, NEAR_S)
    before_turn = before(turn, NEAR_S)
    # predicted-turnover lag: wrong frames where pred still shows the previous true team
    r = dict(n=n_g, wrong=wrong.sum(), false_d=false_d.sum(), missed_d=missed_d.sum())
    w = wrong
    r["w_vis"] = (w & ~hidden).sum()
    r["w_hid"] = (w & hidden).sum()
    r["w_hid_long"] = (w & hidden & (hid_for > HIDDEN_LONG_S * FPS)).sum()
    for s in ("contact", "hold", "area", "none"):
        r["w_src_" + s] = (w & (src == s)).sum()
    r["w_hold_after_hidden"] = (w & (src == "hold") & (hid_for > HIDDEN_LONG_S * FPS)).sum()
    r["w_after_turn"] = (w & after_turn).sum()
    r["w_before_turn"] = (w & before_turn & ~after_turn).sum()
    r["w_after_restart"] = (w & after_rest & ~after_turn).sum()
    r["w_other"] = (w & ~after_turn & ~before_turn & ~after_rest).sum()
    # wrong frames inside long wrong runs (whole stretch on the wrong team) vs short blips
    st, ln = L.runs(w.astype(np.int8))
    long_run = np.repeat([(w[s] and n >= 5 * FPS) for s, n in zip(st, ln)], ln)
    r["w_in_runs_ge5s"] = (w & long_run).sum()
    # false D near restarts / missed D at stop start or end
    r["fd_after_restart"] = (false_d & after_rest).sum()
    r["fd_before_stop"] = (false_d & before(np.r_[False, (t[1:] == "D") & (t[:-1] != "D")], NEAR_S)).sum()
    dst, dln = L.runs((t == "D").astype(np.int8))
    dead_runs = [(s, s + n) for s, n in zip(dst, dln) if t[s] == "D"]
    untouched = np.zeros(len(t), bool)
    for s, e in dead_runs:
        if not (pred[s:e] == "D").any():
            untouched[s:e] = True
    r["md_untouched_stops"] = (missed_d & untouched).sum()
    r["md_partial_stops"] = (missed_d & ~untouched).sum()
    r["true_stops"] = len(dead_runs)
    r["untouched_stops"] = sum(1 for s, e in dead_runs if untouched[s])
    sameteam = 0
    for s, e in dead_runs:
        if untouched[s] and s > 0 and e < len(t) and t[s - 1] == t[e] and t[e] in ("H", "A"):
            sameteam += 1
    r["untouched_same_team"] = sameteam
    return r


def one(g):
    c = dict(np.load(os.path.join(L.CACHE, "{}.npz".format(g))))
    t = c["truth"]
    ev, sc = scenarios(c)
    res = {}
    for name, (e, d) in sc.items():
        lab = postprocess.finalize(e, d, None)
        f = scores.full(lab, t)
        res[name] = (f["spellF1"], f["macroF1"], f["n_pred_spells"], f["n_true_spells"])
        if name.startswith("base"):
            base_lab = lab
    sl = slices(c, ev, base_lab)
    sp = {k: spell_errors(postprocess.finalize(*sc[k], None), t)
          for k in ("base (live C, live D)", "1 true D", "2 oracle C")}
    return res, sl, sp


if __name__ == "__main__":
    t0 = time.time()
    games = external.games()
    with Pool(4) as pool:
        rows = pool.map(one, games)
    print("labels v{}, {} games, {:.0f}s".format(external.LABELS_VERSION, len(rows), time.time() - t0))
    print("\n{:24s} {:>8s} {:>8s} {:>7s} {:>6s}  even/odd full".format(
        "scenario", "full", "macroF1", "pred", "true"))
    for name in rows[0][0]:
        m = np.array([r[0][name] for r in rows])
        print("{:24s} {:8.4f} {:8.4f} {:7.1f} {:6.1f}  {:.4f}/{:.4f}".format(
            name, *m.mean(axis=0), m[0::2, 0].mean(), m[1::2, 0].mean()))
    S = {}
    for r in rows:
        for k, v in r[1].items():
            S[k] = S.get(k, 0) + v
    n = S["n"]
    err = S["wrong"] + S["false_d"] + S["missed_d"]
    print("\nframe errors (base), share of graded frames / of all errors")
    for k in ("wrong", "false_d", "missed_d"):
        print("  {:22s} {:.4f}  {:.3f}".format(k, S[k] / n, S[k] / err))
    print("wrong-team frames split (share of wrong-team frames)")
    for k in sorted(S):
        if k.startswith("w_"):
            print("  {:22s} {:.3f}".format(k, S[k] / max(S["wrong"], 1)))
    print("false D: after restart {:.3f}, before stop {:.3f} (of false D)".format(
        S["fd_after_restart"] / S["false_d"], S["fd_before_stop"] / S["false_d"]))
    print("missed D: untouched stops {:.3f}, partial {:.3f} (of missed D); untouched stops {}/{}"
          ", same team either side {}".format(S["md_untouched_stops"] / S["missed_d"],
                                              S["md_partial_stops"] / S["missed_d"],
                                              S["untouched_stops"], S["true_stops"],
                                              S["untouched_same_team"]))
    print("\nspell-level (own matching, per game mean)")
    for k in rows[0][2]:
        tot = {}
        for r in rows:
            for kk, v in r[2][k].items():
                tot[kk] = tot.get(kk, 0) + v
        print("  {:22s} ".format(k) + " ".join("{} {:.1f}".format(kk, v / len(rows))
                                               for kk, v in tot.items()))

"""Scratch: three ways to LENGTHEN or JOIN the dead runs main already finds, without adding new ones.
20 external games only, never the provided match. Totals and means only.
  holes   two dead runs separated by a short live gap become one run
  walk    a dead run grows backwards / forwards while the players still walk (looser speed than
          rule 2b, which only starts a run), at most a few seconds
  trim    a dead run ends early once the players run again for a sustained time
Scores: dead_F1, the realistic proxy of check_realistic.py (noisy truth control + live postprocess)
and the full pipeline (live control + live postprocess), all, even and odd games.
Run from the repo root:  python experiments/edoardo/sweep_extend.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, control, dead_ball, external, features, postprocess, scores  # noqa: E402

HOLE_S = [1.0, 2.0, 3.0, 5.0]        # source: football reasoning: a possession between two stoppages shorter than a few seconds is rare
WALK_MS = [1.5, 2.0]                 # source: football reasoning: walking to jogging pace (UNSEEN_WALK_MS is 2.0)
WALK_CAP_S = [2.0, 4.0, 8.0]         # source: football reasoning: players slow down over a few seconds after a whistle
TRIM_MS = [2.5, 3.0]                 # source: football reasoning: above jogging pace on average = open play
TRIM_HOLD_S = [1.0, 2.0]             # source: football reasoning: a sustained second or two, not one sprint
SMOOTH_S = 1.0                       # source: same as dead_ball.WALK_SMOOTH_S

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def mean_speed(frames):
    ps = frames["player_speed"]
    tracked = ~np.isnan(ps)
    n_tr = tracked.sum(axis=1)
    mean = np.where(tracked, ps, 0.0).sum(axis=1) / np.maximum(n_tr, 1)
    k = max(int(SMOOTH_S * config.FPS), 1)
    return np.convolve(mean, np.ones(k) / k, mode="same"), n_tr > 0


def close_holes(dead, half, hole_s):
    out = dead.copy()
    for s, e in runs(~dead):
        if 0 < s and e < len(dead) and e - s <= hole_s * config.FPS and half[s - 1] == half[e] != 0:
            out[s:e] = True
    return out


def grow_walk(dead, frames, walk_ms, cap_s, back=True, fwd=True):
    out = dead.copy()
    speed, seen = mean_speed(frames)
    slow = (speed < walk_ms) & seen
    half, cap, n = frames["half"], int(cap_s * config.FPS), len(dead)
    for s, e in runs(dead):
        if back:
            b = s
            while b > 0 and s - b < cap and slow[b - 1] and half[b - 1] == half[s]:
                b -= 1
            out[b:s] = True
        if fwd:
            f = e
            while f < n and f - e < cap and slow[f] and half[f] == half[e - 1]:
                f += 1
            out[e:f] = True
    return out


def trim_run(dead, frames, trim_ms, hold_s):
    out = dead.copy()
    speed, seen = mean_speed(frames)
    fast = (speed > trim_ms) & seen
    hold = int(hold_s * config.FPS)
    for s, e in runs(dead):
        for fs, fe in runs(fast[s:e]):
            if fe - fs >= hold:
                out[s + fs:e] = False
                break
    return out


def variants():
    yield "main", lambda d, fr: d
    for h in HOLE_S:
        yield "holes {:.0f}s".format(h), lambda d, fr, h=h: close_holes(d, fr["half"], h)
    for ms in WALK_MS:
        for cap in WALK_CAP_S:
            yield "walk both {} m/s cap {:.0f}s".format(ms, cap), lambda d, fr, ms=ms, cap=cap: grow_walk(d, fr, ms, cap)
        yield "walk back {} m/s cap 4s".format(ms), lambda d, fr, ms=ms: grow_walk(d, fr, ms, 4.0, fwd=False)
        yield "walk fwd  {} m/s cap 4s".format(ms), lambda d, fr, ms=ms: grow_walk(d, fr, ms, 4.0, back=False)
    for ms in TRIM_MS:
        for hold in TRIM_HOLD_S:
            yield "trim {} m/s {:.0f}s".format(ms, hold), lambda d, fr, ms=ms, hold=hold: trim_run(d, fr, ms, hold)
    yield "walk both 1.5/4 + holes 2s", lambda d, fr: close_holes(grow_walk(d, fr, 1.5, 4.0), fr["half"], 2.0)
    yield "walk both 2.0/4 + holes 2s", lambda d, fr: close_holes(grow_walk(d, fr, 2.0, 4.0), fr["half"], 2.0)


def main():
    games = external.games()
    data = []
    for g in games:
        frames, tags, truth = external.load(g)
        feats = features.compute(frames)
        data.append((frames, truth, dead_ball.detect(frames, feats, tags),
                     control.detect(frames, feats, tags), scores.synthetic_evidence(truth)))
    print("labels v{} reference v{} games {}".format(external.LABELS_VERSION, scores.REFERENCE_VERSION, len(games)))
    print("{:30s} {:>6s} {:>6s} {:>6s} | {:>6s} | {:>6s} {:>6s} {:>6s} {:>6s}".format(
        "variant", "F1", "P", "R", "real.", "full", "even", "odd", "spells"))
    for name, fn in variants():
        rows = []
        for frames, truth, base, evidence, synth in data:
            dead = fn(base, frames)
            d = scores.dead_ball(dead, truth)
            real = scores._evaluate(truth, postprocess.finalize(synth, dead, frames))
            full = scores._evaluate(truth, postprocess.finalize(evidence, dead, frames))
            rows.append((d["dead_F1"], d["dead_precision"], d["dead_recall"], real["spellF1"],
                         full["spellF1"], full["n_pred_spells"]))
        r = np.array(rows)
        m = r.mean(axis=0)
        print("{:30s} {:6.4f} {:6.4f} {:6.4f} | {:6.4f} | {:6.4f} {:6.4f} {:6.4f} {:6.1f}".format(
            name, m[0], m[1], m[2], m[3], m[4], r[0::2, 4].mean(), r[1::2, 4].mean(), m[5]))


if __name__ == "__main__":
    main()

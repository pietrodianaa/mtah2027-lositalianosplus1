"""Scratch: dead-ball variants under a REALISTIC possession check. 20 external games only, never the
provided match. Two proxies, both the official spellF1:
  perfect   teams from the truth, D from us (as in sweep_unseen.py): every true possession, even a
            1 s one, can be matched, so a swallowed mini-possession counts as a full loss
  realistic D from us, teams from postprocess.finalize on scores.synthetic_evidence(truth): the same
            noisy control evidence (ball missing in 1 s bursts, 10% wrong-team flips) and the same
            post-processing the postprocess role score uses. Possessions too short for control +
            post-processing to see are lost anyway, so swallowing them costs nothing
Variants: v2.1, v3, v3.1 (main), v3.1 + v2.3 backdate to the last touch, v3.1 + rule 2b at other
settings, v3.1 + rule 2c (walk near a goal or a line). Totals and means only.
Run from the repo root:  python experiments/edoardo/check_realistic.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, postprocess, scores  # noqa: E402

TOUCH_M = 1.5          # source: football reasoning: v2.3 setting, ball within 1.5 m of a player = at his feet
BACKDATE_SETTINGS = [(2.5, 1.0), (1.5, 2.0), (1.5, 5.0)]   # source: external v1 (sweep_backdate.py): cautious / middle / best dead_F1
ZONE_SETTINGS = [(6.0, 2.0, 3.0), (0.0, 1.5, 5.0)]         # source: external v1 (sweep_zone.py): best dead_F1 / best spell_proxy
BOX_DEPTH_M = 16.5     # source: football reasoning: Law 1, penalty area 16.5 m deep
BOX_WIDTH_M = 40.32    # source: football reasoning: Law 1, penalty area 40.32 m wide
SWALLOWED_SHARE = 0.9  # source: football reasoning: a possession 90% under D is lost for matching

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def proxy_perfect(pred, truth):
    lab = truth.copy()
    for start, end in runs((truth == "D") & ~pred):
        prev = lab[start - 1] if start > 0 else "X"
        nxt = truth[end] if end < len(truth) else "X"
        lab[start:end] = prev if prev in ("H", "A") else nxt if nxt in ("H", "A") else "D"
    lab[pred & (truth != "X")] = "D"
    return scores._evaluate(truth, lab)


def proxy_realistic(pred, truth, frames):
    labels = postprocess.finalize(scores.synthetic_evidence(truth), pred, frames)
    return scores._evaluate(truth, labels)


def backdate(pred, frames, feats, touch_m, back_s):
    """v2.3 replayed on top of any output: a dead run that starts with the ball past a line starts
    at the last touch instead (walk back while nobody is within touch_m, at most back_s)."""
    out = pred.copy()
    length, width = frames.get("pitch", config.PITCH)
    x, y, half = frames["ball"][:, 0], frames["ball"][:, 1], frames["half"]
    with np.errstate(invalid="ignore"):
        past = (x < 0) | (x > length) | (y < 0) | (y > width)
        near = feats["nearest_dist"] <= touch_m
    max_back = int(back_s * config.FPS)
    for start, _ in runs(pred):
        if start == 0 or not past[start]:
            continue
        b = start
        while b > 0 and start - b < max_back and half[b - 1] == half[start] and not near[b - 1]:
            b -= 1
        out[b:start] = True
    return out


def zone_walk(frames, feats, line_m, walk_ms, min_s):
    """Rule 2c replayed (code at fe2c1bd): everyone walks with the ball near a goal or a line."""
    n = len(frames["ball"])
    length, width = frames.get("pitch", config.PITCH)
    half = frames["half"]
    bx, by = feats["ball_v2"][:, 0].copy(), feats["ball_v2"][:, 1].copy()
    for i in range(1, n):
        if np.isnan(bx[i]) and half[i] == half[i - 1]:
            bx[i], by[i] = bx[i - 1], by[i - 1]
    with np.errstate(invalid="ignore"):
        in_box = ((bx <= BOX_DEPTH_M) | (bx >= length - BOX_DEPTH_M)) & (np.abs(by - width / 2) <= BOX_WIDTH_M / 2)
        near = np.minimum(np.minimum(bx, length - bx), np.minimum(by, width - by)) <= line_m
    zone = in_box | (near if line_m > 0 else False)
    ps = frames["player_speed"]
    n_ps = (~np.isnan(ps)).sum(axis=1)
    mean_ps = np.where(np.isnan(ps), 0.0, ps).sum(axis=1) / np.maximum(n_ps, 1)
    k = max(int(dead_ball.WALK_SMOOTH_S * config.FPS), 1)
    smooth = np.convolve(mean_ps, np.ones(k) / k, mode="same")
    w = int(dead_ball.SPEED_WINDOW_S * config.FPS)
    speed = np.full(n, np.nan)
    speed[w:] = np.linalg.norm(frames["ball"][w:] - frames["ball"][:-w], axis=1) / dead_ball.SPEED_WINDOW_S
    visible = ~np.isnan(frames["ball"][:, 0])
    with np.errstate(invalid="ignore"):
        passing = speed > dead_ball.WALK_BALL_FAST_MS
    waiting = zone & (smooth < walk_ms) & (n_ps > 0)
    out = np.zeros(n, dtype=bool)
    for start, end in runs(waiting):
        seen = visible[start:end]
        share = passing[start:end][seen].mean() if seen.any() else 0.0
        if end - start >= int(min_s * config.FPS) and share <= dead_ball.WALK_BALL_FAST_SHARE:
            out[start:end] = True
    return out


variants = [("v2.1", {"UNSEEN_ENABLED": False, "KICKOFF_ENABLED": False}, None),
            ("v3", {"KICKOFF_ENABLED": False}, None),
            ("v3.1 (main)", {}, None)]
variants += [("v3.1 + backdate {} m {} s".format(t, b), {}, ("backdate", t, b)) for t, b in BACKDATE_SETTINGS]
variants += [("v3.1 + 2b 1 s, 6 m/s", {"WALK_BALL_FAST_MS": 6.0}, None),
             ("v3.1 + 2b 3 s, 6 m/s", {"WALK_SMOOTH_S": 3.0, "WALK_BALL_FAST_MS": 6.0}, None)]
variants += [("v3.1 + 2c {} m {} m/s {} s".format(*z), {}, ("zone",) + z) for z in ZONE_SETTINGS]
defaults = {k: getattr(dead_ball, k) for k in ("UNSEEN_ENABLED", "KICKOFF_ENABLED", "WALK_BALL_FAST_MS", "WALK_SMOOTH_S")}

acc = {name: {"f1": [], "perf": [], "real": [], "n_real": [], "swallowed": 0, "sw_short": 0} for name, _, _ in variants}
for g, game_id in enumerate(external.games()):
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    graded = truth != "X"
    for name, params, extra in variants:
        for k, v in defaults.items():
            setattr(dead_ball, k, params.get(k, v))
        pred = dead_ball.detect(frames, feats, tags)
        if extra and extra[0] == "backdate":
            pred = backdate(pred, frames, feats, extra[1], extra[2])
        elif extra and extra[0] == "zone":
            pred = pred | zone_walk(frames, feats, *extra[1:])
        a = acc[name]
        a["f1"].append(scores.dead_ball(pred, truth)["dead_F1"])
        a["perf"].append(proxy_perfect(pred, truth)["spellF1"])
        real = proxy_realistic(pred, truth, frames)
        a["real"].append(real["spellF1"])
        a["n_real"].append(real["n_pred_spells"])
        for team in ("H", "A"):
            for start, end in runs((truth == team) & graded):
                if pred[start:end].mean() >= SWALLOWED_SHARE:
                    a["swallowed"] += 1
                    a["sw_short"] += (end - start) < 2 * config.FPS
    for k, v in defaults.items():
        setattr(dead_ball, k, v)
    print("game {}/{} done".format(g + 1, len(external.games())), flush=True)
    del frames, feats

print("\nlabels v{}; realistic = noisy control (scores.synthetic_evidence) + postprocess.finalize".format(external.LABELS_VERSION))
print("{:<30} {:>8} {:>13} {:>15} {:>12} {:>16}".format("variant", "dead_F1", "proxy perfect", "proxy realistic", "possessions", "swallowed (<2 s)"))
for name, _, _ in variants:
    a = acc[name]
    print("{:<30} {:8.4f} {:13.4f} {:15.4f} {:12.1f} {:>16}".format(
        name, np.mean(a["f1"]), np.mean(a["perf"]), np.mean(a["real"]), np.mean(a["n_real"]),
        "{} ({})".format(a["swallowed"], a["sw_short"])))

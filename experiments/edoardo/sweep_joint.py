"""Scratch: recall-side dead-ball settings scored on the JOINT pipeline (live control + live
postprocess), not on the truth-team proxies. 20 external games only, never the provided match.
Totals and means only.

Why: with live control, a stop we miss is not filled by the right team (as the proxies assume) but
by whatever control says while players wait, and that flickers. decomposition on main (labels v1):
joint 0.5333; with the missed stops added 0.6506; with the false D removed 0.5567; true D 0.6756.
So settings rejected for a lower proxy (more isolated D runs) are re-measured here.
Run from the repo root:  python experiments/edoardo/sweep_joint.py [first_variant last_variant]
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, control, dead_ball, external, features, postprocess, scores  # noqa: E402

BOX_DEPTH_M = 16.5     # source: football reasoning: Law 1, penalty area 16.5 m deep
BOX_WIDTH_M = 40.32    # source: football reasoning: Law 1, penalty area 40.32 m wide

# name, dead_ball overrides, rule 2c (line_m, walk_ms, min_s) or None.   source of every value:
# football reasoning: walking 1-1.5 m/s, a free kick set up in 4-8 s; same grids as sweep_walk.py,
# sweep_zone.py, sweep_unseen.py, sweep_still.py
VARIANTS = [
    ("main", {}, None),
    ("2b walk >= 6 s", {"WALK_MIN_S": 6.0}, None),
    ("2b walk >= 4 s", {"WALK_MIN_S": 4.0}, None),
    ("2b walk < 1.25 m/s", {"WALK_SPEED_MS": 1.25}, None),
    ("2b walk < 1.5 m/s", {"WALK_SPEED_MS": 1.5}, None),
    ("2b walk < 1.25 m/s >= 6 s", {"WALK_SPEED_MS": 1.25, "WALK_MIN_S": 6.0}, None),
    ("2b walk < 1.5 m/s >= 6 s", {"WALK_SPEED_MS": 1.5, "WALK_MIN_S": 6.0}, None),
    ("1c gap >= 2 s", {"UNSEEN_GAP_S": 2.0}, None),
    ("1c no walk check", {"UNSEEN_WALK_SHARE": 0.0}, None),
    ("1c last seen <= 10 m", {"UNSEEN_LAST_M": 10.0}, None),
    ("2 still >= 2 s", {"STILL_MIN_S": 2.0}, None),
    ("2 still clear 1.0 m", {"STILL_CLEAR_M": 1.0}, None),
    ("2c 6 m, 2.0 m/s, 3 s", {}, (6.0, 2.0, 3.0)),
    ("2c box only, 1.5 m/s, 5 s", {}, (0.0, 1.5, 5.0)),
    ("2c 6 m, 1.5 m/s, 5 s", {}, (6.0, 1.5, 5.0)),
    ("2c 6 m, 1.5 m/s, 3 s", {}, (6.0, 1.5, 3.0)),
    ("2b 1.5/6 + 2c 6/2.0/3", {"WALK_SPEED_MS": 1.5, "WALK_MIN_S": 6.0}, (6.0, 2.0, 3.0)),
    ("2b 1.5/4", {"WALK_SPEED_MS": 1.5, "WALK_MIN_S": 4.0}, None),
    ("2b 2.0/6", {"WALK_SPEED_MS": 2.0, "WALK_MIN_S": 6.0}, None),
    ("2b 2.0/4", {"WALK_SPEED_MS": 2.0, "WALK_MIN_S": 4.0}, None),
    ("2b 1.5/4 + 2c 6/2.0/3", {"WALK_SPEED_MS": 1.5, "WALK_MIN_S": 4.0}, (6.0, 2.0, 3.0)),
    ("2b 1.5/6 + 1c gap 2 s + 2c 6/2.0/3", {"WALK_SPEED_MS": 1.5, "WALK_MIN_S": 6.0, "UNSEEN_GAP_S": 2.0}, (6.0, 2.0, 3.0)),
]
DEFAULTS = {k: getattr(dead_ball, k) for _, p, _ in VARIANTS for k in p}


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def zone_walk(frames, feats, line_m, walk_ms, min_s):
    """Rule 2c (code at fe2c1bd, as replayed in check_realistic.py): everyone walks with the ball
    (last known position) in a penalty area or within line_m of a line."""
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


def one_game(args):
    game_id, lo, hi = args
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    evidence = control.detect(frames, feats, tags)
    synth = scores.synthetic_evidence(truth)
    rows = []
    for name, params, zone in VARIANTS[lo:hi]:
        for k, v in DEFAULTS.items():
            setattr(dead_ball, k, params.get(k, v))
        dead = dead_ball.detect(frames, feats, tags)
        if zone:
            dead = dead | zone_walk(frames, feats, *zone)
            for s, e in runs(dead):                       # same MIN_DEAD_S clean-up as the module
                if e - s < int(dead_ball.MIN_DEAD_S * config.FPS):
                    dead[s:e] = False
        d = scores.dead_ball(dead, truth)
        real = scores._evaluate(truth, postprocess.finalize(synth, dead, frames))
        full = scores._evaluate(truth, postprocess.finalize(evidence, dead, frames))
        rows.append((d["dead_F1"], d["dead_precision"], d["dead_recall"], real["spellF1"],
                     full["spellF1"], full["macroF1"], full["n_pred_spells"]))
    return rows


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(VARIANTS)
    games = external.games()
    if not games:
        sys.exit("No external games: run tools/get_external_data.sh first.")
    with Pool(4) as pool:
        per_game = np.array(pool.map(one_game, [(g, lo, hi) for g in games]))   # games x variants x 7
    print("labels v{} reference v{} games {}".format(external.LABELS_VERSION, scores.REFERENCE_VERSION, len(games)))
    print("{:36s} {:>6s} {:>6s} {:>6s} | {:>6s} | {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>5s}".format(
        "variant", "F1", "P", "R", "real.", "JOINT", "even", "odd", "macro", "spells", "wins"))
    base = per_game[:, 0, 4]
    for v, (name, _, _) in enumerate(VARIANTS[lo:hi]):
        r = per_game[:, v]
        m = r.mean(axis=0)
        print("{:36s} {:6.4f} {:6.4f} {:6.4f} | {:6.4f} | {:6.4f} {:6.4f} {:6.4f} {:6.4f} {:6.1f} {:>5s}".format(
            name, m[0], m[1], m[2], m[3], m[4], r[0::2, 4].mean(), r[1::2, 4].mean(), m[5], m[6],
            "{}/{}".format(int((r[:, 4] > base).sum()), len(games)) if lo == 0 else "-"))


if __name__ == "__main__":
    main()

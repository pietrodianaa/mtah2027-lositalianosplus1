"""Scratch: where the isolated D runs of dead ball v3.2 come from, and what tells rule 1c's good runs
from its isolated ones. 20 external games only, never the provided match.
Part A: isolated D runs (touching no true stop) and far ones (>= 10 s from any), by the rule that adds them.
Part B: rule 1c runs (lost ball next to a line, back at a restart spot), split into those touching a true
stop and isolated ones: medians and quartiles of what rule 1c sees. Totals and quantiles only.
Run from the repo root:  python experiments/edoardo/diagnose_isolated_v3.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features  # noqa: E402

FAR_S = 10.0           # source: football reasoning: 10 s from any stoppage = clearly open play
CENTRE_M = 9.15        # source: football reasoning: Law 1, centre circle radius

if not external.games():
    sys.exit("No external games: run tools/get_external_data.sh first.")


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


RULES = [("rule 1c (unseen out)", "UNSEEN_ENABLED"), ("rule 4 (kick-off)", "KICKOFF_ENABLED"),
         ("rule 2b (walk)", "WALK_ENABLED"), ("rule 2 (still)", "STILL_ENABLED"), ("v3.2 backdate", "OUT_BACKDATE")]
part_a = {name: [0, 0] for name, _ in RULES}
part_a["rule 1 (seen out)"] = [0, 0]
feats_b = {"good": {}, "isolated": {}}


def add(group, key, value):
    feats_b[group].setdefault(key, []).append(value)


for game_id in external.games():
    frames, tags, truth = external.load(game_id)
    feats = features.compute(frames)
    graded = truth != "X"
    true_dead = (truth == "D") & graded
    true_idx = np.flatnonzero(true_dead)
    full = dead_ball.detect(frames, feats, tags)
    without = {}
    for name, flag in RULES:
        setattr(dead_ball, flag, False)
        without[name] = dead_ball.detect(frames, feats, tags)
        setattr(dead_ball, flag, True)

    # A: every isolated run, credited to the first rule whose removal shrinks it
    for start, end in runs(full & graded):
        if true_dead[start:end].any():
            continue
        gap = min(abs(true_idx - start).min(), abs(true_idx - (end - 1)).min()) if true_idx.size else np.inf
        who = "rule 1 (seen out)"
        for name, _ in RULES:
            if (~without[name][start:end]).any():
                who = name
                break
        part_a[who][0] += 1
        part_a[who][1] += gap >= FAR_S * config.FPS

    # B: rule 1c runs, good vs isolated
    length, width = frames.get("pitch", config.PITCH)
    ball2 = feats["ball_v2"]
    vis2 = ~np.isnan(ball2[:, 0])
    with np.errstate(invalid="ignore"):
        to_line = np.minimum(np.minimum(ball2[:, 0], length - ball2[:, 0]), np.minimum(ball2[:, 1], width - ball2[:, 1]))
        on_byline = np.minimum(ball2[:, 0], length - ball2[:, 0]) <= np.minimum(ball2[:, 1], width - ball2[:, 1])
    ps = frames["player_speed"]
    n_ps = (~np.isnan(ps)).sum(axis=1)
    mean_ps = np.where(np.isnan(ps), 0.0, ps).sum(axis=1) / np.maximum(n_ps, 1)
    added = full & ~without["rule 1c (unseen out)"]
    nearest = feats["nearest_dist"]
    for start, end in runs(full & graded):
        if not added[start:end].any():
            continue
        group = "good" if true_dead[start:end].any() else "isolated"
        k = start + int(np.flatnonzero(added[start:end])[0])     # first frame rule 1c adds
        p = k - 1                                                # last seen before the gap
        while p > 0 and not vis2[p]:
            p -= 1
        q = k                                                    # first seen after it
        while q < len(vis2) and not vis2[q]:
            q += 1
        if q >= len(vis2) or not vis2[p]:
            continue
        last, back = ball2[p], ball2[q]
        add(group, "unseen gap s", (q - p) / config.FPS)
        add(group, "gap share untracked (X)", float((truth[p:q] == "X").mean()))
        add(group, "last seen: m to line", to_line[p])
        add(group, "back: m to line", to_line[q])
        add(group, "moved while unseen m", float(np.hypot(*(back - last))))
        add(group, "back near byline", float(on_byline[q]))
        add(group, "back: nearest player m", nearest[q])
        add(group, "back with a player <1.5 m", float(nearest[q] < 1.5) if not np.isnan(nearest[q]) else np.nan)
        add(group, "back in centre circle", float(np.hypot(back[0] - length / 2, back[1] - width / 2) <= CENTRE_M))
        sp = mean_ps[p:q][n_ps[p:q] > 0]
        add(group, "players m/s in gap", float(np.median(sp)) if sp.size else np.nan)
        add(group, "D after the ball is back s", max(end - q, 0) / config.FPS)
        add(group, "D run length s", (end - start) / config.FPS)

print("labels v{}, dead ball v3.2, {} games\n".format(external.LABELS_VERSION, len(external.games())))
print("A. isolated D runs by the rule that adds them (total / far >= {:.0f} s)".format(FAR_S))
for name, (n_iso, n_far) in part_a.items():
    print("  {:<24} {:4d} / {:3d}".format(name, n_iso, n_far))
print("\nB. rule 1c runs: touching a true stop vs isolated (median [25th-75th pct]; shares as %)")
print("{:<26} {:>28} {:>28}".format("", "good (n={})".format(len(feats_b["good"].get("unseen gap s", []))),
                                     "isolated (n={})".format(len(feats_b["isolated"].get("unseen gap s", [])))))
for key in feats_b["good"]:
    cells = []
    for group in ("good", "isolated"):
        v = np.array(feats_b[group].get(key, [np.nan]), dtype=float)
        v = v[~np.isnan(v)]
        if not v.size:
            cells.append("-")
        elif key.startswith(("back near", "back in", "back with", "gap share")):
            cells.append("{:.0f}%".format(100 * v.mean()))
        else:
            cells.append("{:.1f} [{:.1f}-{:.1f}]".format(np.median(v), *np.percentile(v, [25, 75])))
    print("{:<26} {:>28} {:>28}".format(key, *cells))

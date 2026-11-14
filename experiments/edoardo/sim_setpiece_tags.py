"""Scratch: what rule 6 (Smart Tagging set pieces) could be worth, simulated on the 20 external games.
External matches have no Smart Tagging, so we build FAKE set-piece tags from the external labels
(never from the provided match): a ~5 s window around the restart that ends a true dead stretch,
as the official loader describes the real tags. The real tags are automatic and miss restarts, so
we also try: only some restarts tagged, false tags in open play, and a window that starts at the
whistle instead of around the restart. This measures the rule's mechanics, not our match.
Metrics: dead_F1, possession proxy with teams = truth, realistic proxy (noisy control + postprocess).
Run from the repo root:  python experiments/edoardo/sim_setpiece_tags.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from possession import config, dead_ball, external, features, postprocess, scores  # noqa: E402

WINDOW_S = 5.0         # source: football reasoning: starter/load_smart_tagging.py, the tag marks a ~5 s window around the restart
MIN_STOP_S = 2.0       # source: football reasoning: gaps between phases under 2 s are not set pieces (labels v1 audit)
TAGGED_SHARE = 0.4     # source: football reasoning: 59 set-piece tags in our match against ~145 dead stretches per external game
FALSE_PER_GAME = 10    # source: football reasoning: a pessimistic count of automatic false positives
SEED = 0               # source: bookkeeping

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
    return scores._evaluate(truth, lab)["spellF1"]


def fake_tags(truth, share, n_false, at_whistle, rng):
    half_w = int(WINDOW_S * config.FPS / 2)
    tags = []
    for start, end in runs((truth == "D") & (truth != "X")):
        if end - start < MIN_STOP_S * config.FPS or end >= len(truth) or rng.random() > share:
            continue
        first = start if at_whistle else end - half_w
        tags.append({"code": "SET PIECES", "start_frame": first, "end_frame": first + 2 * half_w, "tags": {}})
    in_play = np.flatnonzero(np.isin(truth, ["H", "A"]))
    for k in rng.choice(in_play, size=min(n_false, in_play.size), replace=False):
        tags.append({"code": "SET PIECES", "start_frame": int(k), "end_frame": int(k) + 2 * half_w, "tags": {}})
    return tags


SCENARIOS = [("no tags (main)", None),
             ("every restart tagged", (1.0, 0, False)),
             ("40% tagged", (TAGGED_SHARE, 0, False)),
             ("40% + 10 false per game", (TAGGED_SHARE, FALSE_PER_GAME, False)),
             ("40%, window at the whistle", (TAGGED_SHARE, 0, True))]
FILTERS = [0.0, 0.5]   # source: football reasoning: rule 6 without / with the players-walk check (share of the 3 s)

acc = {}
for g, game_id in enumerate(external.games()):
    frames, _, truth = external.load(game_id)
    feats = features.compute(frames)
    for name, sc in SCENARIOS:
        rng = np.random.default_rng(SEED + g)
        tags = [] if sc is None else fake_tags(truth, sc[0], sc[1], sc[2], rng)
        for before in ([None] if sc is None else FILTERS):
            if before is not None:
                dead_ball.SETPIECE_WALK_SHARE = before
            pred = dead_ball.detect(frames, feats, tags)
            key = name if before is None else "{} | {}".format(name, "walk check" if before else "no check")
            a = acc.setdefault(key, {"f1": [], "perf": [], "real": []})
            a["f1"].append(scores.dead_ball(pred, truth)["dead_F1"])
            a["perf"].append(proxy_perfect(pred, truth))
            a["real"].append(scores._evaluate(truth, postprocess.finalize(scores.synthetic_evidence(truth), pred, frames))["spellF1"])
    dead_ball.SETPIECE_WALK_SHARE = 0.5
    print("game {}/{} done".format(g + 1, len(external.games())), flush=True)
    del frames, feats

print("\nlabels v{}; FAKE tags built from the external labels (simulation only)".format(external.LABELS_VERSION))
print("{:<42} {:>8} {:>14} {:>16}".format("scenario | rule 6 version (3 s)", "dead_F1", "proxy perfect", "proxy realistic"))
for key, a in acc.items():
    print("{:<42} {:8.4f} {:14.4f} {:16.4f}".format(key, np.mean(a["f1"]), np.mean(a["perf"]), np.mean(a["real"])))

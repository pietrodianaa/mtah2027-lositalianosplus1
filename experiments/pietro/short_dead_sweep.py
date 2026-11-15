"""Diagnostic: bridge D runs shorter than S seconds when the same team holds the ball on both sides
(a false dead-ball cut splits one possession in two). External practice set only.
Run: PYTHONPATH=. python experiments/pietro/short_dead_sweep.py"""
import numpy as np
from possession import config, external, postprocess as pp

orig = pp.finalize

def bridge(labels, min_len):
    labels = labels.copy()
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    b = np.concatenate(([0], change, [len(labels)]))
    for k in range(1, len(b) - 2):
        s, e = b[k], b[k + 1]
        if labels[s] == "D" and e - s < min_len and labels[s - 1] == labels[e] and labels[e] in "HA":
            labels[s:e] = labels[e]
    return labels

for v in [None, 1.0, 2.0, 3.0]:
    pp.finalize = orig if v is None else (lambda ev, d, f, v=v: bridge(orig(ev, d, f), int(v * config.FPS)))
    out = []
    for role in ("postprocess", "full"):
        r = external.evaluate(role)
        out.append("{} spellF1={:.4f} macroF1={:.4f} pred={:.1f}".format(role, r["spellF1"], r["macroF1"], r["n_pred_spells"]))
    print("BRIDGE_DEAD_S={} | {}".format(v, " | ".join(out)), flush=True)

"""Sweep postprocess options on the postprocess role score only (external practice set, labels v1).
Prints one line per setting: spellF1, macroF1, predicted vs true possessions. Diagnostic; numbers
in postprocess.py still need a football reason. Run: PYTHONPATH=. python experiments/pietro/postprocess_sweep.py"""
import itertools
import sys

from possession import external, postprocess as pp

GRID = {
    "SWITCH_CONFIRM_S": [0.6, 0.8, 1.0, 1.5, 2.0],
    "MIN_SPELL_S": [None, 1.0, 2.0],
    "RESTART_FIX_S": [None, 2.0, 3.0],
    "RESTART_CONFIRM_S": [None, 0.4, 0.6],
}
only = sys.argv[1:]
keys = list(GRID)
for values in itertools.product(*GRID.values()):
    cfg = dict(zip(keys, values))
    if (cfg["RESTART_FIX_S"] is None) != (cfg["RESTART_CONFIRM_S"] is None):
        continue
    for k, v in cfg.items():
        setattr(pp, k, v)
    r = external.evaluate("postprocess")
    print("{} spellF1={:.4f} macroF1={:.4f} pred={:.1f} true={:.1f}".format(
        " ".join("{}={}".format(k, v) for k, v in cfg.items()), r["spellF1"], r["macroF1"],
        r["n_pred_spells"], r["n_true_spells"]), flush=True)

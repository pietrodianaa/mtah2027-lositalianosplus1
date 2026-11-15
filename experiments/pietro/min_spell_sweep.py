"""Sweep MIN_SPELL_S (blip merge) on the postprocess role score and the joint check (external
practice set only). Diagnostic; the chosen value still needs a football reason.
Run: PYTHONPATH=. python experiments/pietro/min_spell_sweep.py"""
from possession import external, postprocess as pp

for v in [None, 1.0, 1.5, 2.0, 3.0]:
    pp.MIN_SPELL_S = v
    out = []
    for role in ("postprocess", "full"):
        r = external.evaluate(role)
        out.append("{} spellF1={:.4f} macroF1={:.4f} pred={:.1f}".format(
            role, r["spellF1"], r["macroF1"], r["n_pred_spells"]))
    print("MIN_SPELL_S={} | {}".format(v, " | ".join(out)), flush=True)

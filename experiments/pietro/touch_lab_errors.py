"""Where does wrong-team control evidence come from? Splits live-play evidence frames by source
(confirmed contact, 5 s hold, team-area fill) and reports share of frames and team accuracy each.
External practice set only. Run: PYTHONPATH=.:experiments/pietro python experiments/pietro/touch_lab_errors.py"""
import os
import numpy as np
from possession import control, external
from possession.features import NOBODY
import touch_lab as L

tot = {}
for g in external.games():
    c = dict(np.load(os.path.join(L.CACHE, "{}.npz".format(g))))
    t = c["truth"]
    tt = np.where(t == "H", 0, np.where(t == "A", 1, NOBODY))
    live = tt != NOBODY
    conf = L.confirmed(L.contact(c))
    held = control._hold(conf, L.frames_of(control.HOLD_S))
    ev = L.area_fill(held, c)
    src = np.where(conf != NOBODY, "contact", np.where(held != NOBODY, "hold",
                   np.where(ev != NOBODY, "area", "none")))
    for s in ("contact", "hold", "area", "none"):
        m = live & (src == s)
        a = tot.setdefault(s, [0, 0])
        a[0] += m.sum(); a[1] += (ev[m] == tt[m]).sum()
n = sum(v[0] for v in tot.values())
for s, (k, ok) in tot.items():
    print("{:8s} share {:.3f}  team accuracy {:.3f}  wrong share of live {:.3f}".format(
        s, k / n, ok / max(k, 1), (k - ok) / n))

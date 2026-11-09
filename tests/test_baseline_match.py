"""Regression anchor: with default config, our pipeline must reproduce baseline_smooth50 exactly.

It compares against the organisers' baseline output, not against any ground truth, so it proves
our frame alignment without tuning on anything. It reads the match, so it runs only when a HUMAN
asks for it (AGENTS.md section 0):   MTAH_HUMAN=1 python -m pytest -q
"""

import os

import numpy as np
import pytest

from possession import dead_ball, postprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
DATA = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")


@pytest.mark.skipif(not os.path.isfile(DATA) or os.environ.get("MTAH_HUMAN") != "1",
                    reason="reads the match: humans only, run with MTAH_HUMAN=1")
def test_reproduces_baseline_smooth50(monkeypatch):
    from possession import pipeline
    monkeypatch.setattr(dead_ball, "ENABLED", False)
    monkeypatch.setattr(postprocess, "SMOOTH_WINDOW", 50)
    ours = pipeline.run(verbose=False)
    reference = np.loadtxt(REFERENCE, delimiter=",", skiprows=1, dtype=str, usecols=1)
    assert len(ours) == len(reference)
    assert (ours == reference).all(), "{} frames differ".format(int((ours != reference).sum()))

"""Fake-data checks for the post-processing rules (no match data)."""
import numpy as np

from possession import postprocess as pp


def test_merge_short_spells_merges_blip_between_same_team():
    labels = np.array(list("HHHHHAAHHHHH"))
    assert "".join(pp.merge_short_spells(labels, 3)) == "HHHHHHHHHHHH"


def test_merge_short_spells_keeps_runs_next_to_dead_ball():
    labels = np.array(list("DDAAHHHHH"))
    assert "".join(pp.merge_short_spells(labels, 3)) == "DDAAHHHHH"


def test_early_switch_after_restart_backfills_to_restart():
    dead = np.array([True] * 2 + [False] * 10)
    evidence = np.array([-1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])   # first frame reads the wrong team
    labels = pp.hysteresis(evidence, dead, confirm=5, restart_fix=4, restart_confirm=2)
    assert "".join(labels) == "DD" + "H" * 10

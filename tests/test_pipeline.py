"""Fast checks that need no match data. Run before every push:  python -m pytest -q

A tiny synthetic match checks every stage honours its contract, the add-only feature contract,
that every number has a source, and that each role's score is independent of the other roles.
"""

import ast
import os

import numpy as np

from possession import control, dead_ball, features, pipeline, postprocess, scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N, SLOTS = 400, 4


def synthetic_frames():
    rng = np.random.default_rng(0)
    players = rng.uniform(0, 60, size=(N, SLOTS, 2))
    ball = players[:, 0, :] + 0.5                 # ball next to slot 0 (home) ...
    ball[200:] = players[200:, 2, :] + 0.5        # ... then next to slot 2 (away)
    ball[150:170] = np.nan                        # a gap
    return {
        "ball": ball, "ball_speed": np.full(N, 5.0),
        "players": players, "player_speed": np.full((N, SLOTS), 3.0),
        "team": np.array([0, 0, 1, 1], dtype=np.int8),
        "half": np.ones(N, dtype=np.int8),
    }


def synthetic_truth():
    truth = np.array(["H"] * 180 + ["D"] * 20 + ["A"] * 200)
    truth[:5] = "X"
    return truth


def test_contracts():
    frames = synthetic_frames()
    feats = features.compute(frames)
    for key, value in feats.items():
        assert len(value) == N, key
    dead = dead_ball.detect(frames, feats, [])
    assert dead.dtype == bool and dead.shape == (N,)
    evidence = control.detect(frames, feats, [])
    assert set(np.unique(evidence)) <= {-1, 0, 1}
    labels = postprocess.finalize(evidence, dead, frames)
    assert labels.shape == (N,) and set(np.unique(labels)) <= {"H", "A", "D", "X"}
    assert pipeline.label(frames, []).shape == (N,)


def test_v0_gets_the_obvious_case_right():
    labels = pipeline.label(synthetic_frames(), [])
    assert (labels[:140] == "H").all()
    assert (labels[260:] == "A").all()


def test_feature_contract_is_add_only():
    frozen = open(os.path.join(ROOT, "tests", "feature_contract.txt")).read().split()
    produced = set(features.compute(synthetic_frames()))
    missing = [name for name in frozen if name not in produced]
    assert not missing, "features removed or renamed (add-only contract): {}".format(missing)


def test_every_number_has_a_source():
    for name in ("config", "features", "dead_ball", "control", "postprocess", "scores", "external"):
        path = os.path.join(ROOT, "possession", name + ".py")
        lines = open(path).read().splitlines()
        for node in ast.parse("\n".join(lines)).body:
            if isinstance(node, ast.Assign) and all(
                    isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
                line = lines[node.lineno - 1]
                if node.targets[0].id in ("TEAM_NAME", "ROOT", "HOME", "AWAY", "NONE", "NOBODY",
                                         "EXTERNAL_DIR", "MATCHES_DIR", "CACHE_DIR", "ROLES"):
                    continue
                assert "# source:" in line or (
                    node.lineno >= 2 and "source:" in lines[node.lineno - 2]), \
                    "{}.py: number without a source: {}".format(name, line.strip())


def test_scores_are_independent_of_other_roles(monkeypatch):
    """Changing control or dead ball must not move the post-processing score, and changing
    post-processing must not move the control score."""
    truth = synthetic_truth()
    frames = synthetic_frames()
    base_post = scores.postprocess(postprocess.finalize, truth, frames)
    monkeypatch.setattr(control, "detect", lambda *a: np.zeros(N, dtype=np.int8))
    monkeypatch.setattr(dead_ball, "detect", lambda *a: np.ones(N, dtype=bool))
    assert scores.postprocess(postprocess.finalize, truth, frames) == base_post

    evidence = np.where(truth == "A", 1, 0).astype(np.int8)
    base_control = scores.control(evidence, truth)
    monkeypatch.setattr(postprocess, "SMOOTH_WINDOW", 3)
    monkeypatch.setattr(postprocess, "finalize", lambda *a: np.full(N, "H"))
    assert scores.control(evidence, truth) == base_control


def test_dead_ball_score():
    truth = synthetic_truth()
    perfect = scores.dead_ball(truth == "D", truth)
    assert perfect["dead_F1"] == 1.0


def test_ball_v2_fills_only_short_gaps():
    n = 60
    ball = np.column_stack((np.arange(n, dtype=float) * 0.4, np.full(n, 30.0)))
    ball[10:15] = np.nan                          # short gap: filled on the straight line
    ball[20:50] = np.nan                          # longer than max_frames: left as NaN
    ball[57:] = np.nan                            # no seen position after it: left as NaN
    half = np.ones(n, dtype=np.int8)
    out, filled = features.fill_ball_gaps(ball, half, max_frames=25, max_step_m=1.4)
    assert filled[10:15].all() and filled.sum() == 5
    assert np.allclose(out[10:15, 0], np.arange(10, 15) * 0.4)
    assert np.isnan(out[20:50, 0]).all() and np.isnan(out[57:, 0]).all()
    seen = ~np.isnan(ball[:, 0])
    assert np.array_equal(out[seen], ball[seen])

    half[13:] = 2                                 # the gap spans half time: not filled
    assert not features.fill_ball_gaps(ball, half, 25, 1.4)[1].any()
    jump = ball.copy()
    jump[15:20, 0] += 40.0                        # 40 m in 6 frames: tracker jump, not filled
    assert not features.fill_ball_gaps(jump, None, 25, 1.4)[1][10:15].any()


def test_ball_v2_leaves_v1_features_unchanged():
    frames = synthetic_frames()
    feats = features.compute(frames)
    assert np.array_equal(np.isnan(frames["ball"][:, 0]), ~feats["ball_visible"])
    assert feats["ball_v2"].shape == (N, 2)
    seen = feats["ball_visible"]
    assert np.array_equal(feats["ball_v2"][seen], frames["ball"][seen])
    assert not feats["ball_v2_filled"][seen].any()

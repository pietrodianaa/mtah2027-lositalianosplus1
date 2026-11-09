"""A tiny fake SkillCorner game checks external.load: 10 -> 25 fps, truth rules, ball hiding."""

import json

import numpy as np

from possession import external

N10 = 100


def write_fake_game(root, game_id="1"):
    folder = root / game_id
    folder.mkdir(parents=True)
    match = {"home_team": {"id": 10}, "pitch_length": 100, "pitch_width": 60,
             "players": [{"id": 1, "team_id": 10}, {"id": 2, "team_id": 20}]}
    (folder / "{}_match.json".format(game_id)).write_text(json.dumps(match))
    with open(folder / "{}_tracking_extrapolated.jsonl".format(game_id), "w") as handle:
        for i in range(N10):
            players = [] if 80 <= i < 90 else [
                {"x": i / 10, "y": 0.0, "player_id": 1, "is_detected": True},
                {"x": -i / 10, "y": 0.0, "player_id": 2, "is_detected": True}]
            ball = {"x": i / 10, "y": 1.0, "z": 0.1, "is_detected": True} if players else \
                {"x": None, "y": None, "z": None, "is_detected": None}
            group = "away team" if i < 30 else None          # disagrees with phase: phase wins
            handle.write(json.dumps({"frame": i, "period": 1, "ball_data": ball,
                                     "possession": {"player_id": None, "group": group},
                                     "player_data": players}) + "\n")
    (folder / "{}_phases_of_play.csv".format(game_id)).write_text(
        "frame_start,frame_end,team_in_possession_id\n0,29,10\n35,59,20\n")


def test_fake_game(tmp_path, monkeypatch):
    write_fake_game(tmp_path)
    monkeypatch.setattr(external, "MATCHES_DIR", str(tmp_path))
    monkeypatch.setattr(external, "CACHE_DIR", str(tmp_path / "cache"))
    assert external.games() == ["1"]
    frames, tags, truth = external.load("1")
    n = int((N10 - 1) * 2.5) + 1
    assert tags == [] and truth.shape == (n,) and frames["players"].shape == (n, 2, 2)
    assert list(frames["team"]) == [0, 1] and list(frames["pitch"]) == [100, 60]
    at = lambda i10: truth[int(i10 * 2.5)]                            # noqa: E731
    assert at(10) == "H" and at(32) == "H" and at(40) == "A"         # short gap = play, last team
    assert at(70) == "D" and at(85) == "X"                            # long gap = D, untracked = X
    assert np.isclose(frames["players"][5, 0, 0], 50 + 0.2)           # 25 fps, origin at corner
    graded = truth != "X"
    assert np.isnan(frames["ball"][graded, 0]).mean() >= external.BALL_MISSING_SHARE
    assert (frames["half"][~graded] == 0).all()
    again = external.load("1")[0]                                     # cached copy is identical
    assert np.array_equal(again["ball"], frames["ball"], equal_nan=True)

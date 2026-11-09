"""[External data + doc] The external matches we may tune on, and per-role scores on them.

Rules (organisers, 3 Nov 2026): never tune on the ground truth, the public 5 minutes, the video
or the provided game. External data is optional: football reasoning is the default source of
every number.

Dataset: SkillCorner open data (github.com/SkillCorner/opendata), MIT license, free: 10 A-League
2024/25 matches, broadcast tracking of players and ball at 10 fps with per-frame possession info.
Clone it into external_data/ (gitignored); never commit it. Credit SkillCorner in the tech doc.

Contract (what the owner implements)
------------------------------------
games()          -> list of game ids found locally (empty until set up)
load(game_id)    -> (frames, tags, truth)
                    frames: same dict as io.load_frames(), RESAMPLED TO 25 FPS (metres, m/s, NaN)
                    tags:   [] (SkillCorner has no Smart Tagging) or a list shaped like io.load_tags()
                    truth:  (N,) 'H' / 'A' / 'D' / 'X', derived BY CODE from SkillCorner's
                            possession info under OUR rules: loose ball stays with the last team,
                            out of play = D. Home = the team listed first.
LABELS_VERSION   bump it (and post on the board) whenever the conversion changes: every score is
                 recorded with the version it was computed on, so old and new are never compared.

Conversion decisions to write in the tech doc: resampling 10 -> 25 fps; possession mapping; how
ball positions are hidden to mimic our ~41% missing ball (do it once, seeded, saved with the
version, so every role sees the same degraded input).

evaluate(role) is already written (lead): it runs only the role asked for, with the other roles
replaced by the truth (possession/scores.py), and averages over games.
"""

import csv
import json
import os

import numpy as np

from . import config, control, dead_ball, features, postprocess, scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DIR = os.path.join(ROOT, "external_data")
MATCHES_DIR = os.path.join(EXTERNAL_DIR, "opendata", "data", "matches")
LABELS_VERSION = 1             # source: bookkeeping (0 = not set up yet; bump on every change)
CACHE_DIR = os.path.join(EXTERNAL_DIR, "cache_v{}".format(LABELS_VERSION))

SC_FPS = 10                    # source: data format (SkillCorner README: tracking at 10 fps)
SHORT_GAP_S = 2.0              # source: football reasoning: a real stoppage takes longer than 2 s, so shorter gaps between phases are play
MASK_SEED = 0                  # source: bookkeeping (fixed seed: every role sees the same hidden ball)
BALL_MISSING_SHARE = 0.41      # source: DATA.md (ball missing on ~41% of graded frames)
BALL_MISSING_BURST_S = 1.0     # source: football reasoning: the tracker loses the ball in short bursts (same as scores.py)

ROLES = ("dead_ball", "control", "postprocess", "full")
HOME, AWAY, NONE = 0, 1, -1


def _path(game_id, suffix):
    return os.path.join(MATCHES_DIR, str(game_id), "{}_{}".format(game_id, suffix))


def games():
    """Games whose tracking file is real (not a git-lfs pointer)."""
    if not os.path.isdir(MATCHES_DIR):
        return []
    found = [g for g in os.listdir(MATCHES_DIR)
             if os.path.isfile(_path(g, "tracking_extrapolated.jsonl"))
             and os.path.getsize(_path(g, "tracking_extrapolated.jsonl")) > 1000]   # lfs pointer ~130 B
    return sorted(found, key=int)


def read_game(game_id):
    """SkillCorner match at 10 fps: positions (metres, origin at a corner) and possession info."""
    match = json.load(open(_path(game_id, "match.json")))
    length, width = match["pitch_length"], match["pitch_width"]
    home_id = match["home_team"]["id"]
    people = sorted(match["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    slot = {p["id"]: i for i, p in enumerate(people)}
    team = np.array([HOME if p["team_id"] == home_id else AWAY for p in people], dtype=np.int8)

    rows = [json.loads(line) for line in open(_path(game_id, "tracking_extrapolated.jsonl"))]
    n = len(rows)
    assert all(r["frame"] == i for i, r in enumerate(rows)), "frames not contiguous"
    players = np.full((n, len(people), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    tracked = np.zeros(n, dtype=bool)
    period = np.zeros(n, dtype=np.int8)
    group = np.full(n, NONE, dtype=np.int8)
    for i, r in enumerate(rows):
        period[i] = r["period"] or 0
        group[i] = {"home team": HOME, "away team": AWAY}.get(r["possession"]["group"], NONE)
        tracked[i] = bool(r["player_data"])
        for p in r["player_data"]:
            players[i, slot[p["player_id"]]] = (p["x"] + length / 2, p["y"] + width / 2)
        b = r["ball_data"]
        if b.get("is_detected") and b.get("x") is not None:   # extrapolated ball counts as missing
            ball[i] = (b["x"] + length / 2, b["y"] + width / 2)

    phases = []                                  # (first, last, team or None), frames inclusive
    for p in csv.DictReader(open(_path(game_id, "phases_of_play.csv"))):
        tid = p["team_in_possession_id"]
        phases.append((int(p["frame_start"]), int(p["frame_end"]),
                       None if not tid else (HOME if int(tid) == home_id else AWAY)))
    phases.sort()
    return {"players": players, "ball": ball, "team": team, "tracked": tracked, "period": period,
            "group": group, "phases": phases, "pitch": np.array((length, width), dtype=float)}


def _fill(values):
    """Forward-fill NONE, then back-fill what is left at the start."""
    out = values.copy()
    for order in (slice(None), slice(None, None, -1)):
        seq = out[order]
        for i in range(1, len(seq)):
            if seq[i] == NONE:
                seq[i] = seq[i - 1]
    return out


def truth_10fps(game):
    """H/A/D/X at 10 fps under OUR rules: in a phase of play the phase's team has the ball (loose
    ball included); between phases the ball is out of play (D), unless the gap is shorter than
    SHORT_GAP_S; untracked frames (replays) and frames outside the periods are X."""
    n = len(game["tracked"])
    team = np.full(n, NONE, dtype=np.int8)
    inplay = np.zeros(n, dtype=bool)
    for first, last, phase_team in game["phases"]:
        inplay[first:last + 1] = True
        team[first:last + 1] = (phase_team if phase_team is not None
                                else _fill(game["group"][first:last + 1]))
    short = int(SHORT_GAP_S * SC_FPS)
    for (_, prev_last, _), (next_first, _, _) in zip(game["phases"], game["phases"][1:]):
        if 0 < next_first - prev_last - 1 < short:
            inplay[prev_last + 1:next_first] = True
            team[prev_last + 1:next_first] = team[prev_last]          # loose ball: last team
    truth = np.full(n, "D", dtype="<U1")
    truth[inplay & (team == HOME)] = "H"
    truth[inplay & (team == AWAY)] = "A"
    truth[~game["tracked"] | (game["period"] == 0)] = "X"
    return truth


def _resample(values, src):
    """Linear interpolation at fractional 10 fps indices; NaN if either neighbour is NaN."""
    low = np.floor(src).astype(int)
    high = np.minimum(low + 1, len(values) - 1)
    w = (src - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return np.where(w == 0, values[low], values[low] * (1 - w) + values[high] * w)


def _speed(positions):
    step = np.linalg.norm(np.diff(positions, axis=0), axis=-1) * config.FPS
    return np.concatenate((np.full((1,) + step.shape[1:], np.nan), step))


def _hide_ball(missing, graded, game_id):
    """Add seeded 1 s bursts until the ball is missing on BALL_MISSING_SHARE of graded frames."""
    rng = np.random.default_rng([MASK_SEED, int(game_id)])
    burst = int(BALL_MISSING_BURST_S * config.FPS)
    missing = missing.copy()
    while (missing & graded).sum() < BALL_MISSING_SHARE * graded.sum():
        start = rng.integers(0, len(missing) - burst)
        missing[start:start + burst] = True
    return missing


def _build(game_id):
    game = read_game(game_id)
    n10 = len(game["tracked"])
    src = np.arange(int((n10 - 1) * config.FPS / SC_FPS) + 1) * SC_FPS / config.FPS
    nearest = np.minimum(np.round(src).astype(int), n10 - 1)
    truth = truth_10fps(game)[nearest]
    graded = truth != "X"
    players = _resample(game["players"], src)
    ball = _resample(game["ball"], src)
    hidden = _hide_ball(np.isnan(ball[:, 0]), graded, game_id)
    ball[hidden] = np.nan
    frames = {"ball": ball, "ball_speed": _speed(ball), "players": players,
              "player_speed": _speed(players), "team": game["team"],
              "half": np.where(graded, game["period"][nearest], 0).astype(np.int8),
              "pitch": game["pitch"]}
    return frames, truth


def load(game_id):
    """(frames, tags, truth) at 25 fps; cached per labels version under external_data/."""
    path = os.path.join(CACHE_DIR, "{}.npz".format(game_id))
    if not os.path.isfile(path):
        frames, truth = _build(game_id)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, truth=truth, **frames)
    with np.load(path) as data:
        frames = {key: data[key] for key in data.files if key != "truth"}
        truth = data["truth"]
    return frames, [], truth


def _score_one(role, frames, tags, truth):
    if role == "postprocess":
        return scores.postprocess(postprocess.finalize, truth, frames)
    feats = features.compute(frames)
    if role == "dead_ball":
        return scores.dead_ball(dead_ball.detect(frames, feats, tags), truth)
    if role == "control":
        return scores.control(control.detect(frames, feats, tags), truth)
    from . import pipeline
    return scores.full(pipeline.label(frames, tags), truth)


def evaluate(role, game_ids=None):
    if role not in ROLES:
        raise ValueError("role must be one of {}".format(ROLES))
    rows = [_score_one(role, *load(g)) for g in (game_ids or games())]
    result = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    result.update({"role": role, "n_games": len(rows), "labels_version": LABELS_VERSION,
                   "reference_version": scores.REFERENCE_VERSION})
    return result

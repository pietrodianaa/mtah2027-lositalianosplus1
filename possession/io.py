"""[Integration] Load the match data once, cache it, and write submissions.

load_frames() returns a dict of numpy arrays, every per-frame array indexed by VIDEO frame
(0 .. N_FRAMES-1, the submission row). This dict is the contract every other module reads:

    ball          (N, 2)      ball x, y in metres, NaN when the tracker lost the ball
    ball_speed    (N,)        m/s, NaN when missing
    players       (N, 46, 2)  x, y in metres per track slot, NaN when the slot is empty
    player_speed  (N, 46)     m/s
    team          (46,)       0 = home (Barcelona, H), 1 = away (Athletic, A)
    half          (N,)        1 or 2 on graded frames, 0 elsewhere (pre-match, halftime, after)
    pitch         (2,)        pitch length, width in metres

A track slot is a tracklet, not a person: slots appear, vanish and swap identity.

Parsing the 196 MB text file takes a minute or two, so the result is cached in cache/frames.npz
(gitignored). Delete that file to force a re-parse.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, require  # noqa: E402

from . import config  # noqa: E402

CACHE = os.path.join(ROOT, "cache", "frames.npz")
HOME, AWAY = 0, 1


def _team_of_each_track(meta_path):
    """0/1 per track slot. First team in the XML is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_raw(raw_path, n_slots):
    """Stream the raw ATD: '<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>' with NaN for missing."""
    n = config.N_FRAMES
    length, width = config.PITCH
    players = np.full((n, n_slots, 2), np.nan)
    player_speed = np.full((n, n_slots), np.nan)
    ball = np.full((n, 2), np.nan)
    ball_speed = np.full(n, np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1          # ATD frame_id counts from 1, video frames from 0
            for slot, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                x, y, s = group.split(",")
                players[frame, slot, 0] = float(x) * length
                players[frame, slot, 1] = float(y) * width
                player_speed[frame, slot] = float(s) * config.KPH_TO_MS
            if parts[2][0] != "N":
                x, y, s = parts[2].split(",")
                ball[frame] = (float(x) * length, float(y) * width)
                ball_speed[frame] = float(s) * config.KPH_TO_MS
    return players, player_speed, ball, ball_speed


def _half_index():
    half = np.zeros(config.N_FRAMES, dtype=np.int8)
    for h, first, last in half_video_blocks(GAME):
        half[first:last + 1] = h
    return half


def load_frames(refresh=False):
    """The frame dict described above. Cached after the first call."""
    if os.path.isfile(CACHE) and not refresh:
        with np.load(CACHE) as data:
            frames = {key: data[key] for key in data.files}
        frames["pitch"] = np.array(config.PITCH)
        return frames

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    team = _team_of_each_track(meta_path)
    print("parsing {} (one-off, then cached) ...".format(os.path.basename(raw_path)))
    players, player_speed, ball, ball_speed = _parse_raw(raw_path, len(team))
    frames = {
        "ball": ball, "ball_speed": ball_speed,
        "players": players, "player_speed": player_speed,
        "team": team, "half": _half_index(),
    }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **frames)
    frames["pitch"] = np.array(config.PITCH)
    return frames


def load_tags():
    """Smart Tagging as [{code, start_frame, end_frame, tags}]; end_frame is exclusive.

    Team tag values are club names ('Barcelona' -> H, 'Atletic Club' -> A, 'N/A' for every
    throw-in). Match codes and values case-insensitively.
    """
    from starter.load_smart_tagging import load
    return load()


def write_submission(path, labels):
    """frame,label CSV, one row per video frame. Run the validator on the result."""
    labels = np.asarray(labels)
    assert len(labels) == config.N_FRAMES, "need {} labels, got {}".format(
        config.N_FRAMES, len(labels))
    assert set(np.unique(labels)) <= {"H", "A", "D", "X"}, "labels must be H/A/D/X"
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path

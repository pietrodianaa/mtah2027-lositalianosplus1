"""[Integration] Glue the stages together. Changes here go through the lead (CODEOWNERS)."""

from . import control, dead_ball, features, io, postprocess


def label(frames, tags):
    """Frame dict (io.load_frames contract) + tags -> labels. Same code for any match, so the
    pipeline runs unchanged on the provided game and on external games (external.py)."""
    feats = features.compute(frames)
    dead = dead_ball.detect(frames, feats, tags)
    evidence = control.detect(frames, feats, tags)
    return postprocess.finalize(evidence, dead, frames)


def run(refresh_cache=False, verbose=True):
    """Labels for the provided game (BAR-ATH). Run by a human, never by an agent (AGENTS.md §0)."""
    frames = io.load_frames(refresh=refresh_cache)
    labels = label(frames, io.load_tags())
    if verbose:
        print("labels written for {} frames".format(len(labels)))
    return labels

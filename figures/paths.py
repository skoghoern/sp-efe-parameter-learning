"""Locations for saved arrays and the images drawn from them."""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, "data")
RENDER_DIR = os.path.join(HERE, "figures")


def split_stem(stem: str) -> tuple[str, str]:
    """Return ``(array_stem, image_stem)`` for a save prefix.

    A prefix inside this directory is split: arrays under ``data/``, images
    under ``figures/``, with the same relative path. ``figures/horizon_sweep/run``
    becomes ``figures/data/horizon_sweep/run`` and
    ``figures/figures/horizon_sweep/run``. A prefix that already names one of
    those folders is paired with the other. A prefix outside this directory is
    used for both, so an explicit path keeps the npz next to its plots.
    """
    absolute = os.path.isabs(stem)
    abs_stem = os.path.abspath(stem)

    def _rel(root: str):
        root = os.path.abspath(root)
        if abs_stem == root or abs_stem.startswith(root + os.sep):
            return os.path.relpath(abs_stem, root)
        return None

    def _pair(rel: str) -> tuple[str, str]:
        data = os.path.join(DATA_DIR, rel)
        image = os.path.join(RENDER_DIR, rel)
        if absolute:
            return data, image
        return os.path.relpath(data), os.path.relpath(image)

    rel = _rel(DATA_DIR)
    if rel is not None:
        return _pair(rel)
    rel = _rel(RENDER_DIR)
    if rel is not None:
        return _pair(rel)
    rel = _rel(HERE)
    if rel is not None and rel.split(os.sep, 1)[0] not in ("data", "figures"):
        return _pair(rel)
    return stem, stem

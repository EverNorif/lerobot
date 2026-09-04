#!/usr/bin/env python
"""Example user transform for ``lerobot-edit-dataset transform-features``.

Usage::

    lerobot-edit-dataset transform-features \\
        --input path/to/ds \\
        --transform examples/transform_state_example.py \\
        --features observation.state action \\
        --output path/to/out

Edit ``transform`` / ``feature_meta`` for your own arithmetic or dimension edits.
``x`` is always a float array of shape ``(T, D)`` for one episode.
"""

from __future__ import annotations

import numpy as np

# Drop this observation.state index (example). Set to None to keep all dims.
DROP_STATE_DIM: int | None = None


def transform(key: str, x: np.ndarray) -> np.ndarray:
    """Apply per-feature transforms. Return ``(T, D')`` (``T`` must be unchanged)."""
    if key == "observation.state":
        y = x.copy()
        # Example arithmetic on specific dims (no-ops if D is smaller — guard first).
        if y.shape[1] > 0:
            y[:, 0] = y[:, 0] + 0.0
        if y.shape[1] > 1:
            y[:, 1] = y[:, 1] * 1.0
        if DROP_STATE_DIM is not None:
            y = np.delete(y, DROP_STATE_DIM, axis=-1)
        return y

    if key == "action":
        # Identity by default; customize as needed.
        return x

    return x


def feature_meta(key: str, feature: dict) -> dict:
    """Update shape/names after transform (required when deleting dims)."""
    ft = dict(feature)
    if key == "observation.state" and DROP_STATE_DIM is not None and ft.get("names"):
        names = list(ft["names"])
        del names[DROP_STATE_DIM]
        ft["names"] = names
        ft["shape"] = (len(names),)
    return ft

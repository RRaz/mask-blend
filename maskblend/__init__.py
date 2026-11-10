"""Mask growth, crop planning and distance-ramp paste-back.

Extracted from the LanPaint-for-Affinity experiment. See ``core.py`` for the
convention: 0 replaces, 255 keeps, grey counts as selected.
"""

from .core import (
    DEFAULT_GRID,
    DEFAULT_THRESHOLD,
    binary_mask,
    fill_alpha,
    grow_mask,
    mask_box,
    pad_to_grid,
    paste_back,
    plan_crop,
)

__all__ = [
    "DEFAULT_GRID",
    "DEFAULT_THRESHOLD",
    "binary_mask",
    "fill_alpha",
    "grow_mask",
    "mask_box",
    "pad_to_grid",
    "paste_back",
    "plan_crop",
]

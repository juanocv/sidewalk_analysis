"""
Obstacle label helpers.

Segmenters emit instance labels like ``"tree#4:base2"``; several layers need the
bare type behind them. This module is a dependency-free leaf so both
``sidewalk_ai.io`` and ``sidewalk_ai.processing`` can use it without one
importing the other.
"""

from __future__ import annotations

import re

__all__ = ["label_to_type"]

_LABEL_TYPE_RE = re.compile(r"^([a-zA-Z0-9 _\-]+)")


def label_to_type(label: str) -> str:
    """``"tree#4:base2"`` -> ``"tree"``."""
    match = _LABEL_TYPE_RE.match(label)
    return match.group(1).strip().lower() if match else label.lower()

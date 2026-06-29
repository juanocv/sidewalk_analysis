# sidewalk_ai/__init__.py
"""
Sidewalk-AI – Automatic sidewalk-width estimation & obstacle detection
=====================================================================

Main entry‐points
-----------------
* :class:`~sidewalk_ai.core.pipeline.SidewalkPipeline` – high-level, one-call API
* :func:`~sidewalk_ai.models.factory.build_segmenter`  – convenience model factory
* :class:`~sidewalk_ai.models.midas.MidasEstimator`    – depth wrapper
* :func:`~sidewalk_ai.io.streetview.StreetViewClient`  – Google Street View I/O
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = [
    "SidewalkPipeline",
    "build_segmenter",
    "build_depth",
    "MidasEstimator",
    "StreetViewClient",
    "Coordinate",
    "haversine",
]

# ------------------------------------------------------------
# 1)  Re-export light helpers immediately
# ------------------------------------------------------------
from .io.geo import Coordinate, haversine

# ------------------------------------------------------------
# 2)  Lazy re-exports for heavy modules (Torch, Detectron2…)
# ------------------------------------------------------------
_lazy_map: dict[str, str] = {
    "SidewalkPipeline": "sidewalk_ai.core.pipeline",
    "build_segmenter": "sidewalk_ai.models.factory",
    "build_depth": "sidewalk_ai.models.factory",
    "MidasEstimator": "sidewalk_ai.models.midas",
    "StreetViewClient": "sidewalk_ai.io.streetview",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _lazy_map:
        mod: ModuleType = import_module(_lazy_map[name])
        obj = getattr(mod, name)
        globals()[name] = obj  # cache for next time
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# sidewalk_ai/io/image_io.py
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central place for image-loading defaults."""
    auto_crop_google_logo: bool = True
    google_bar_height_px: int   = 20          # adjust if Google changes UI
    class Config:
        env_prefix = "SWAI_IMG_"


_cfg = Settings()


class ImageLoadError(RuntimeError):
    """Raised when an image cannot be decoded."""


def read_rgb(
    src: Union[str, Path, bytes, np.ndarray],
    *,                           # force kwargs after this
    crop_bar: bool | None = None,
) -> np.ndarray:
    """
    Load an image and **always** return H×W×3 uint8 RGB.

    Parameters
    ----------
    src
        Path/str, raw bytes, or already-loaded BGR/RGB ndarray.
    crop_bar
        Overrides the default behaviour of cropping the 20-pixel
        Street-View logo bar at the bottom of the frame.

    Raises
    ------
    ImageLoadError  – if OpenCV/Pillow cannot decode the bytes.
    """
    crop_bar = _cfg.auto_crop_google_logo if crop_bar is None else crop_bar

    # ─── handle the 3 possible input types ────────────────────────────────
    if isinstance(src, (str, Path)):
        arr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if arr is None:
            raise ImageLoadError(f"OpenCV failed to read “{src}”.")
    elif isinstance(src, bytes):
        arr = cv2.imdecode(np.frombuffer(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ImageLoadError("OpenCV failed to decode in-memory bytes.")
    else:                                         # already a NumPy array
        arr = src.copy()

    # ─── assure channel order + optional crop ─────────────────────────────
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)   # idempotent if already RGB

    if crop_bar and rgb.shape[0] > _cfg.google_bar_height_px:
        rgb = rgb[:-_cfg.google_bar_height_px, :]

    return rgb
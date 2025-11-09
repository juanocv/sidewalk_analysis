# sidewalk_ai/io/image_io.py
from __future__ import annotations

import base64
from pathlib import Path
import re
from typing import Union

import cv2
import numpy as np
from pydantic_settings import BaseSettings

from sidewalk_ai.processing.accessibility import _label_to_type

# ─── color palette for known types ────────────────────────────────

_PALETTE = {
    "tree": (0,160,0),
    "grass": (0,200,0),
    "plant": (0,180,180),
    "signboard": (80,80,255),
    "step": (220,220,220),
    "stair": (220,220,220),
    "pole": (180,0,180),
    "bench": (0,140,255),
}

# ─── regex for label type extraction ───────────────────────────────

_LABEL_TYPE_RE = re.compile(r"^([a-zA-Z0-9 _\-]+)")

# ─── configuration ────────────────────────────────────────────────

class Settings(BaseSettings):
    """Central place for image-loading defaults."""
    auto_crop_google_logo: bool = False
    google_bar_height_px: int   = 20          # adjust if Google changes UI
    class Config:
        env_prefix = "SWAI_IMG_"


_cfg = Settings()

# ─── custom exceptions ────────────────────────────────────────────

class ImageLoadError(RuntimeError):
    """Raised when an image cannot be decoded."""

# ─── main function ────────────────────────────────────────────────

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

# ─── auxiliary functions ─────────────────────────────────────────────

def _color_for_type(t: str) -> tuple[int,int,int]:
    if t in _PALETTE:
        return _PALETTE[t]
    h = abs(hash(t)) & 0xFFFFFF
    r = 50 + (h & 0xFF) % 206
    g = 50 + ((h >> 8) & 0xFF) % 206
    b = 50 + ((h >> 16) & 0xFF) % 206
    return (int(b), int(g), int(r))  # BGR

def objects_overlay_bgr(rgb_bgr: np.ndarray, obstacles) -> np.ndarray:
    """
    Desenha overlay colorido por tipo de obstáculo.
    `obstacles` deve ser lista de (label, mask_bool/uint8).
    """
    if not obstacles:
        return rgb_bgr
    out = rgb_bgr.copy()
    for lbl, m in obstacles:
        if m is None:
            continue
        t = _label_to_type(lbl)
        color = _color_for_type(t)
        mask = (m > 0) if m.dtype != bool else m
        # pinta região (blend local)
        out[mask] = (0.6 * out[mask] + 0.4 * np.array(color)).astype(np.uint8)
        # contorno fino ajuda a “separar” objetos vizinhos
        try:
            cnts,_ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, color, 2)
        except Exception:
            pass
    return out

def png_b64(arr: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", arr)[1]).decode()

def png_triplet(result):
    if getattr(result, "rgb_image", None) is None:
        return None
    rgb_bgr = cv2.cvtColor(result.rgb_image, cv2.COLOR_RGB2BGR)

    # overlay calçada
    over_sw = rgb_bgr.copy()
    over_sw[result.sidewalk_mask.astype(bool)] = (0,255,0)
    over_sw = cv2.addWeighted(over_sw, 0.4, rgb_bgr, 0.6, 0)

    # overlay objetos
    over_obj = objects_overlay_bgr(rgb_bgr, getattr(result, "obstacles", None))

    return {
        "gsv_png_b64": png_b64(rgb_bgr),
        "overlay_sidewalk_png_b64": png_b64(over_sw),
        "overlay_obstacle_png_b64": png_b64(over_obj),
        #"width_m": float(getattr(result.width, "width_m", 0.0) or 0.0),
        #"margin_m": float(getattr(result.width, "margin_m", 0.0) or 0.0),
    }

def sample_indices(n, k=3):
    if n<=0: return []
    if n<=k: return list(range(n))
    return sorted(set([n//4, n//2, (3*n)//4]))[:k]
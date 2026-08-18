"""Camera intrinsics, and the horizon they place.

`cy` used to be hardcoded to 200, the half-height of a 600x400 Street View frame.
`ImageRequest.size` is configurable and the API serves up to 640x640, so any other
capture height put the horizon on the wrong row — and the horizon is what the
width band is measured against.
"""

from __future__ import annotations

import numpy as np
import pytest

from sidewalk_ai.io import image_io
from sidewalk_ai.processing.geometry import _ground_intersection, _intrinsics_after_crop


def _horizon_row(H: int, *, fov_deg: float = 90.0, pitch_deg: float = 0.0) -> float:
    _, fy, _, cy = _intrinsics_after_crop(600, H, fov_deg)
    return cy - fy * np.tan(np.radians(pitch_deg))


def test_the_standard_frame_is_unchanged():
    # 600x400 is what the pipeline actually fetches, and 400/2 is the 200 the
    # old constant hardcoded: this generalisation must not move it.
    fx, fy, cx, cy = _intrinsics_after_crop(600, 400, 90.0)

    assert cx == 300.0
    assert cy == 200.0
    assert fx == pytest.approx(300.0)
    assert fy == fx


@pytest.mark.parametrize("height", [320, 400, 480, 640])
def test_the_principal_point_follows_the_frame(height):
    _, _, _, cy = _intrinsics_after_crop(600, height, 90.0)

    assert cy == height / 2


def test_a_taller_frame_moves_the_horizon():
    # With the constant, a 640-tall frame kept its horizon at row 200 instead of
    # 320, so 120 rows of sky counted as ground and the band was measured against
    # the wrong reference.
    assert _horizon_row(640) == 320.0
    assert _horizon_row(640) - _horizon_row(400) == 120.0


def test_cropping_the_bottom_leaves_the_horizon_where_it_was():
    # Trimming the logo strip does not move the rows above it, so the principal
    # point has to be reported in the original frame's coordinates.
    full = _intrinsics_after_crop(600, 400, 90.0)
    cropped = _intrinsics_after_crop(600, 380, 90.0, crop_bottom_px=20)

    assert cropped == full


def test_a_crop_left_undeclared_shifts_the_horizon():
    # Guards the parameter itself: forgetting it is not harmless.
    undeclared = _intrinsics_after_crop(600, 380, 90.0)[3]
    declared = _intrinsics_after_crop(600, 380, 90.0, crop_bottom_px=20)[3]

    assert undeclared == 190.0
    assert declared == 200.0


def test_the_horizon_separates_ground_from_sky():
    """Rows below the horizon back-project; rows above it do not."""
    fx, fy, cx, cy = _intrinsics_after_crop(600, 640, 90.0)
    us = np.full(4, 300.0, dtype=np.float32)
    vs = np.array([cy - 50, cy, cy + 50, 639], dtype=np.float32)

    _, z = _ground_intersection(us, vs, fx, fy, cx, cy, pitch_deg=0.0)

    assert not np.isfinite(z[0])  # above the horizon
    assert not np.isfinite(z[1])  # on it
    assert np.isfinite(z[2]) and z[2] > 0  # below
    assert np.isfinite(z[3]) and z[3] < z[2]  # nearer still


# --------------------------------------------------------------------------- #
# the logo-strip crop                                                         #
# --------------------------------------------------------------------------- #
def test_crop_is_a_no_op_when_no_bar_height_is_configured(monkeypatch):
    # Regression: `rgb[:-0]` is `rgb[:0]`, so asking for the crop with the
    # default height of 0 returned an empty image rather than an uncropped one.
    monkeypatch.setattr(image_io._cfg, "google_bar_height_px", 0)
    img = np.zeros((400, 600, 3), np.uint8)

    assert image_io.read_rgb(img, crop_bar=True).shape == (400, 600, 3)


def test_crop_removes_exactly_the_configured_bar(monkeypatch):
    monkeypatch.setattr(image_io._cfg, "google_bar_height_px", 20)
    img = np.zeros((400, 600, 3), np.uint8)

    assert image_io.read_rgb(img, crop_bar=True).shape == (380, 600, 3)
    assert image_io.read_rgb(img, crop_bar=False).shape == (400, 600, 3)


def test_crop_leaves_an_image_shorter_than_the_bar_alone(monkeypatch):
    monkeypatch.setattr(image_io._cfg, "google_bar_height_px", 500)
    img = np.zeros((400, 600, 3), np.uint8)

    assert image_io.read_rgb(img, crop_bar=True).shape == (400, 600, 3)

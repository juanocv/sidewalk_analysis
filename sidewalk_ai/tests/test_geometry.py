import numpy as np

from sidewalk_ai.processing.geometry import compute_width


def test_compute_width_returns_positive_estimate_for_streetview_sized_mask():
    height, width = 400, 600
    depth = np.full((height, width), 5.0, dtype="float32")

    mask = np.zeros((height, width), bool)
    mask[250:380, 100:500] = True

    result = compute_width(
        mask,
        depth,
        fov_deg=60,
        band_frac=(0.0, 1.0),
        bottom_ignore_px=0,
        min_valid_rows=3,
    )

    assert result.width_m > 0.0
    assert result.margin_m > 0.0
    assert result.n_pixels >= 3

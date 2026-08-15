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


def _diverging_scene():
    """Mask plus a depth map far enough from the ground plane to disagree."""
    mask = np.zeros((400, 600), bool)
    mask[300:390, 120:480] = True
    # Ground geometry puts these rows around 3-5 m; a depth map at 12 m makes the
    # depth-assisted width diverge from the geometric one by well over 25%.
    depth = np.full((400, 600), 12.0, dtype="float32")
    depth[:, 300:] = 12.6  # slight gradient so rows pass the parallax gate
    return mask, depth


def test_divergent_depth_hands_the_estimate_back_to_geometry():
    mask, depth = _diverging_scene()
    kwargs = dict(pitch_deg=-10, fov_deg=90, bottom_ignore_px=0, min_valid_rows=3)

    strict = compute_width(mask, depth, divergence_pct=0.25, **kwargs)
    disabled = compute_width(mask, depth, divergence_pct=None, **kwargs)

    # With the gate off the depth path wins, so the two must not agree.
    assert strict.width_m != disabled.width_m
    # Geometry is the conservative one here; depth inflates the width.
    assert strict.width_m < disabled.width_m


def test_agreeing_paths_are_unaffected_by_the_divergence_gate():
    mask = np.zeros((400, 600), bool)
    mask[300:390, 120:480] = True
    depth = np.full((400, 600), 5.0, dtype="float32")
    kwargs = dict(pitch_deg=-10, fov_deg=90, bottom_ignore_px=0, min_valid_rows=3)

    assert (
        compute_width(mask, depth, divergence_pct=0.25, **kwargs).width_m
        == compute_width(mask, depth, divergence_pct=None, **kwargs).width_m
    )


def test_compute_width_survives_the_band_retry_path():
    # Regression: every row rejected by the Δu gate falls through to the retry
    # loop, whose bridge-fill block used to run outside its own `if` and raised
    # UnboundLocalError on `left_end`.
    mask = np.zeros((400, 600), bool)
    mask[300:360, 290:300] = True

    result = compute_width(
        mask,
        None,
        pitch_deg=-10,
        fov_deg=90,
        bottom_ignore_px=0,
        min_valid_rows=3,
        du_range_px=(50, 220),
    )

    assert result.width_m >= 0.0

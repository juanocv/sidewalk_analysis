"""The estimation path must give identical numbers for identical inputs."""

import numpy as np

from sidewalk_ai.io.image_io import _color_for_type
from sidewalk_ai.processing.geometry import compute_width
from sidewalk_ai.processing.refinement import fit_line_ransac, refine_sidewalk_mask

# Δu well above du_hi drives the near-perpendicular correction, whose Δu target
# used to be jittered with the unseeded global RNG. Few rows keep the per-row
# jitter from being averaged away by the median. On the unseeded code this
# spread across 2.82-2.88 m over repeated identical runs.
_WIDTH_KWARGS = dict(pitch_deg=-10, bottom_ignore_px=0, min_valid_rows=3, du_range_px=(20, 250))


def _mask() -> np.ndarray:
    mask = np.zeros((400, 600), bool)
    mask[352:362, 100:500] = True
    return mask


def _depth() -> np.ndarray:
    return np.full((400, 600), 5.0, dtype="float32")


def test_compute_width_is_repeatable():
    mask, depth = _mask(), _depth()

    first = compute_width(mask, depth, **_WIDTH_KWARGS)
    for _ in range(6):
        assert compute_width(mask, depth, **_WIDTH_KWARGS) == first


def test_compute_width_seed_is_honoured():
    mask, depth = _mask(), _depth()

    # Same seed twice must agree...
    assert compute_width(mask, depth, seed=7, **_WIDTH_KWARGS) == compute_width(
        mask, depth, seed=7, **_WIDTH_KWARGS
    )
    # ...and the seed must actually reach the jitter, otherwise this whole
    # module would be asserting nothing.
    seeds = {compute_width(mask, depth, seed=s, **_WIDTH_KWARGS).width_m for s in range(12)}
    assert len(seeds) > 1


def test_ransac_fit_is_repeatable():
    rng = np.random.default_rng(1)
    xs = np.arange(200, dtype=float)
    ys = 0.4 * xs + 12.0 + rng.normal(0, 2.0, xs.size)

    first = fit_line_ransac(xs, ys)
    for _ in range(4):
        assert fit_line_ransac(xs, ys) == first


def test_refine_sidewalk_mask_is_repeatable():
    raw = np.zeros((200, 300), np.uint8)
    raw[80:150, 40:130] = 1
    raw[80:150, 160:260] = 1

    first_mask, first_lines = refine_sidewalk_mask(raw, min_keep_area_px=100)
    for _ in range(4):
        mask, lines = refine_sidewalk_mask(raw, min_keep_area_px=100)
        assert np.array_equal(mask, first_mask)
        assert lines == first_lines


def test_obstacle_colours_are_stable_across_processes():
    # hash() is salted per process; the palette must not depend on it.
    assert _color_for_type("bollard") == _color_for_type("bollard")
    assert _color_for_type("bollard") == (160, 75, 77)

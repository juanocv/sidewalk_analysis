"""Metric-scale recovery for depth back-ends that report relative depth."""

import numpy as np
import pytest

from sidewalk_ai.processing.geometry import (
    _ground_intersection,
    _intrinsics_after_crop,
    estimate_ground_scale,
    to_metric_depth,
)

HEIGHT, WIDTH = 400, 600
FOV_DEG = 90.0
PITCH_DEG = -10.0


def _ground_truth_depth() -> tuple[np.ndarray, np.ndarray]:
    """A sidewalk lying on the ground plane, with its exact metric depth map."""
    fx, fy, cx, cy = _intrinsics_after_crop(WIDTH, HEIGHT, FOV_DEG)

    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[300:400, 80:520] = True

    us, vs = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32))
    _, z = _ground_intersection(us.ravel(), vs.ravel(), fx, fy, cx, cy, PITCH_DEG)
    depth = z.reshape(HEIGHT, WIDTH).astype(np.float32)
    return mask, depth


def _scale_args():
    fx, fy, cx, cy = _intrinsics_after_crop(WIDTH, HEIGHT, FOV_DEG)
    return fx, fy, cx, cy


# --------------------------------------------------------------------------- #
# estimate_ground_scale                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.25, 1.0, 4.0])
def test_ground_scale_recovers_the_missing_factor(alpha):
    mask, metric_depth = _ground_truth_depth()
    relative_depth = metric_depth / alpha  # what a non-metric back-end would emit

    recovered = estimate_ground_scale(mask, relative_depth, *_scale_args())

    assert recovered is not None
    assert recovered == pytest.approx(alpha, rel=0.05)


def test_ground_scale_is_none_without_enough_support():
    _, depth = _ground_truth_depth()
    tiny = np.zeros((HEIGHT, WIDTH), bool)
    tiny[398:400, 300:305] = True  # far fewer pixels than min_inliers

    assert estimate_ground_scale(tiny, depth, *_scale_args()) is None


def test_ground_scale_ignores_non_finite_depth():
    mask, metric_depth = _ground_truth_depth()
    holed = metric_depth.copy()
    holed[-20:, :] = np.nan  # the Google logo strip the pipeline blanks out

    recovered = estimate_ground_scale(mask, holed, *_scale_args())

    assert recovered is not None
    assert recovered == pytest.approx(1.0, rel=0.05)


def test_ground_scale_is_repeatable():
    mask, metric_depth = _ground_truth_depth()
    first = estimate_ground_scale(mask, metric_depth / 2.0, *_scale_args())
    for _ in range(3):
        assert estimate_ground_scale(mask, metric_depth / 2.0, *_scale_args()) == first


# --------------------------------------------------------------------------- #
# to_metric_depth                                                             #
# --------------------------------------------------------------------------- #
def test_to_metric_depth_uses_the_ground_plane():
    mask, metric_depth = _ground_truth_depth()
    relative = metric_depth / 3.0

    scaled, alpha, source = to_metric_depth(relative, mask, fov_deg=FOV_DEG)

    assert source == "ground"
    assert alpha == pytest.approx(3.0, rel=0.05)
    finite = np.isfinite(metric_depth)
    assert scaled[finite] == pytest.approx(metric_depth[finite], rel=0.05)


def test_to_metric_depth_falls_back_when_the_fit_has_no_support():
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    empty = np.zeros((HEIGHT, WIDTH), bool)

    scaled, alpha, source = to_metric_depth(depth, empty, fov_deg=FOV_DEG, fallback_scale=0.075)

    assert source == "fallback"
    assert alpha == pytest.approx(0.075)
    assert scaled == pytest.approx(depth * 0.075)


def test_force_fallback_skips_the_ground_fit():
    mask, metric_depth = _ground_truth_depth()

    _, alpha, source = to_metric_depth(
        metric_depth,
        mask,
        fov_deg=FOV_DEG,
        fallback_scale=0.5,
        force_fallback=True,
    )

    # Without force_fallback this frame would have resolved to alpha ~= 1.0.
    assert source == "fallback"
    assert alpha == pytest.approx(0.5)


def test_no_scale_available_is_reported_rather_than_guessed():
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    empty = np.zeros((HEIGHT, WIDTH), bool)

    scaled, alpha, source = to_metric_depth(depth, empty, fov_deg=FOV_DEG)

    assert source == "none"
    assert alpha is None
    # The map is handed back untouched; the caller must not treat it as metres.
    assert scaled == pytest.approx(depth)


# --------------------------------------------------------------------------- #
# MiDaS disparity conversion                                                  #
# --------------------------------------------------------------------------- #
def test_disparity_is_inverted_into_relative_depth():
    pytest.importorskip("torch")
    from sidewalk_ai.models.midas import _disparity_to_relative_depth

    # Disparity grows towards the bottom of the frame, i.e. the ground gets closer.
    disparity = np.tile(np.linspace(1.0, 20.0, HEIGHT, dtype=np.float32)[:, None], (1, WIDTH))

    relative = _disparity_to_relative_depth(disparity)

    column = relative[:, 0]

    # Larger disparity must map to smaller depth, and never the other way round.
    assert column[-1] < column[0]
    assert np.all(np.diff(column) <= 0)
    assert np.all(relative > 0)

    # The far field saturates at the documented 100x cap rather than blowing up.
    assert relative.max() <= 100.0 + 1e-3
    unsaturated = column[column < 99.0]
    assert np.all(np.diff(unsaturated) < 0)


def test_disparity_conversion_preserves_missing_pixels():
    pytest.importorskip("torch")
    from sidewalk_ai.models.midas import _disparity_to_relative_depth

    disparity = np.linspace(1.0, 20.0, 100, dtype=np.float32).reshape(10, 10)
    disparity[0, :] = np.nan

    relative = _disparity_to_relative_depth(disparity)

    assert np.isnan(relative[0]).all()
    assert np.isfinite(relative[1:]).all()

"""Ensemble fusion: every member votes, and obstacles survive the merge."""

import numpy as np
import pytest

from sidewalk_ai.models.ensemble import EnsembleSegmenter

SHAPE = (40, 60)


class FakeSeg:
    """Minimal Segmenter double with a controllable mask and panoptic map."""

    def __init__(self, mask, seg_map=None, seg_info=None, obstacles=None, arity=4):
        self._mask = mask
        self._seg_map = seg_map
        self._seg_info = seg_info
        self._obstacles = obstacles or []
        self._arity = arity

    def segment(self, img_rgb, target_label="sidewalk", **kw):
        if self._arity == 3:
            return self._mask, self._seg_map, self._seg_info
        return self._mask, self._seg_map, self._seg_info, self._obstacles


def _mask(cols: slice) -> np.ndarray:
    m = np.zeros(SHAPE, bool)
    m[10:30, cols] = True
    return m


def _panoptic():
    seg_map = np.zeros(SHAPE, np.int32)
    seg_map[10:30, 5:55] = 1  # sidewalk
    seg_map[20:30, 20:26] = 2  # tree standing on it
    return seg_map, [(1, "sidewalk"), (2, "tree")]


# --------------------------------------------------------------------------- #
# fusion rules                                                                #
# --------------------------------------------------------------------------- #
def test_or_takes_the_union():
    a, b = _mask(slice(5, 30)), _mask(slice(25, 55))
    fused, _, _, _ = EnsembleSegmenter(FakeSeg(a), FakeSeg(b), method="or").segment(None)

    assert np.array_equal(fused, a | b)


def test_and_takes_the_intersection():
    a, b = _mask(slice(5, 30)), _mask(slice(25, 55))
    fused, _, _, _ = EnsembleSegmenter(FakeSeg(a), FakeSeg(b), method="and").segment(None)

    assert np.array_equal(fused, a & b)


def test_majority_is_accepted_and_needs_three_to_differ_from_and():
    # Regression: the CLI advertised --ensemble-method majority while the
    # ensemble rejected anything but or/and, raising at construction time.
    a, b, c = _mask(slice(5, 30)), _mask(slice(25, 55)), _mask(slice(5, 55))

    two = EnsembleSegmenter(FakeSeg(a), FakeSeg(b), method="majority").segment(None)[0]
    three = EnsembleSegmenter(FakeSeg(a), FakeSeg(b), FakeSeg(c), method="majority").segment(None)[
        0
    ]

    assert np.array_equal(two, a & b)  # degenerate with two members
    assert np.array_equal(three, (a & b) | (a & c) | (b & c))
    assert not np.array_equal(three, a & b & c)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        EnsembleSegmenter(FakeSeg(_mask(slice(5, 30))), FakeSeg(_mask(slice(5, 30))), method="xor")


def test_a_single_member_is_not_an_ensemble():
    with pytest.raises(ValueError, match="at least two"):
        EnsembleSegmenter(FakeSeg(_mask(slice(5, 30))))


# --------------------------------------------------------------------------- #
# panoptic passthrough — the bug that silently zeroed every clearance          #
# --------------------------------------------------------------------------- #
def test_panoptic_map_reaches_the_caller():
    seg_map, seg_info = _panoptic()
    a = FakeSeg(_mask(slice(5, 30)), seg_map, seg_info)
    b = FakeSeg(_mask(slice(25, 55)))

    _, out_map, out_info, _ = EnsembleSegmenter(a, b).segment(None)

    # Previously this returned (fused, None, None, []), so the pipeline skipped
    # obstacle extraction and reported a perfectly clear sidewalk.
    assert out_map is not None
    assert np.array_equal(out_map, seg_map)
    assert out_info == seg_info


def test_panoptic_map_is_taken_from_the_second_member_when_the_first_has_none():
    seg_map, seg_info = _panoptic()
    a = FakeSeg(_mask(slice(5, 30)), arity=3)
    b = FakeSeg(_mask(slice(25, 55)), seg_map, seg_info)

    _, out_map, out_info, _ = EnsembleSegmenter(a, b).segment(None)

    assert np.array_equal(out_map, seg_map)
    assert out_info == seg_info


def test_misaligned_panoptic_map_is_dropped_rather_than_misapplied():
    seg_map = np.ones((20, 30), np.int32)  # half size: ids would not line up
    a = FakeSeg(_mask(slice(5, 30)), seg_map, [(1, "sidewalk")])
    b = FakeSeg(_mask(slice(25, 55)))

    _, out_map, out_info, _ = EnsembleSegmenter(a, b).segment(None)

    assert out_map is None
    assert out_info is None


# --------------------------------------------------------------------------- #
# obstacle fallback when nobody exposes a panoptic map                        #
# --------------------------------------------------------------------------- #
def test_obstacles_are_pooled_when_no_panoptic_map_exists():
    tree = np.zeros(SHAPE, bool)
    tree[20:30, 20:26] = True
    off_sidewalk = np.zeros(SHAPE, bool)
    off_sidewalk[0:5, 0:5] = True

    a = FakeSeg(_mask(slice(5, 30)), obstacles=[("tree", tree)])
    b = FakeSeg(_mask(slice(25, 55)), obstacles=[("kite", off_sidewalk)])

    _, out_map, _, obstacles = EnsembleSegmenter(a, b).segment(None)

    assert out_map is None
    labels = [label for label, _ in obstacles]
    assert labels == ["tree"]  # the one that never touches the sidewalk is dropped


def test_three_tuple_members_are_accepted():
    a = FakeSeg(_mask(slice(5, 30)), arity=3)
    b = FakeSeg(_mask(slice(25, 55)), arity=3)

    fused, _, _, obstacles = EnsembleSegmenter(a, b).segment(None)

    assert fused.any()
    assert obstacles == []

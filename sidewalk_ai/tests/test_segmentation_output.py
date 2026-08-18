"""The segmentation contract.

Back-ends used to return a bare 3- or 4-tuple that callers told apart with
`len(out)`. Falling through that check left every name unbound and surfaced far
away as a NameError, and nothing recorded what the positions meant.
"""

from __future__ import annotations

import numpy as np
import pytest

from sidewalk_ai.models.base import SegmentationOutput


def _mask() -> np.ndarray:
    return np.zeros((4, 6), bool)


def test_obstacles_default_to_empty_rather_than_none():
    # The pipeline iterates this without a None check.
    out = SegmentationOutput(_mask())

    assert out.obstacles == []
    assert out.seg_map is None and out.seg_info is None


def test_each_instance_gets_its_own_obstacle_list():
    first, second = SegmentationOutput(_mask()), SegmentationOutput(_mask())
    first.obstacles.append(("tree", _mask()))

    assert second.obstacles == []


def test_the_type_is_frozen():
    out = SegmentationOutput(_mask())

    with pytest.raises(Exception):
        out.mask = _mask()


# --------------------------------------------------------------------------- #
# coercing what a third-party back-end returns                                #
# --------------------------------------------------------------------------- #
def test_coerce_passes_the_type_through_untouched():
    out = SegmentationOutput(_mask())

    assert SegmentationOutput.coerce(out) is out


def test_a_three_tuple_still_works():
    seg_map, seg_info = np.zeros((4, 6), np.int32), [(1, "sidewalk")]

    out = SegmentationOutput.coerce((_mask(), seg_map, seg_info))

    assert out.seg_info == seg_info
    assert out.obstacles == []


def test_a_four_tuple_still_works():
    obstacles = [("tree", _mask())]

    out = SegmentationOutput.coerce((_mask(), None, None, obstacles))

    assert out.obstacles == obstacles


def test_a_none_obstacle_list_becomes_empty():
    out = SegmentationOutput.coerce((_mask(), None, None, None))

    assert out.obstacles == []


@pytest.mark.parametrize("arity", [1, 2, 5])
def test_a_wrong_arity_is_rejected_at_the_boundary(arity):
    with pytest.raises(TypeError, match="expected a SegmentationOutput"):
        SegmentationOutput.coerce(tuple([_mask()] * arity))


def test_the_error_names_the_offending_back_end():
    # The point of `source`: a bad return value should point at its author.
    with pytest.raises(TypeError, match="MySegmenter.segment"):
        SegmentationOutput.coerce((_mask(), None), source="MySegmenter")


def test_a_non_sequence_is_rejected_without_a_len_crash():
    with pytest.raises(TypeError, match="returned ndarray"):
        SegmentationOutput.coerce(_mask(), source="MySegmenter")

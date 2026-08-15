"""CLI back-end string parsing: '--seg a+b+c' must build every member."""

import numpy as np
import pytest
import sidewalk_ai as sw

from sidewalk_ai.cli._builder import LABEL_MAP, build_segmenter
from sidewalk_ai.models.ensemble import EnsembleSegmenter


class StubSeg:
    def __init__(self, backend, **kwargs):
        self.backend = backend
        self.kwargs = kwargs

    def segment(self, img_rgb, target_label="sidewalk", **kw):
        return np.zeros((8, 8), bool), None, None, []


@pytest.fixture
def stub_backends(monkeypatch):
    """Replace the model factory so no weights are loaded."""
    built = []

    def fake_build_segmenter(backend, **kwargs):
        built.append((backend, kwargs))
        return StubSeg(backend, **kwargs)

    monkeypatch.setattr(sw, "build_segmenter", fake_build_segmenter, raising=False)
    return built


def _build(seg_flag, stub_backends, **overrides):
    kwargs = dict(ckpt=None, dl_model="deeplabv3plus_resnet101", device="cpu", method="or")
    kwargs.update(overrides)
    return build_segmenter(seg_flag, **kwargs)


def test_single_backend_is_not_wrapped_in_an_ensemble(stub_backends):
    alias = _build("oneformer", stub_backends)

    assert isinstance(alias.base, StubSeg)
    assert [name for name, _ in stub_backends] == ["oneformer"]


def test_every_listed_backend_takes_part(stub_backends):
    # Regression: only backends[0] and backends[1] were built, so the third
    # back-end in '--seg a+b+c' was silently ignored.
    alias = _build("oneformer+detectron2+deeplab", stub_backends, ckpt="weights.pth")

    assert isinstance(alias.base, EnsembleSegmenter)
    assert len(alias.base.members) == 3
    assert [name for name, _ in stub_backends] == ["oneformer", "detectron2", "deeplab"]


def test_deeplab_gets_its_checkpoint_inside_an_ensemble(stub_backends):
    _build("oneformer+deeplab", stub_backends, ckpt="weights.pth")

    deeplab_kwargs = dict(stub_backends)["deeplab"]
    assert deeplab_kwargs["ckpt_path"] == "weights.pth"


def test_deeplab_without_a_checkpoint_fails_in_an_ensemble_too(stub_backends):
    # Used to raise a bare KeyError('ckpt_path') from deep inside the factory.
    with pytest.raises(ValueError, match="--ckpt is required"):
        _build("oneformer+deeplab", stub_backends, ckpt=None)


def test_unknown_backend_is_named_in_the_error(stub_backends):
    with pytest.raises(ValueError, match="Unknown segmentation back-end 'onefromer'"):
        _build("onefromer", stub_backends)


def test_majority_reaches_the_ensemble(stub_backends):
    alias = _build("oneformer+detectron2+deeplab", stub_backends, ckpt="w.pth", method="majority")

    assert alias.base.method == "majority"


def test_synonyms_are_deterministic_and_deduplicated(stub_backends):
    first = _build("oneformer+detectron2", stub_backends).syn
    second = _build("oneformer+detectron2", stub_backends).syn

    assert first == second
    assert len(first) == len(set(first))
    # Order follows the back-ends as listed, not set iteration order.
    assert first[: len(LABEL_MAP["oneformer"])] == LABEL_MAP["oneformer"]

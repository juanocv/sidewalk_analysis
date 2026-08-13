import cv2
import numpy as np

from sidewalk_ai.core.pipeline import DepthScale, Result, SidewalkPipeline
from sidewalk_ai.io.streetview import Settings, StreetViewClient


class DummySeg:
    def segment(self, img, target_label="sidewalk", **kw):
        height, width = img.shape[:2]
        mask = np.zeros((height, width), bool)
        mask[250:380, 100:500] = True
        return mask, None, None


class DummyDepth:
    is_metric = True

    def predict(self, img):
        return np.full(img.shape[:2], 5.0, dtype="float32")


class DummyRelativeDepth:
    """A MiDaS-like back-end: plausible shape, arbitrary unit."""

    is_metric = False

    def predict(self, img):
        return np.full(img.shape[:2], 5.0, dtype="float32")


def _pipeline(tmp_path, **kwargs) -> SidewalkPipeline:
    return SidewalkPipeline(
        segmenter=DummySeg(),
        streetview=StreetViewClient(
            cache_dir=tmp_path / "streetview-cache",
            settings=Settings(google_api_key="test-key"),
        ),
        **kwargs,
    )


def _stub_streetview(tmp_path, monkeypatch):
    img = np.zeros((400, 600, 3), np.uint8)
    local = tmp_path / "dummy.jpg"
    cv2.imwrite(str(local), img)

    monkeypatch.setattr(StreetViewClient, "fetch", lambda self, _addr: local)
    monkeypatch.setattr(StreetViewClient, "geocode", lambda self, _address: (-23.0, -46.0))
    return local


def test_pipeline_no_gpu(tmp_path, monkeypatch):
    _stub_streetview(tmp_path, monkeypatch)

    pipe = _pipeline(tmp_path, depth=DummyDepth())
    res: Result = pipe.analyse_address("any address")
    assert res.width.width_m > 0


def test_refine_false_keeps_the_raw_mask_downstream(tmp_path, monkeypatch):
    _stub_streetview(tmp_path, monkeypatch)

    pipe = _pipeline(tmp_path, depth=DummyDepth(), refine=False)
    res: Result = pipe.analyse_address("any address")

    assert np.array_equal(res.refined_mask.astype(bool), res.sidewalk_mask.astype(bool))


def test_refine_true_actually_changes_the_mask(tmp_path, monkeypatch):
    _stub_streetview(tmp_path, monkeypatch)

    refined = _pipeline(tmp_path, depth=DummyDepth(), refine=True).analyse_address("a")
    raw = _pipeline(tmp_path, depth=DummyDepth(), refine=False).analyse_address("a")

    # If these matched, the refine flag would still be doing nothing.
    assert not np.array_equal(refined.refined_mask.astype(bool), raw.refined_mask.astype(bool))


def test_relative_depth_without_a_fallback_falls_back_to_geometry(tmp_path, monkeypatch):
    _stub_streetview(tmp_path, monkeypatch)

    # A flat relative map gives the ground-plane fit nothing to lock onto, and no
    # fallback scale is configured, so the depth path is dropped for this frame
    # rather than reported in an arbitrary unit.
    pipe = _pipeline(tmp_path, depth=DummyRelativeDepth())
    res: Result = pipe.analyse_address("any address")

    assert res.width.width_m > 0


def test_relative_depth_uses_the_configured_fallback_scale(tmp_path, monkeypatch):
    _stub_streetview(tmp_path, monkeypatch)

    pipe = _pipeline(
        tmp_path,
        depth=DummyRelativeDepth(),
        depth_scale=DepthScale(fallback_scale=0.5, force_fallback=True),
    )
    res: Result = pipe.analyse_address("any address")

    assert res.width.width_m > 0

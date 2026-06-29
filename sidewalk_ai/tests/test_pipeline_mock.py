import cv2
import numpy as np

from sidewalk_ai.core.pipeline import Result, SidewalkPipeline
from sidewalk_ai.io.streetview import Settings, StreetViewClient


class DummySeg:
    def segment(self, img, target_label="sidewalk", **kw):
        height, width = img.shape[:2]
        mask = np.zeros((height, width), bool)
        mask[250:380, 100:500] = True
        return mask, None, None


class DummyDepth:
    def predict(self, img):
        return np.full(img.shape[:2], 5.0, dtype="float32")


def test_pipeline_no_gpu(tmp_path, monkeypatch):
    # monkey-patch StreetView to avoid network
    img = np.zeros((400, 600, 3), np.uint8)
    local = tmp_path / "dummy.jpg"
    cv2.imwrite(str(local), img)

    def fake_fetch(self, _addr):
        return local

    def fake_geocode(self, _address):
        return -23.0, -46.0

    monkeypatch.setattr(StreetViewClient, "fetch", fake_fetch)
    monkeypatch.setattr(StreetViewClient, "geocode", fake_geocode)

    pipe = SidewalkPipeline(
        segmenter=DummySeg(),
        depth=DummyDepth(),
        streetview=StreetViewClient(
            cache_dir=tmp_path / "streetview-cache",
            settings=Settings(google_api_key="test-key"),
        ),
    )
    res: Result = pipe.analyse_address("any address")
    assert res.width.width_m > 0

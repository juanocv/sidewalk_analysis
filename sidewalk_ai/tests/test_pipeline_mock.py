# tests/test_pipeline_mock.py
import numpy as np
from pathlib import Path
from sidewalk_ai.core.pipeline import SidewalkPipeline, Result
from sidewalk_ai.io.streetview import StreetViewClient

class DummySeg:
    def segment(self, img, target_label="sidewalk", **kw):
        H, W = img.shape[:2]
        mask = np.zeros((H, W), bool)
        mask[:, 10:-10] = True      # fake sidewalk
        return mask, None, None

class DummyDepth:
    def predict(self, img):
        return np.full(img.shape[:2], 5.0, dtype="float32")

def test_pipeline_no_gpu(tmp_path, monkeypatch):
    # monkey-patch StreetView to avoid network
    img = np.zeros((120,160,3), np.uint8)
    local = tmp_path / "dummy.jpg"
    import cv2; cv2.imwrite(str(local), img)

    def fake_fetch(self, _addr):
        return local
    monkeypatch.setattr(StreetViewClient, "fetch", fake_fetch)

    pipe = SidewalkPipeline(
        segmenter=DummySeg(),
        depth=DummyDepth(),
        streetview=StreetViewClient(),
    )
    res: Result = pipe.analyse_address("any address")
    assert res.width.width_m > 0
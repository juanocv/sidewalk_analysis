import numpy as np, sidewalk_ai as sw

def test_zoe_depth():
    est = sw.build_depth("zoe", device="cpu")
    dummy = np.zeros((224,224,3), np.uint8) + 127
    out   = est.predict(dummy)
    assert out.shape == dummy.shape[:2]
    assert out.dtype == np.float32
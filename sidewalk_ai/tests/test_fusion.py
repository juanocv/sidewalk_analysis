import numpy as np
from sidewalk_ai.processing.fusion import logical_fuse

def test_majority_vote():
    a = np.zeros((4, 4), bool)
    b = np.ones((4, 4),  bool)
    c = np.ones((4, 4),  bool)
    fused = logical_fuse([a, b, c], method="majority")
    assert fused.all()
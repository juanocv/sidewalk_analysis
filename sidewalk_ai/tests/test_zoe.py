"""ZoeDepth smoke test.

Downloads model weights on first run, so it is marked `gpu` and `network` and
stays out of the default suite. Run it explicitly with:

    python -m pytest -m gpu
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.network]


@pytest.fixture(scope="module")
def zoe_estimator():
    pytest.importorskip("torch")
    import sidewalk_ai as sw

    return sw.build_depth("zoe", device="cpu")


def test_zoe_depth(zoe_estimator):
    dummy = np.zeros((224, 224, 3), np.uint8) + 127

    out = zoe_estimator.predict(dummy)

    assert out.shape == dummy.shape[:2]
    assert out.dtype == np.float32


def test_zoe_reports_metric_depth(zoe_estimator):
    # The pipeline skips scale recovery for back-ends that claim metres.
    assert zoe_estimator.is_metric is True

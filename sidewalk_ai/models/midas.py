# sidewalk_ai/models/midas.py
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from sidewalk_ai.log import get_logger

logger = get_logger(__name__)


class MidasEstimator:
    """
    Thin wrapper around any MiDaS checkpoint that fulfils the
    ``DepthEstimator`` protocol declared in ``sidewalk_ai/models/base.py``.
    A single instance keeps the weights on GPU for the whole lifetime
    of your process (FastAPI / CLI), so inference is *much* faster than
    loading MiDaS every call as the old implementation did.
    """

    _HUB_REPO = "intel-isl/MiDaS"
    _DEFAULT_MODEL = "DPT_BEiT_L_512"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        trust_repo: bool = True,
        output_is_disparity: bool = True,
    ) -> None:
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.model_name = model_name or self._DEFAULT_MODEL

        logger.info("Loading MiDaS model '%s' on device '%s'", self.model_name, self.device)
        # ↓ torch already caches the weights under ~/.cache/torch/hub
        self._model = torch.hub.load(self._HUB_REPO, self.model_name, trust_repo=trust_repo).to(
            self.device
        )
        logger.info("Loaded MiDaS model '%s'", self.model_name)
        self._model.eval()
        logger.info("MiDaS model moved to device '%s'", self.device)

        # Fixed Imagenet stats expected by all MiDaS variants
        self._pre = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # MiDaS emits inverse relative depth (disparity-like: larger = closer).
        # Set to False only for a checkpoint that already returns depth.
        self.output_is_disparity = output_is_disparity

        # depth values are *relative* → we must still fit a scale afterwards
        self.is_metric = False

    # --------------------------------------------------------------------- #
    # public API required by the DepthEstimator Protocol
    # --------------------------------------------------------------------- #

    @torch.inference_mode()
    def predict(self, img: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Parameters
        ----------
        img
            H×W×3 **uint8** array *or* a path/str to an image file.

        Returns
        -------
        depth : np.ndarray[float32]   (same H×W as the input)
            Relative **depth** — larger means further away — normalised so the
            nearest surfaces sit near 1.0.  Still needs
            :func:`~sidewalk_ai.processing.geometry.to_metric_depth` before it
            can be read as metres.
        """
        if isinstance(img, (str, Path)):
            pil = Image.open(img).convert("RGB")
            np_img = np.asarray(pil)
        else:
            np_img = img
            pil = Image.fromarray(np_img)

        batch = self._pre(pil).unsqueeze(0).to(self.device)

        depth = self._model(batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(np_img.shape[0], np_img.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()  # H × W

        out = depth.cpu().numpy().astype("float32")
        if not self.output_is_disparity:
            return out
        return _disparity_to_relative_depth(out)


def _disparity_to_relative_depth(disparity: np.ndarray) -> np.ndarray:
    """
    Turn MiDaS' affine-invariant inverse depth into a relative depth map.

    The raw output is disparity-like (larger = closer) with an unknown offset,
    so it is first shifted by its 1st percentile to make the far field
    approach zero, then inverted.  The dynamic range is normalised so the
    nearest surfaces land near 1.0: without that, the metres-per-unit constant
    behind ``--fallback-scale`` would mean something different on every frame.

    The far field is clamped to 100× the near distance to keep the inversion
    from producing infinities where the disparity flattens out.
    """
    disp = np.asarray(disparity, dtype=np.float32)
    finite = np.isfinite(disp)
    if not finite.any():
        return np.full(disp.shape, np.nan, dtype=np.float32)

    lo = float(np.percentile(disp[finite], 1))
    hi = float(np.percentile(disp[finite], 99))
    span = max(hi - lo, 1e-6)

    shifted = np.clip(disp - lo, 0.01 * span, None)
    relative = span / shifted  # ≈1 at the near end, ≤100 at the far end
    relative[~finite] = np.nan
    return relative.astype(np.float32)

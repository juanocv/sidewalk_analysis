# sidewalk_ai/models/midas.py
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class MidasEstimator:
    """
    Thin wrapper around any MiDaS checkpoint that fulfils the
    ``DepthEstimator`` protocol declared in ``sidewalk_ai/models/base.py``.
    A single instance keeps the weights on GPU for the whole lifetime
    of your process (FastAPI / CLI), so inference is *much* faster than
    loading MiDaS every call as the old implementation did. :contentReference[oaicite:0]{index=0}
    """

    _HUB_REPO = "intel-isl/MiDaS"
    _DEFAULT_MODEL = "DPT_Large"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cuda",
        trust_repo: bool = True,
    ) -> None:
        self.device = (
            torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model_name = model_name or self._DEFAULT_MODEL

        # ↓ torch already caches the weights under ~/.cache/torch/hub
        self._model = torch.hub.load(
            self._HUB_REPO, self.model_name, trust_repo=trust_repo
        ).to(self.device)
        self._model.eval()

        # Fixed Imagenet stats expected by all MiDaS variants
        self._pre = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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

        return depth.cpu().numpy().astype("float32")
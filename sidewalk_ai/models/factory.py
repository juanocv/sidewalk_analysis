# sidewalk_ai/models/factory.py
from __future__ import annotations
from typing import Any, Literal
# ─── segmentation back-ends ────────────────────────────────────────
from .detectron2 import Detectron2Segmenter
from .oneformer  import OneFormerSegmenter
from .deeplab    import DeepLabSegmenter, load_deeplab_checkpoint
# ─── depth back-ends ───────────────────────────────────────────────
from .midas import MidasEstimator
from .zoe   import ZoeDepthEstimator          

# ------------------------------------------------------------------ #
#  SEGMENTER  FACTORY                                                #
# ------------------------------------------------------------------ #
def build_segmenter(
    backend: Literal["oneformer", "detectron2", "deeplab"] = "oneformer",
    **kwargs: Any,
):
    if backend == "oneformer":
        return OneFormerSegmenter(**kwargs)
    if backend == "detectron2":
        return Detectron2Segmenter.from_zoo(**kwargs)
    if backend == "deeplab":
        ckpt = kwargs.pop("ckpt_path")

        # ── kwargs meant for the *loader* ───────────────────────────
        loader_keys = {"model_name", "num_classes",
                       "output_stride", "allow_pickle"}
        loader_kwargs = {k: kwargs.pop(k) for k in loader_keys if k in kwargs}

        dl = load_deeplab_checkpoint(ckpt, **loader_kwargs)

        # remaining kwargs (e.g. sidewalk_class_id, device) go to Segmenter
        return DeepLabSegmenter(dl, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")

# ------------------------------------------------------------------ #
#  DEPTH  FACTORY                                                    #
# ------------------------------------------------------------------ #
def build_depth(
    backend: Literal["midas", "zoe"] = "midas",
    variant: str | None = None,
    **kwargs: Any,
):
    """
    Returns a depth-estimator instance with a `.predict(np.uint8 H×W×3)` method
    compatible with the rest of the pipeline.
    
    Parameters
    ----------
    backend : str
        The depth estimation backend to use ("midas" or "zoe")
    variant : str, optional
        For ZoeDepth: "zoed_n", "zoed_k", "zoed_nk"
        For MiDaS: model variant if supported
    **kwargs
        Additional arguments passed to the depth estimator
    """
    if backend == "midas":
        return MidasEstimator(**kwargs)

    if backend == "zoe":
        # Pass variant to ZoeDepth if provided
        if variant is not None:
            kwargs['variant'] = variant
        return ZoeDepthEstimator(**kwargs)

    raise ValueError(f"Unknown depth backend: {backend}")
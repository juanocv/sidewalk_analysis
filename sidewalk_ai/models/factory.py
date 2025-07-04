# sidewalk_ai/models/factory.py
from __future__ import annotations
from typing import Any, Literal
from .detectron2 import Detectron2Segmenter
from .oneformer  import OneFormerSegmenter
from .deeplab    import DeepLabSegmenter, load_deeplab_checkpoint

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
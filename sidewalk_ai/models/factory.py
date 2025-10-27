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

# Cache for model instances
_MODEL_CACHE = {}

def _get_cache_key(model_type: str, **kwargs) -> str:
    """Generate a unique cache key based on model type and parameters."""
    key_parts = [model_type]
    for k, v in sorted(kwargs.items()):
        if k not in ['device']:  # Exclude device from cache key for consistency
            key_parts.append(f"{k}={v}")
    return ":".join(key_parts)

def clear_model_cache():
    """Clear the model cache (useful for memory management)."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()

# ------------------------------------------------------------------ #
#  SEGMENTER  FACTORY                                                #
# ------------------------------------------------------------------ #
def build_segmenter(
    backend: Literal["oneformer", "detectron2", "deeplab"] = "oneformer",
    use_cache: bool = True,
    **kwargs: Any,
):
    
    if use_cache:
        cache_key = _get_cache_key(f"segmenter:{backend}", **kwargs)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    
    if backend == "oneformer":
        instance = OneFormerSegmenter(**kwargs)
    elif backend == "detectron2":
        instance = Detectron2Segmenter.from_zoo(**kwargs)
    elif backend == "deeplab":
        ckpt = kwargs.pop("ckpt_path")

        # ── kwargs meant for the *loader* ───────────────────────────
        loader_keys = {"model_name", "num_classes",
                       "output_stride", "allow_pickle"}
        loader_kwargs = {k: kwargs.pop(k) for k in loader_keys if k in kwargs}

        dl = load_deeplab_checkpoint(ckpt, **loader_kwargs)

        # remaining kwargs (e.g. sidewalk_class_id, device) go to Segmenter
        instance = DeepLabSegmenter(dl, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    if use_cache:
        _MODEL_CACHE[cache_key] = instance
    
    return instance

# ------------------------------------------------------------------ #
#  DEPTH  FACTORY                                                    #
# ------------------------------------------------------------------ #
def build_depth(
    backend: Literal["midas", "zoe"] = "midas",
    variant: str | None = None,
    use_cache: bool = True,
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
# Build cache key (exclude device for consistent caching)
    cache_kwargs = kwargs.copy()
    cache_kwargs.pop('device', None)
    if variant is not None:
        cache_kwargs['variant'] = variant
        
    if use_cache:
        cache_key = _get_cache_key(f"depth:{backend}", **cache_kwargs)
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]
    
    if backend == "midas":
        instance = MidasEstimator(**kwargs)
    elif backend == "zoe":
        if variant is not None:
            kwargs['variant'] = variant
        instance = ZoeDepthEstimator(**kwargs)
    else:
        raise ValueError(f"Unknown depth backend: {backend}")
    
    if use_cache:
        _MODEL_CACHE[cache_key] = instance
    
    return instance
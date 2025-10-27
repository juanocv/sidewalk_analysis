# sidewalk_ai/core/pipeline_manager.py
from __future__ import annotations
from typing import Optional
from .pipeline import SidewalkPipeline
from sidewalk_ai.models.factory import build_segmenter, build_depth, clear_model_cache
from sidewalk_ai.io.streetview import StreetViewClient

class PipelineManager:
    """
    Manages pipeline instances with model caching for optimal performance.
    """
    _instance: Optional[PipelineManager] = None
    _pipelines: dict = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_pipeline(
        self,
        segmenter_backend: str = "oneformer",
        depth_backend: str = "midas",
        depth_variant: str = "zoed_n",
        segmenter_kwargs: dict | None = None,
        depth_kwargs: dict | None = None,
        streetview_kwargs: dict | None = None,
        refine: bool = True,
        fuse_method: str | None = None,
    ) -> SidewalkPipeline:
        """
        Get a pipeline with the given configuration. Returns cached instance if available.
        """
        # Create a cache key based on the configuration
        cache_key = (
            segmenter_backend,
            depth_backend,
            depth_variant,
            refine,
            fuse_method,
            str(sorted((segmenter_kwargs or {}).items())),
            str(sorted((depth_kwargs or {}).items())),
        )
        
        if cache_key in self._pipelines:
            return self._pipelines[cache_key]
        
        # Build new pipeline with caching enabled
        segmenter_kwargs = segmenter_kwargs or {}
        depth_kwargs = depth_kwargs or {}
        streetview_kwargs = streetview_kwargs or {}
        
        segmenter = build_segmenter(
            backend=segmenter_backend, 
            use_cache=True,
            **segmenter_kwargs
        )
        
        depth_estimator = build_depth(
            backend=depth_backend,
            variant=depth_variant,
            use_cache=True,
            **depth_kwargs
        )
        
        streetview = StreetViewClient(**streetview_kwargs)
        
        pipeline = SidewalkPipeline(
            segmenter=segmenter,
            depth=depth_estimator,
            streetview=streetview,
            refine=refine,
            fuse_method=fuse_method,
        )
        
        self._pipelines[cache_key] = pipeline
        return pipeline
    
    def clear_cache(self):
        """Clear pipeline and model caches."""
        clear_model_cache()
        self._pipelines.clear()

# Global instance for easy access
pipeline_manager = PipelineManager()
from __future__ import annotations

from typing import Any, Literal


def _optional_import_error(component: str, install_hint: str, exc: ModuleNotFoundError) -> None:
    missing = exc.name or "unknown"
    raise ModuleNotFoundError(
        f"{component} requires optional dependencies that are not installed. "
        f"Missing module: {missing}. {install_hint}"
    ) from exc


def build_segmenter(
    backend: Literal["oneformer", "detectron2", "deeplab"] = "oneformer",
    **kwargs: Any,
):
    if backend == "oneformer":
        try:
            from .oneformer import OneFormerSegmenter
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The OneFormer segmentation backend",
                'Install the ML extra with `python -m pip install -e ".[ml]"`.',
                exc,
            )
        return OneFormerSegmenter(**kwargs)

    if backend == "detectron2":
        try:
            from .detectron2 import Detectron2Segmenter
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The Detectron2 segmentation backend",
                "Install Detectron2 using its upstream Windows/CUDA instructions.",
                exc,
            )
        return Detectron2Segmenter.from_zoo(**kwargs)

    if backend == "deeplab":
        try:
            from .deeplab import DeepLabSegmenter, load_deeplab_checkpoint
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The DeepLab segmentation backend",
                "Install the ML extra and ensure the local DeepLab `network` package is importable.",
                exc,
            )

        ckpt = kwargs.pop("ckpt_path")
        loader_keys = {"model_name", "num_classes", "output_stride", "allow_pickle"}
        loader_kwargs = {k: kwargs.pop(k) for k in loader_keys if k in kwargs}
        dl = load_deeplab_checkpoint(ckpt, **loader_kwargs)
        return DeepLabSegmenter(dl, **kwargs)

    raise ValueError(f"Unknown backend: {backend}")


def build_depth(
    backend: Literal["midas", "zoe"] = "midas",
    variant: str | None = None,
    **kwargs: Any,
):
    """
    Returns a depth-estimator instance with a `.predict(np.uint8 HxWx3)` method
    compatible with the rest of the pipeline.
    """
    if backend == "midas":
        try:
            from .midas import MidasEstimator
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The MiDaS depth backend",
                'Install the ML extra with `python -m pip install -e ".[ml]"`.',
                exc,
            )
        return MidasEstimator(**kwargs)

    if backend == "zoe":
        try:
            from .zoe import ZoeDepthEstimator
        except ModuleNotFoundError as exc:
            _optional_import_error(
                "The ZoeDepth backend",
                'Install the ML extra with `python -m pip install -e ".[ml]"` and make '
                "the ZoeDepth package available according to `docs/reproducibility.md`.",
                exc,
            )

        if variant is not None:
            kwargs["variant"] = variant
        return ZoeDepthEstimator(**kwargs)

    raise ValueError(f"Unknown depth backend: {backend}")

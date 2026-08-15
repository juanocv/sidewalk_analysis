from __future__ import annotations

from pathlib import Path

import sidewalk_ai as sw
from sidewalk_ai.models.base import Segmenter
from sidewalk_ai.models.ensemble import EnsembleSegmenter
from sidewalk_ai.log import get_logger

logger = get_logger(__name__)

# hard-coded synonyms per back-end
LABEL_MAP = {
    "oneformer": ["sidewalk", "path"],
    "detectron2": ["sidewalk", "pavement", "path", "footpath"],
    "deeplab": ["sidewalk"],  # City-scapes trainId 1
}


class AliasSegmenter(Segmenter):
    def __init__(self, backend_name: str, base: Segmenter, synonyms: list[str]):
        self.base = base
        self.syn = synonyms
        self.backend_name = backend_name

    def segment(self, img_rgb, target_label="sidewalk", **kw):
        result = self.base.segment(img_rgb, target_label=self.syn, **kw)
        return result


# ───────────────────── Build the base segmenter(s) ──────────────────
def _build_one(backend: str, *, ckpt: str | None, dl_model: str | None, device: str) -> Segmenter:
    """Build a single back-end, applying its own construction requirements."""
    if backend not in LABEL_MAP:
        raise ValueError(
            f"Unknown segmentation back-end {backend!r}; "
            f"choose from {', '.join(sorted(LABEL_MAP))}"
        )
    if backend == "deeplab":
        # Required in an ensemble too, not just when deeplab runs alone.
        if ckpt is None:
            raise ValueError("--ckpt is required for Deeplab")

        # An explicit --deeplab-model always wins; otherwise read the
        # architecture off the checkpoint name, which upstream encodes there.
        # A fixed default silently loaded mobilenet weights into resnet101.
        model_name = dl_model
        if model_name is None:
            from sidewalk_ai.models.deeplab import infer_model_name

            model_name = infer_model_name(ckpt)
            if model_name is None:
                raise ValueError(
                    f"Could not infer the DeepLab architecture from {Path(ckpt).name!r}. "
                    "Pass --deeplab-model explicitly (e.g. deeplabv3plus_mobilenet)."
                )
            logger.info("DeepLab architecture inferred from the checkpoint name: %s", model_name)

        return sw.build_segmenter(
            "deeplab",
            ckpt_path=ckpt,
            model_name=model_name,
            allow_pickle=True,
            device=device,
        )
    return sw.build_segmenter(backend, device=device)


def build_segmenter(
    seg_flag: str, *, ckpt: str | None, dl_model: str | None, device: str, method: str | None
) -> Segmenter:
    """Build a segmenter based on the given flag and parameters."""
    backends = [name.strip() for name in seg_flag.split("+") if name.strip()]
    if not backends:
        raise ValueError("--seg must name at least one back-end")

    members = [_build_one(b, ckpt=ckpt, dl_model=dl_model, device=device) for b in backends]

    # dict.fromkeys keeps insertion order; a set comprehension would order the
    # synonyms differently on every process (PYTHONHASHSEED).
    synonyms = list(dict.fromkeys(s for b in backends for s in LABEL_MAP[b]))

    # Every back-end listed takes part, not just the first two.
    base_seg = (
        members[0] if len(members) == 1 else EnsembleSegmenter(*members, method=method or "or")
    )

    # Wrap the base segmenter with an alias segmenter
    return AliasSegmenter(backend_name=seg_flag, base=base_seg, synonyms=synonyms)

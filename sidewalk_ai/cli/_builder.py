from __future__ import annotations
import sidewalk_ai as sw
from sidewalk_ai.models.base import Segmenter
from sidewalk_ai.models.ensemble import EnsembleSegmenter

# hard-coded synonyms per back-end
LABEL_MAP = {
    "oneformer":   ["sidewalk", "path"],
    "detectron2":  ["sidewalk", "pavement", "path", "footpath"],
    "deeplab":     ["sidewalk"],          # City-scapes trainId 1
}

class AliasSegmenter(Segmenter):            
    def __init__(self, backend_name: str, base: Segmenter, synonyms: list[str]):
        self.base = base
        self.syn  = synonyms
        self.backend_name = backend_name

    def segment(self, img_rgb, target_label="sidewalk", **kw):
        result = self.base.segment(img_rgb, target_label=self.syn, **kw)
        return result

# ───────────────────── Build the base segmenter(s) ──────────────────
def build_segmenter(seg_flag: str, *, ckpt: str|None,
                    dl_model: str, device: str, method: str|None) -> Segmenter:
    """Build a segmenter based on the given flag and parameters."""
    backends = seg_flag.split("+")
    if len(backends) == 1:
        synonyms = LABEL_MAP[backends[0]]  # <-- Always set synonyms here
        if backends[0] == "deeplab":
            if ckpt is None:
                raise ValueError("--ckpt is required for Deeplab")
            base_seg = sw.build_segmenter(
                "deeplab",
                ckpt_path=ckpt,
                model_name=dl_model,
                allow_pickle=True,
                device=device,
            )
        else:
            base_seg = sw.build_segmenter(backends[0])
    else:
        # Two back-ends, build an ensemble segmenter
        base_seg = EnsembleSegmenter(
            seg1=sw.build_segmenter(backends[0], device=device),
            seg2=sw.build_segmenter(backends[1], device=device),
            method=method,  # default ensemble method
        )
        # Combine synonyms from both backends, removing duplicates
        synonyms = list({s for b in backends for s in LABEL_MAP[b]})

    # Wrap the base segmenter with an alias segmenter
    return AliasSegmenter(
        backend_name=seg_flag,
        base=base_seg,
        synonyms=synonyms
    )   


    def get_class_labels(self):
        """Get class labels mapping for the current backend"""
        if hasattr(self.base, 'get_class_labels'):
            return self.base.get_class_labels()
        
        # Fallback: try to get from common attributes
        if hasattr(self.base, 'id2label'):
            return self.base.id2label
        elif hasattr(self.base, 'config') and hasattr(self.base.config, 'id2label'):
            return self.base.config.id2label
        
        return None
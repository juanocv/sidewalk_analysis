# sidewalk_ai/models/detectron2.py
from __future__ import annotations
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from .base import Segmenter, SegmentInfo


class Detectron2Segmenter(Segmenter):
    def __init__(
        self,
        config_yml: str,
        weights_path: str,
        *,
        score_thresh: float = 0.4,
        device: str | None = None,
    ):
        cfg = get_cfg()
        cfg.merge_from_file(config_yml)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self._meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

        # id  ➜  readable label  (needed by legend & mask-matching)
        stuff = {i: n for i, n in enumerate(self._meta.stuff_classes)}
        thing = {i: n for i, n in enumerate(self._meta.thing_classes)}
        self.id2lbl = {**stuff, **thing}

    @torch.inference_mode()
    def segment(self, img_rgb, target_label="sidewalk", *, device=None):
        outs = self.predictor(img_rgb)
        seg_map, seg_info_raw = outs["panoptic_seg"]
        seg_map = seg_map.cpu().numpy()

        sidewalk = np.zeros_like(seg_map, dtype=bool)
        seg_info: list[SegmentInfo] = []

        for seg in seg_info_raw:
            cat_id = seg["category_id"]
            is_thing = seg["isthing"]
            name = (
                self._meta.thing_classes[cat_id]
                if is_thing
                else self._meta.stuff_classes[cat_id]
            )

            seg_info.append((int(seg["id"]), name))
            if _match(name, target_label):
                sidewalk |= seg_map == seg["id"]

        return sidewalk, seg_map, seg_info
    
    # ------------------------------------------------------------------ #
    # Convenience ctor – mirrors old  initialize_model(model_path)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_zoo(
        cls,
        cfg_name: str = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml",
        *,
        score_thresh: float = 0.4,
        device: str | None = None,
    ) -> "Detectron2Segmenter":
        """
        Build a Detectron2Segmenter from a **model-zoo config string**
        exactly like the legacy `initialize_model()` helper did.
        """
        from detectron2 import model_zoo
        cfg_file = model_zoo.get_config_file(cfg_name)
        weights  = model_zoo.get_checkpoint_url(cfg_name)

        return cls(
            config_yml=cfg_file,
            weights_path=weights,
            score_thresh=score_thresh,
            device=device,
        )


def _match(name: str, target) -> bool:
    if isinstance(target, str):
        target = [target]
    return any(name.lower() == t.lower() for t in target)
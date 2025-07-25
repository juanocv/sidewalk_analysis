# sidewalk_ai/models/oneformer.py
from __future__ import annotations
import numpy as np
import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

from .base import Segmenter, SegmentInfo


class OneFormerSegmenter(Segmenter):
    def __init__(
        self,
        model_name: str = "shi-labs/oneformer_ade20k_swin_large",
        *,
        device: str | None = None,
    ):
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).train()

    @torch.inference_mode()
    def segment(self, img_rgb, target_label="sidewalk", *, device=None):
        from PIL import Image

        pil = Image.fromarray(img_rgb)
        inputs = self.processor(
            images=pil, task_inputs=["panoptic"], return_tensors="pt"
        ).to(self.device)

        outs = self.model(**inputs)
        res = self.processor.post_process_panoptic_segmentation(
            outs,
            target_sizes=[img_rgb.shape[:2]],
            threshold=0.2,
            mask_threshold=0.5,
            overlap_mask_area_threshold=0.8,
            label_ids_to_fuse=set(range(200)),
        )[0]

        seg_map = res["segmentation"].cpu().numpy()
        seg_info_raw = res["segments_info"]
        id2lbl = self.model.config.id2label

        sidewalk = np.zeros_like(seg_map, dtype=bool)
        seg_info: list[SegmentInfo] = []

        for seg in seg_info_raw:
            name = id2lbl[seg["label_id"]]
            seg_info.append((int(seg["id"]), name))
            if _match(name, target_label):
                sidewalk |= seg_map == seg["id"]

        return sidewalk, seg_map, seg_info


def _match(name: str, target) -> bool:
    if isinstance(target, str):
        target = [target]
    return any(name.lower().startswith(t.lower()) for t in target)
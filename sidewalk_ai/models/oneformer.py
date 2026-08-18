# sidewalk_ai/models/oneformer.py
from __future__ import annotations
from typing import Literal
import numpy as np
import torch
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

from sidewalk_ai.processing.refinement import shave_above_top_envelope
from .base import SegmentationOutput, Segmenter, SegmentInfo


class OneFormerSegmenter(Segmenter):
    backend_name = "oneformer"

    def __init__(
        self,
        model_name: str = "shi-labs/oneformer_ade20k_swin_large",
        *,
        device: str | None = None,
        fuse_stuff: bool = False,  # funde *apenas* stuff quando True
        postproc_threshold: float = 0.20,
        mask_threshold: float = 0.50,
        overlap_mask_area_threshold: float = 0.80,
        seg_task: Literal["panoptic", "instance", "semantic"] = "panoptic",
        dual_pass_for_instances: bool = True,  # panoptic p/ calçada + instance p/ obstáculos
    ):
        # 1) inicializações corretas
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # 2) configs de pós-processamento
        self.fuse_stuff = fuse_stuff
        self._postproc_threshold = postproc_threshold
        self._mask_threshold = mask_threshold
        self._overlap_mask_area_threshold = overlap_mask_area_threshold
        self._stuff_ids = set()  # popular com ids de stuff se/ quando quiser
        self.seg_task = seg_task
        self.dual_pass_for_instances = dual_pass_for_instances

        # 3) buffers p/ debug
        self.last_segments_info = None
        self.last_seg_map = None

    @torch.inference_mode()
    def segment(self, img_rgb, target_label="sidewalk", *, device=None):
        from PIL import Image

        pil = Image.fromarray(img_rgb)

        id2lbl = getattr(self.model.config, "id2label", {})
        H, W = img_rgb.shape[:2]

        # ------------------ Passo A: Panoptic (sempre) ------------------
        pan_inputs = self.processor(images=pil, task_inputs=["panoptic"], return_tensors="pt")
        pan_inputs = {k: v.to(self.device) for k, v in pan_inputs.items()}
        pan_outs = self.model(**pan_inputs)

        fuse_ids = self._stuff_ids if self.fuse_stuff else set()
        pan = self.processor.post_process_panoptic_segmentation(
            pan_outs,
            target_sizes=[(H, W)],
            threshold=self._postproc_threshold,
            mask_threshold=self._mask_threshold,
            overlap_mask_area_threshold=self._overlap_mask_area_threshold,
            label_ids_to_fuse=fuse_ids,
        )[0]

        seg_map = pan["segmentation"].cpu().numpy()
        seg_info_raw = pan["segments_info"]

        seg_info: list[SegmentInfo] = []
        sidewalk_raw = np.zeros_like(seg_map, dtype=bool)

        def _match(name: str, target) -> bool:
            if isinstance(target, str):
                target = [target]
            return any(name.lower().startswith(t.lower()) for t in target)

        for seg in seg_info_raw:
            lbl_id = seg.get("label_id", seg.get("category_id"))
            name = id2lbl.get(lbl_id, str(lbl_id))
            seg_id = int(seg["id"])
            seg_info.append((seg_id, name))
            if _match(name, target_label):
                sidewalk_raw |= seg_map == seg_id

        # guardar para debug_viz
        self.last_segments_info = seg_info_raw
        self.last_seg_map = seg_map

        # ------------------ Passo B: Obstáculos ------------------
        import cv2

        def _split_components(bin_mask: np.ndarray, min_area: int = 30) -> list[np.ndarray]:
            m = bin_mask.astype(np.uint8)
            n, lab = cv2.connectedComponents(m, connectivity=8)
            comps = []
            for cid in range(1, n):
                c = lab == cid
                if int(c.sum()) >= min_area:
                    comps.append(c)
            return comps

        obstacles: list[tuple[str, np.ndarray]] = []

        # (B1) PANOPTIC → instâncias por componente (cobre 'tree' e outros *stuff*)
        for seg_id, seg_name in seg_info:
            if _match(seg_name, target_label):
                continue
            inst_mask = seg_map == seg_id
            # sep. por componentes para evitar "duas árvores virarem 1"
            for j, comp in enumerate(_split_components(inst_mask, min_area=30)):
                # manter apenas se tocar a calçada
                if int((comp & sidewalk_raw).sum()) == 0:
                    continue
                obstacles.append((f"{seg_name}#{seg_id}:{j}", comp))

        # (B2) INSTANCE (opcional) → adicionar *things* úteis que a panóptica não captou bem
        if self.dual_pass_for_instances or self.seg_task == "instance":
            ins_inputs = self.processor(images=pil, task_inputs=["instance"], return_tensors="pt")
            ins_inputs = {k: v.to(self.device) for k, v in ins_inputs.items()}
            ins_outs = self.model(**ins_inputs)
            ins = self.processor.post_process_instance_segmentation(
                ins_outs,
                target_sizes=[(H, W)],
                threshold=self._postproc_threshold,
                mask_threshold=self._mask_threshold,
            )[0]

            ins_map = ins["segmentation"].cpu().numpy().astype(np.int32)
            ins_info = ins["segments_info"]

            # whitelist simples de things que costumam ser obstáculos
            THINGS_ALLOW = {
                "pole",
                "bollard",
                "trash",
                "trash bin",
                "bench",
                "traffic light",
                "traffic sign",
            }

            # deduplicação grosseira por IoU para não duplicar com panóptica
            def _iou(a, b):
                inter = int((a & b).sum())
                denom = int(a.sum() + b.sum() - inter) or 1
                return inter / denom

            for seg in ins_info:
                seg_i = int(seg["id"])
                cls_id = int(seg["label_id"])
                name = id2lbl.get(cls_id, str(cls_id)).lower().strip()

                # só considerar classes úteis; evite 'fence', 'rail' etc. aqui
                if name not in THINGS_ALLOW:
                    continue

                mask_i = ins_map == seg_i
                # só se tocar a calçada
                if int((mask_i & sidewalk_raw).sum()) == 0:
                    continue

                # não adicionar se já houver máscara bem parecida vinda da panóptica
                dup = False
                for _, m_exist in obstacles:
                    if _iou(mask_i, m_exist) > 0.6:
                        dup = True
                        break
                if not dup:
                    # separar componentes (garante 1 instância por blob)
                    for j, comp in enumerate(_split_components(mask_i, min_area=30)):
                        if int((comp & sidewalk_raw).sum()) == 0:
                            continue
                        obstacles.append((f"{name}#ins{seg_i}:{j}", comp))

        # ------------------ Passo C: Refinamento ------------------
        mask = shave_above_top_envelope(
            sidewalk_raw.astype(np.uint8),
            max_above_px=None,
            smooth_kernel=11,
            min_cols=30,
        ).astype(bool)

        return SegmentationOutput(mask, seg_map, seg_info, obstacles)

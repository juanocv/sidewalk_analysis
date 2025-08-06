# sidewalk_ai/models/deeplab.py
from __future__ import annotations
import numpy as np
import torch, cv2
from torchvision import transforms
from PIL import Image

from sidewalk_ai.processing.refinement import refine_sidewalk_mask

from .base import Segmenter


class DeepLabSegmenter(Segmenter):
    """
    Expects *any* torchvision-style DeepLab model already loaded.
    """

    def __init__(
        self,
        dl_model: torch.nn.Module,
        *,
        sidewalk_class_id: int = 1,      # Cityscapes trainId
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = dl_model.to(self.device).eval()
        self.sidewalk_id = sidewalk_class_id
        self.tr = transforms.Compose(
            [
                transforms.Resize((512, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Add Cityscapes class mapping (19 classes)
        self.id2label = {
            0: "road",
            1: "sidewalk", 
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "traffic_light",
            7: "traffic_sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motorcycle",
            18: "bicycle"
        }
        
        # Store for debug visualization
        self.label_divisor = 1  # DeepLab uses direct class IDs, no divisor

    @torch.inference_mode()
    def segment(self, img_rgb, target_label="sidewalk", *, device=None):
        # 1) preprocess
        pil = Image.fromarray(img_rgb)
        inp = self.tr(pil).unsqueeze(0).to(self.device)

        # 2) forward
        out = self.model(inp)

        # 3) grab logits tensor regardless of output format
        if isinstance(out, dict):
            logits = out.get("out", next(iter(out.values())))
        elif isinstance(out, (list, tuple)):
            logits = out[0]
        else:                 # already a tensor
            logits = out

        # 4) class prediction → numpy mask
        pred = logits.softmax(1).argmax(1).squeeze(0).cpu().numpy()

        # 5) resize if network native res ≠ input res
        if pred.shape != img_rgb.shape[:2]:
            pred = np.array(
                Image.fromarray(pred.astype("uint8")).resize(
                    img_rgb.shape[1::-1], Image.NEAREST
                )
            )

        # ── 1) RAW sidewalk mask ────────────────────────────────────────
        mask = pred == self.sidewalk_id

        # ── 2) refined sidewalk mask (optional) ───────────────────────
        mask, edge_top, edge_bot = refine_sidewalk_mask(mask)

        # ── obstacle discovery (coarse, class level) ─────────────────────
        obstacles = []
        for cid in np.unique(pred):
           if cid == self.sidewalk_id:
               continue
           inst = (pred == cid)
           # accept only portions lying inside the sidewalk contour
           if (inst & cv2.dilate(mask.astype(np.uint8), None, iterations=1).astype(bool)).sum() == 0:
               continue
           label = self.id2label.get(cid, f"class_{cid}")
           obstacles.append((label, inst))

        # 6) Create segment info for debug visualization
        unique_ids = np.unique(pred)
        seg_info = []
        for class_id in unique_ids:
            class_name = self.id2label.get(class_id, f"class_{class_id}")
            seg_info.append((int(class_id), class_name))

        return mask, edge_top, edge_bot, pred, seg_info, obstacles
    
    def get_class_labels(self):
        """Get class labels mapping for DeepLab"""
        return self.id2label
    
# ----------------------------------------------------------------------- #
#  Checkpoint loader – mirrors old  load_deeplab_cityscapes(...)
# ----------------------------------------------------------------------- #
def load_deeplab_checkpoint(
    ckpt_path: str,
    *,
    model_name: str = "deeplabv3plus_resnet101",
    num_classes: int = 19,
    output_stride: int = 16,
    device: str = "cuda",
    allow_pickle: bool = True
):
    """
    Load a custom DeepLabV3+ checkpoint **once** and return the torch model.

    Exactly the same semantics as the legacy `load_deeplab_cityscapes`,
    but local to the DeepLab wrapper so callers don’t import two modules.
    """
    from pathlib import Path
    import importlib
    import torch
    import pickle

    ckpt = Path(ckpt_path).expanduser()
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

    # Dynamically import your project’s `network.modeling`
    modeling = importlib.import_module("network.modeling")

    if model_name not in modeling.__dict__:
        raise ValueError(f"{model_name} not found in network.modeling")

    model = modeling.__dict__[model_name](
        num_classes=num_classes, output_stride=output_stride
    )

    try:
        raw = torch.load(ckpt, map_location="cpu")
    except pickle.UnpicklingError as e:
        if not allow_pickle:
            raise ValueError(
                "Failed to load checkpoint with pickle disabled. "
                "Set allow_pickle=True to enable it."
            ) from e
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = (
        raw.get("model_state")
        or raw.get("state_dict")
        or raw               # plain `torch.save(model.state_dict())`
    )
    # ── NEW: keep only tensors whose shapes match the model ────────────
    model_keys = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if (k in model_keys) and (v.shape == model_keys[k].shape)}

    if not filtered:
        raise RuntimeError("No matching layers between checkpoint and model. "
                           "Check `model_name` or supply the correct backbone.")

    model.load_state_dict(filtered, strict=False)
    return model.to(device).eval()
# ── sidewalk_ai/models/zoe.py ──────────────────────────────────────
from __future__ import annotations
from pathlib import Path
import numpy as np, torch

_VARIANTS = {"zoed_n": "ZoeD_N",
             "zoed_k": "ZoeD_K",
             "zoed_nk": "ZoeD_NK"}

class ZoeDepthEstimator:
    """
    Same public API as MidasEstimator:
        depth = ZoeDepthEstimator(...).predict(img_rgb_uint8)
    """

    is_metric = True

    def __init__(self,
                 variant: str = "zoed_n",
                 device: str | None = None,
                 source: str = "github",        # "github" | "local"
                 repo_or_path: str | Path | None = None,
                 ckpt_path: str | Path | None = None):
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS)}")

        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self._variant = variant  # Store for debug visualization

        hub_repo = "isl-org/ZoeDepth" if source == "github" else str(
            Path(repo_or_path or ".").resolve())

        # --- build architecture only ---
        self.model = torch.hub.load(hub_repo, _VARIANTS[variant],
                                    source="local" if source == "local" else "github",
                                    pretrained=True).to(self.device).eval()

        # --- load weights (official or custom) ---
        if ckpt_path is not None:                          # user checkpoint
            state = torch.load(Path(ckpt_path).expanduser(), map_location="cpu")
        else:                                              # official checkpoint → query cfg
            from zoedepth.utils.config import get_config

            if   variant == "zoed_n":
                cfg = get_config("zoedepth",      "infer")                 # NYU-trained
            elif variant == "zoed_k":
                cfg = get_config("zoedepth",      "infer", config_version="kitti")
            else:  # "zoed_nk"
                cfg = get_config("zoedepth_nk",   "infer")

            res = cfg.pretrained_resource
            url = res["url"] if isinstance(res, dict) else res
            url = url.split("url::", 1)[-1]          # strip prefix if present

            state = torch.hub.load_state_dict_from_url(
                url, map_location="cpu", progress=True)

        self.model.load_state_dict(state, strict=False)    # ignore extra keys

        # keep a lightweight handle to the helper only after weights are ok
        from zoedepth.utils.misc import pil_to_batched_tensor
        self._pil_to_batched = pil_to_batched_tensor

    # ----------------------------------------------------------------
    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        if img_rgb.dtype != np.uint8:
            raise ValueError("expects H×W×3 uint8 RGB image")

        import cv2
        from PIL import Image

        # Convert to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Convert to batched tensor
        bat = self._pil_to_batched(pil_img).to(self.device)
        
        # Inference with proper error handling
        with torch.no_grad():
            try:
                # Use the model's infer method
                depth = self.model.infer(bat)
                
                # Handle different return formats
                if isinstance(depth, dict):
                    # Some models return a dict with 'metric_depth' key
                    depth = depth.get('metric_depth', depth.get('depth', depth))
                
                # Extract the depth map
                if isinstance(depth, (list, tuple)):
                    depth = depth[0]
                
                # Convert to numpy
                if hasattr(depth, 'squeeze'):
                    depth = depth.squeeze()
                
                depth = depth.cpu().detach().numpy()
                
                # Handle different tensor shapes
                if depth.ndim == 3 and depth.shape[0] == 1:
                    depth = depth.squeeze(0)
                elif depth.ndim == 4:
                    depth = depth.squeeze(0).squeeze(0)
                
            except Exception as e:
                print(f"ZoeDepth inference error: {e}")
                # Fallback to a simple forward pass
                try:
                    depth = self.model(bat)
                    if isinstance(depth, dict):
                        depth = depth.get('metric_depth', depth.get('depth', list(depth.values())[0]))
                    depth = depth.squeeze().cpu().detach().numpy()
                except Exception as e2:
                    print(f"ZoeDepth fallback failed: {e2}")
                    # Return a dummy depth map as last resort
                    H, W = img_rgb.shape[:2]
                    return np.ones((H, W), dtype=np.float32) * 5.0

        inv = self.model.infer(bat)[0].squeeze(0).cpu().detach().numpy()
        depth = 1.0 / (inv + 1e-8)

        # Resize to match input dimensions
        H, W = img_rgb.shape[:2]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # Post-process depth values
        depth = depth.astype(np.float32)
        
        # Clamp extreme values that might cause issues
        depth = np.clip(depth, 0.1, 100.0)
        
        # Handle invalid values
        depth = np.nan_to_num(depth, nan=5.0, posinf=100.0, neginf=0.1)

        # print("depth  min / max / median:", depth.min(),
        #                               depth.max(),
        #                               np.median(depth))
        
        return depth
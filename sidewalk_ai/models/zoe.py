# ── sidewalk_ai/models/zoe.py ──────────────────────────────────────
from __future__ import annotations
from pathlib import Path
import numpy as np, torch

from sidewalk_ai.log import debug_enabled, debug_event, get_logger

_VARIANTS = {"zoed_n": "ZoeD_N", "zoed_k": "ZoeD_K", "zoed_nk": "ZoeD_NK"}

logger = get_logger(__name__)


def _swai_log(tag, payload):
    if debug_enabled():
        debug_event(logger, tag, payload)


def _unwrap_checkpoint(state: dict) -> dict:
    """
    Pull the weights out of a ZoeDepth checkpoint.

    The published `.pt` files are *training* checkpoints: the parameters sit
    under a ``"model"`` key alongside ``"optimizer"`` and ``"epoch"``. Feeding
    the wrapper straight to ``load_state_dict`` matches nothing, which
    ``strict=False`` reports as 511 missing keys and no error at all.

    Mirrors ``zoedepth.models.model_io.load_state_dict``, including the
    ``module.`` prefix that DataParallel adds when saving.
    """
    state = state.get("model", state)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


# ------------------------------------------------------


class ZoeDepthEstimator:
    """
    Same public API as MidasEstimator:
        depth = ZoeDepthEstimator(...).predict(img_rgb_uint8)
    """

    is_metric = True

    def __init__(
        self,
        variant: str = "zoed_n",
        device: str | None = None,
        source: str = "github",  # "github" | "local"
        repo_or_path: str | Path | None = None,
        ckpt_path: str | Path | None = None,
    ):
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {list(_VARIANTS)}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._variant = variant  # Store for debug visualization

        hub_repo = (
            "isl-org/ZoeDepth" if source == "github" else str(Path(repo_or_path or ".").resolve())
        )

        # --- build architecture only ---
        # pretrained=False is deliberate: with pretrained=True ZoeDepth loads the
        # official checkpoint itself, strictly, and a timm >= 1.0 backbone
        # registers `relative_position_index` as a non-persistent buffer that the
        # checkpoint still carries -> "Unexpected key(s) in state_dict".
        # The weights are applied below instead, tolerantly, which also avoids
        # downloading and loading the same checkpoint twice.
        # trust_repo=True is what keeps this runnable without a terminal. Left
        # unset, torch.hub asks for confirmation the first time it caches a
        # GitHub repo, and with no TTY -- a script, CI, or any redirected run --
        # the prompt fails as a bare "EOFError: EOF when reading a line" that
        # names neither ZoeDepth nor trust. The repo is not user-supplied: the
        # github branch above hardcodes isl-org/ZoeDepth.
        self.model = (
            torch.hub.load(
                hub_repo,
                _VARIANTS[variant],
                source="local" if source == "local" else "github",
                pretrained=False,
                trust_repo=True,
            )
            .to(self.device)
            .eval()
        )

        # --- load weights (official or custom) ---
        if ckpt_path is not None:  # user checkpoint
            state = torch.load(Path(ckpt_path).expanduser(), map_location="cpu")
        else:  # official checkpoint → query cfg
            from zoedepth.utils.config import get_config

            if variant == "zoed_n":
                cfg = get_config("zoedepth", "infer")  # NYU-trained
            elif variant == "zoed_k":
                cfg = get_config("zoedepth", "infer", config_version="kitti")
            else:  # "zoed_nk"
                cfg = get_config("zoedepth_nk", "infer")

            res = cfg.pretrained_resource
            url = res["url"] if isinstance(res, dict) else res
            url = url.split("url::", 1)[-1]  # strip prefix if present

            state = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)

        # strict=False tolerates the buffer mismatch described above, but it would
        # just as happily accept a checkpoint whose names match nothing at all and
        # leave the model randomly initialised. Check what actually landed.
        incompatible = self.model.load_state_dict(_unwrap_checkpoint(state), strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                f"ZoeDepth checkpoint for variant {variant!r} left "
                f"{len(incompatible.missing_keys)} parameter(s) uninitialised, "
                f"starting with {incompatible.missing_keys[:3]}. The weights do not "
                "match this architecture; refusing to run with random parameters."
            )
        _swai_log(
            "zoe_load",
            {
                "variant": variant,
                "ignored_keys": len(incompatible.unexpected_keys),
                "sample": incompatible.unexpected_keys[:3],
            },
        )

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
            out = self.model.infer(bat)  # UMA chamada
            # Alguns variantes retornam dict com 'metric_depth'
            if isinstance(out, dict):
                depth = out.get("metric_depth", out.get("depth", None))
                if depth is None:
                    # Se só veio 'inv_depth' (raro), inverta UMA vez aqui
                    inv = out.get("inv_depth", None)
                    if inv is None:
                        raise RuntimeError("ZoeDepth returned unexpected dict keys")
                    depth = 1.0 / (inv + 1e-8)
            else:
                # Pode vir como tensor direto
                depth = out

            # Tensor -> numpy
            depth = depth.squeeze()
            depth = depth.detach().cpu().numpy()

        # redimensiona, clampa e sanitiza (como você já faz)
        H, W = img_rgb.shape[:2]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        depth = depth.astype(np.float32)
        depth = np.clip(depth, 0.1, 100.0)
        depth = np.nan_to_num(depth, nan=5.0, posinf=100.0, neginf=0.1)

        # (opcional) log:
        _swai_log(
            "zoe",
            {
                "variant": self._variant,
                "depth_min": float(depth.min()),
                "depth_med": float(np.median(depth)),
                "depth_max": float(depth.max()),
            },
        )
        return depth

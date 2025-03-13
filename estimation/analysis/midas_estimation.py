import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

from estimation import analysis
from analysis import *

def estimate_width_m(image_path, predictor, cfg):
    # Obtain sidewalk_mask, cfg and img.read
    sidewalk_mask, img, panoptic_seg, segments_info = analysis.segment_sidewalk_mask(image_path, predictor, cfg)

    # Initialize MiDaS for depth estimation
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(cfg.MODEL.DEVICE)
    midas.eval()

    # 3. Estimate depth
    img_pil = Image.open(image_path).convert("RGB")
    original_size = img_pil.size

    # MiDaS preprocessing
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_batch = transform(img_pil).unsqueeze(0).to(cfg.MODEL.DEVICE)

    with torch.no_grad():
        depth_map = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=original_size[::-1],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    
    # 4. Calculate metrics
    focal_length = img.shape[1] / (2 * np.tan(np.radians(70/2)))  # FOV from Street View API

    height, width = sidewalk_mask.shape
    bottom_frac = 0.8  # keep the bottom 20%
    y_start = int(height * bottom_frac)

    # Restrict to just that slice
    mask_bottom = sidewalk_mask[y_start:, :]
    if not np.any(mask_bottom):
        return None  # or some fallback

    # Step 2) Locate sidewalk pixels
    y_coords, x_coords = np.where(mask_bottom)
    if len(x_coords) == 0:
        return None  # or return some fallback
        
    # Step 3) Get their depth
    z_values = depth_map[y_start + y_coords, x_coords]

    # Step 4) Convert to real-world X, ignoring camera Y dimension for width
    x_real = (x_coords - width/2) * z_values / focal_length

    # Step 5) Sort and remove outliers
    x_sorted = np.sort(x_real)
    low_idx = int(0.02 * len(x_sorted))
    high_idx = int(0.98 * len(x_sorted))
    trimmed = x_sorted[low_idx:high_idx]

    width = (trimmed[-1] - trimmed[0]) / 10  # final width measure

    # Step 6) Estimate error (e.g. 25% or based on std dev)
    margin = (0.25 * width) / 10
    
    result = width, margin, sidewalk_mask, panoptic_seg, segments_info

    if result:
        width, margin, sidewalk_mask, panoptic_seg, segments_info = result
        analysis.midas_visualize(img, cfg, panoptic_seg, segments_info)
        return width, margin
    else:
        raise ValueError("No sidewalk detected!")

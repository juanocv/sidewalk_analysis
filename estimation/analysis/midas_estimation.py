import numpy as np
import torch
from torchvision import transforms as T

from estimation import analysis
from analysis import *
from utils import *

from PIL import Image

def largest_dense_cluster(array, gap_threshold=0.2):
    clusters = []
    current_cluster = [array[0]]

    for i in range(1, len(array)):
        if array[i] - array[i-1] <= gap_threshold:
            current_cluster.append(array[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [array[i]]

    if current_cluster:
        clusters.append(current_cluster)

    print("Formed clusters:")
    for idx, c in enumerate(clusters):
        print(f"Cluster {idx}: {len(c)} points, range={c[-1]-c[0]:.3f}")

    largest_cluster = max(clusters, key=len)
    return np.array(largest_cluster)

def estimate_width_m(
    image_path,
    backend=None,  # or "detectron2" or "oneformer"
    detectron_predictor=None,
    detectron_cfg=None,
    detectron_label_name=None,
    oneformer_model_name=None,
    oneformer_label_name=None,
    device="cuda"
):
    # Open the image (for both backends)
    img_rgb = read_rgbimg(image_path)

    # Obtain sidewalk_mask, cfg and img.read
    sidewalk_mask, panoptic_seg, segments_info, detectron_cfg = segment_sidewalk_mask(
        img_rgb,
        backend=backend,
        detectron_predictor=detectron_predictor,
        detectron_cfg=detectron_cfg,
        detectron_label_name=detectron_label_name,
        oneformer_model_name=oneformer_model_name,
        oneformer_label_name=oneformer_label_name,
        device=device
    )

    # Initialize MiDaS for depth estimation
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device)
    midas.eval()

    # Estimate depth
    # img_pil = Image.open(image_path).convert("RGB")
    #original_size = img.size

    # MiDaS preprocessing
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_pil = Image.fromarray(img_rgb)
    input_batch = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_map = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(image_pil.height, image_pil.width),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    
    # Calculate metrics
    focal_length = img_rgb.shape[1] / (2 * np.tan(np.radians(75/2)))  # FOV from Street View API

    height, width = sidewalk_mask.shape
    bottom_frac = 0.8  # keep the bottom 20%
    y_start = int(height * bottom_frac)

    # Restrict to just that slice
    mask_bottom = sidewalk_mask[y_start:, :]
    if not np.any(mask_bottom):
        return None  # or some fallback

    # Locate sidewalk pixels
    y_coords, x_coords = np.where(mask_bottom)
    if len(x_coords) == 0:
        return None  # or return some fallback
        
    # Get their depth
    z_values = depth_map[y_start + y_coords, x_coords]

    # Convert to real-world X, ignoring camera Y dimension for width
    x_real = (x_coords - width/2) * z_values / focal_length

    # Sort and remove outliers
    x_sorted = np.sort(x_real)
    n = len(x_sorted)
    low_idx  = int(0.02 * n)
    high_idx = int(0.98 * n)
    trimmed  = x_sorted[low_idx:high_idx]

    stable_points = largest_dense_cluster(trimmed, gap_threshold=0.2)
    if len(stable_points) < 2:
        return 0.0  # or None

    computed_width = (stable_points[-1] - stable_points[0])

    final_width = computed_width * 0.1 # applying manual scale factor

    # Estimate error (e.g. 25% or based on std dev)
    margin = (0.25 * final_width)
    
    result = final_width, margin, sidewalk_mask, panoptic_seg, segments_info

    if result:
        print(oneformer_model_name)
        analysis.midas_visualize(img_rgb, panoptic_seg, segments_info, backend, oneformer_model_name, detectron_cfg)
        return final_width, margin
    else:
        raise ValueError("No sidewalk detected!")

def get_depth_map(img_rgb, device="cuda"):
        # Initialize MiDaS for depth estimation
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    midas.to(device)
    midas.eval()

    # MiDaS preprocessing
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_pil = Image.fromarray(img_rgb)
    input_batch = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_map = midas(input_batch)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(image_pil.height, image_pil.width),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    
    return depth_map

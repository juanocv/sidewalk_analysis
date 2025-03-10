import cv2
import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from PIL import Image

# Global variables
model_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"

# Initialize Detectron2 for panoptic segmentation
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_path))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.3
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

# Initialize MiDaS for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(cfg.MODEL.DEVICE)
midas.eval()

image_path = "generic/images/streetview_test9.jpg"
img = cv2.imread(image_path)

def analyze_sidewalk():
    # 1. Load image
    if img is None:
        print(f"Error: Failed to load image from {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Detect sidewalk using panoptic segmentation
    outputs = predictor(img_rgb)
    panoptic_seg, segments_info = outputs["panoptic_seg"]
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Convert to CPU numpy
    panoptic_array = panoptic_seg.cpu().numpy()
    sidewalk_mask = np.zeros_like(panoptic_array, dtype=np.uint8)

    # Find sidewalk segments (COCO class for sidewalk)
    for seg in segments_info:
        cat_id = seg["category_id"]
        if seg["isthing"]:
            # thing class
            cat_name = metadata.thing_classes[cat_id]
        else:
            # stuff class
            cat_name = metadata.stuff_classes[cat_id]

        # Compare to "pavement"
        if cat_name == "pavement":
            mask_area = (panoptic_array == seg["id"])
            sidewalk_mask[mask_area] = 1  # mark sidewalk pixels
    
    # 3. Estimate depth
    img_pil = Image.open(image_path).convert("RGB")
    original_size = img_pil.size

    # MiDaS preprocessing
    transform = T.Compose([
        T.Resize(384),  # MiDaS_small expects 384x384
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
    focal_length = img.shape[1] / (2 * np.tan(np.radians(120/2)))  # FOV from Street View API
    y_coords, x_coords = np.where(sidewalk_mask)
    
    if len(y_coords) == 0:
        return None
        
    z_values = depth_map[y_coords, x_coords]
    x_real = (x_coords - img.shape[1]/2) * z_values / focal_length
    width = np.ptp(x_real)  # Peak-to-peak width
    
    return width, sidewalk_mask, depth_map, panoptic_seg, segments_info

# Example usage
result = analyze_sidewalk()

if result:
    width, mask, depth, panoptic_seg, segments_info = result
    print(f"Estimated sidewalk width: {width:.2f} meters")
    
    # Visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    v = Visualizer(
        img_rgb,
        metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=1.0,
        instance_mode=ColorMode.IMAGE
    )
    
    # Draw instances and return image
    out = v.draw_panoptic_seg_predictions(panoptic_seg.cpu(), segments_info)
    plt.imshow(out.get_image())
    plt.show()
else:
    print("No sidewalk detected")
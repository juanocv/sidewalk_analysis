import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def initialize_model(model_path):
    # Initialize Detectron2's model for panoptic segmentation
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    return predictor, cfg

def segment_sidewalk_mask(image_path, predictor, cfg):
    """
    1) Initialize panoptic model for panoptic segmentation
    2) Loads the image
    3) Runs panoptic on it
    4) Returns a binary mask (H,W) with 1 for sidewalk/pavement, else 0 and image
    """

    # Loads the image
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Failed to load {image_path}")
    img = img[:400-20,:] # resize img to crop out google's logo which may interfere
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 'imread' loads img as BGR so it must be converted to RGB
    
    # Detect sidewalk using panoptic segmentation
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
            sidewalk_mask[mask_area] = 1  # mark sidewalk pixels'

    #print(sidewalk_mask.shape)

    # Return sidewalk mask
    return sidewalk_mask, img, panoptic_seg, segments_info


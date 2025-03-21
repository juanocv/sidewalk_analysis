import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

import numpy as np

def initialize_model(model_path):
    # Initialize Detectron2's model for panoptic segmentation
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    return predictor, cfg
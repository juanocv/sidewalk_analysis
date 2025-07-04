import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def initialize_model(model_path):
    # Initialize Detectron2's model for panoptic segmentation
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    return predictor, cfg

def load_deeplab_cityscapes(ckpt_path, model_name=None, num_classes=19, 
                           output_stride=16, device="cuda"):
    """
    Carrega checkpoint do DeepLabV3+ para Cityscapes.
    """
    import torch
    from network import modeling
    
    # Verificar se o arquivo existe
    import os
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    
    # Construir modelo
    if model_name not in modeling.__dict__:
        raise ValueError(f"{model_name} not found in network.modeling")

    model = modeling.__dict__[model_name](
        num_classes=num_classes,
        output_stride=output_stride
    )
    
    # Carregar checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Diferentes formatos de checkpoint
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Carregar state dict
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded DeepLab model from {ckpt_path}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    
    return model.to(device).eval()
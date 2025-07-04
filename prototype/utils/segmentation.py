import numpy as np
import torch
from pathlib import Path
from PIL import Image

from estimation.analysis.visualization import show_or_save_sidewalk_mask
from utils.image_processing import refine_sidewalk_mask

def segment_with_detectron2(img_rgb, predictor, cfg, label_name=None):
    from detectron2.data import MetadataCatalog

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
        if cat_name == label_name:
            mask_area = (panoptic_array == seg["id"])
            sidewalk_mask[mask_area] = 1  # mark sidewalk pixels'

    #print(sidewalk_mask.shape)

    # Return sidewalk mask
    return sidewalk_mask, segments_info, panoptic_seg

def segment_with_oneformer(
    img_rgb, model, processor, device=None, label_name=None
    ):

    image_pil = Image.fromarray(img_rgb)

    inputs = processor(images=image_pil, task_inputs=["panoptic"], return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs, 
        target_sizes=[(img_rgb.shape[0], img_rgb.shape[1])], 
        threshold=0.2,
        mask_threshold=0.5,
        overlap_mask_area_threshold=0.8,
        label_ids_to_fuse=set(range(200))
    )[0]

    seg_map = result["segmentation"].cpu().numpy()
    seg_info = result["segments_info"]

    sidewalk_mask = np.zeros_like(seg_map, dtype=np.uint8)
    for seg in seg_info:
        label_id = seg["label_id"]
        class_name = model.config.id2label[label_id]
        #print(label_id, class_name)
        if class_name.lower() == label_name.lower():
            sidewalk_mask[seg_map == seg["id"]] = 1

    panoptic_seg = {
        "segmentation": seg_map  # or you could store the entire "result" if you want
    }

    return sidewalk_mask, panoptic_seg, seg_info

def segment_with_deeplab(img_rgb, dl_model, device="cuda",
                         sidewalk_class_id=1):   # Cityscapes trainId para sidewalk
    """
    Returns binary sidewalk mask (H×W uint8).
    """
    import torch
    import numpy as np
    from torchvision import transforms
    from PIL import Image

    # Verificar se o input é PIL Image ou numpy array
    if isinstance(img_rgb, np.ndarray):
        img_pil = Image.fromarray(img_rgb)
    else:
        img_pil = img_rgb
    
    # Preprocessing adequado para Cityscapes
    dl_trans = transforms.Compose([
        transforms.Resize((512, 1024)),  # Tamanho padrão do Cityscapes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dl_model.eval()
    with torch.no_grad():
        inp = dl_trans(img_pil).unsqueeze(0).to(device)
        
        # Verificar se o modelo retorna um dicionário ou tensor direto
        out = dl_model(inp)
        
        # Alguns modelos DeepLab retornam dict com 'out' como chave
        if isinstance(out, dict):
            out = out['out']
        
        # Aplicar softmax se necessário
        if out.dim() == 4 and out.size(1) > 1:  # [B, C, H, W]
            pred = torch.softmax(out, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
        else:
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
    
    # Redimensionar para o tamanho original
    if pred.shape != (img_pil.height, img_pil.width):
        pred_pil = Image.fromarray(pred.astype(np.uint8))
        pred_pil = pred_pil.resize((img_pil.width, img_pil.height), Image.NEAREST)
        pred = np.array(pred_pil)
    
    # Criar máscara binária para sidewalk
    mask = (pred == sidewalk_class_id).astype(np.uint8)
    
    # DEBUG: Verificar se a máscara contém pixels de sidewalk
    print(f"DeepLab mask - Unique values: {np.unique(pred)}")
    print(f"DeepLab mask - Sidewalk pixels: {np.sum(mask)}")
    
    return mask

def segment_sidewalk_mask(img_rgb, backend=None, detectron_predictor=None, detectron_cfg=None, 
                          detectron_label_name=None, oneformer_model_name=None, oneformer_label_name=None, 
                          ensemble_model1=None, ensemble_model2=None,
                          ensemble_label1=None, ensemble_label2=None,
                          apply_refine: bool = True, refine_kwargs: dict | None = None,
                          debug_vis: bool = False, debug_out: str | Path | None = None, 
                          device="cuda"):
    """
    Returns:
      sidewalk_mask (H x W, np.uint8),
      image_pil (PIL image),
      panoptic_data (could be the dictionary from post-processing),
      segments_info (list of segments)
    """

    # 1) Initiate variables
    ens_mask1 = None
    ens_mask2 = None
    panoptic_seg = None
    segments_info = None

    backend = backend.lower()

    # 2) Depending on backend, run the corresponding code
    if backend == "detectron2":
        sidewalk_mask, segments_info, panoptic_seg = segment_with_detectron2(
            img_rgb, detectron_predictor, detectron_cfg, detectron_label_name
        )

    elif backend == "oneformer":
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        oneformer_processor = OneFormerProcessor.from_pretrained(oneformer_model_name)
        oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(
            oneformer_model_name).to(device)
        sidewalk_mask, panoptic_seg, segments_info = segment_with_oneformer(
            img_rgb, oneformer_model, oneformer_processor, device, oneformer_label_name
        )

    elif backend == "ensemble_oneformer_deeplab":                               
        # -- OneFormer part --
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        processor = OneFormerProcessor.from_pretrained(ensemble_model1)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(
            ensemble_model1).to(device).eval()
        
        ens_mask1, _, _ = segment_with_oneformer(
            img_rgb, of_model, processor, device, ensemble_label1)

        # -- DeepLab part --
        if ensemble_model2 is None:
            raise ValueError("deeplab_model must be provided when backend='ensemble'")
        
        print("Running DeepLab segmentation...")
        ens_mask2 = segment_with_deeplab(img_rgb, ensemble_model2, device)
        
        # Check whether masks were generated or not
        if ens_mask1 is None or ens_mask2 is None:
            raise ValueError("Fail when generating one mask")
        
        # Fuse with debug
        from utils import image_processing
        print("Fusing masks...")
        sidewalk_mask = image_processing.fuse_sidewalk_masks(ens_mask1, ens_mask2, method="or")

    elif backend == "ensemble_oneformer_oneformer":                               
        # -- OneFormer part --
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        processor = OneFormerProcessor.from_pretrained(ensemble_model1)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(
            ensemble_model1).to(device).eval()
        
        ens_mask1, _, _ = segment_with_oneformer(
            img_rgb, of_model, processor, device, ensemble_label1)

        # -- OneFormer part --
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        processor = OneFormerProcessor.from_pretrained(ensemble_model2)
        of_model = OneFormerForUniversalSegmentation.from_pretrained(
            ensemble_model2).to(device).eval()
        
        ens_mask2, _, _ = segment_with_oneformer(
            img_rgb, of_model, processor, device, ensemble_label2)
        
        # Check whether masks were generated or not
        if ens_mask1 is None or ens_mask2 is None:
            raise ValueError("Fail when generating one mask")
        
        # Fuse with debug
        from utils import image_processing
        print("Fusing masks...")
        sidewalk_mask = image_processing.fuse_sidewalk_masks(ens_mask1, ens_mask2, method="or")

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Morphological refinement
    if apply_refine:
        refine_kwargs = refine_kwargs or {}
        sidewalk_mask = refine_sidewalk_mask(sidewalk_mask, **refine_kwargs)

    # ── DEBUG VISUALISATION ──
    if debug_vis:
        show_or_save_sidewalk_mask(
            img_rgb, sidewalk_mask, save_path=debug_out
        )

    return sidewalk_mask, panoptic_seg, segments_info, detectron_cfg, ens_mask1, ens_mask2

def segment_sidewalk_and_obstacles(
    img_rgb,
    oneformer_model_name: str,
    sidewalk_label: str = "sidewalk, pavement",
    obstacle_labels: list = None,
    device: str = "cuda",
):
    """
    Returns:
      sidewalk_mask: (H,W) bool
      obstacle_info: list of (label:str, mask:np.bool_)
    """
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    from PIL import Image    

    if obstacle_labels is None:
        obstacle_labels = [
            "traffic light", "fire hydrant", "stop sign", "bench",
            "pole", "tree", "trash can", "curb", "pothole"
        ]

    processor = OneFormerProcessor.from_pretrained(oneformer_model_name)
    model = OneFormerForUniversalSegmentation.from_pretrained(oneformer_model_name).to(device)
    model.eval()

    pil = Image.fromarray(img_rgb)
    inputs = processor(images=pil, task_inputs=["panoptic"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[img_rgb.shape[:2]],
        threshold=0.5,
        mask_threshold=0.5,
        overlap_mask_area_threshold=0.8,
        label_ids_to_fuse=set(range(200)),
    )[0]

    seg_map = result["segmentation"].cpu().numpy()
    segments = result["segments_info"]
    id2label = model.config.id2label

    # sidewalk mask
    sidewalk_mask = np.zeros_like(seg_map, dtype=bool)
    for seg in segments:
        if id2label[seg["label_id"]].lower() == sidewalk_label.lower():
            sidewalk_mask |= (seg_map == seg["id"])

    # obstacle masks with labels
    obstacle_info = []
    for label in obstacle_labels:
        mask = np.zeros_like(seg_map, dtype=bool)
        for seg in segments:
            if id2label[seg["label_id"]].lower() == label.lower():
                mask |= (seg_map == seg["id"])
                obstacle_info.append((label, mask))

    return sidewalk_mask, obstacle_info
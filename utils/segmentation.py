import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation    
from PIL import Image
import torch

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
    img_rgb, model, processor, device="cuda", label_name="sidewalk, pavement"
    ):

    import torch
    import numpy as np
    from PIL import Image

    image_pil = Image.fromarray(img_rgb)

    inputs = processor(images=image_pil, task_inputs=["panoptic"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs, 
        target_sizes=[(img_rgb.shape[0], img_rgb.shape[1])], 
        threshold=0.5,
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

def segment_sidewalk_mask(img_rgb, backend=None, detectron_predictor=None, detectron_cfg=None, 
                          detectron_label_name=None, oneformer_model_name=None, oneformer_label_name=None, 
                          device="cuda"):
    """
    Returns:
      sidewalk_mask (H x W, np.uint8),
      image_pil (PIL image),
      panoptic_data (could be the dictionary from post-processing),
      segments_info (list of segments)
    """

    # 2) Depending on backend, run the corresponding code
    if backend.lower() == "detectron2":
        sidewalk_mask, segments_info, panoptic_seg = segment_with_detectron2(
            img_rgb, detectron_predictor, detectron_cfg, detectron_label_name
        )

    elif backend.lower() == "oneformer":
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        oneformer_processor = OneFormerProcessor.from_pretrained(oneformer_model_name)
        oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(
            oneformer_model_name).to(device)
        sidewalk_mask, panoptic_seg, segments_info = segment_with_oneformer(
            img_rgb, oneformer_model, oneformer_processor, device, oneformer_label_name
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    return sidewalk_mask, panoptic_seg, segments_info, detectron_cfg


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
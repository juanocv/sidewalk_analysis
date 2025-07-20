from __future__ import annotations
import cv2, numpy as np, torch
from pathlib import Path
from ._builder import LABEL_MAP
from detectron2.data.catalog import MetadataCatalog

def make_palette():
    rng = np.random.default_rng(0)
    lut = rng.integers(0,255,(256,3),np.uint8); lut[0]=(0,0,255); return lut

def add_title(img, text):
    canvas = img.copy()
    cv2.putText(canvas, text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas

def overlay_mask(img, mask, color=(0,255,0), alpha=0.4):
    ovl = img.copy()
    ovl[mask] = color
    return cv2.addWeighted(ovl, alpha, img, 1-alpha, 0)

def get_depth_model_name(depth_estimator):
    """Get the correct depth model name for display"""
    if hasattr(depth_estimator, '__class__'):
        class_name = depth_estimator.__class__.__name__
        if 'Zoe' in class_name or 'ZoeDepth' in class_name:
            # Check if it has variant info
            if hasattr(depth_estimator, 'model') and hasattr(depth_estimator.model, 'core'):
                # Try to get the variant from the model architecture
                variant = getattr(depth_estimator, '_variant', 'unknown')
                if variant != 'unknown':
                    return f"ZoeDepth-{variant.upper()}"
                return "ZoeDepth"
            elif hasattr(depth_estimator, '_variant'):
                return f"ZoeDepth-{depth_estimator._variant.upper()}"
            return "ZoeDepth"
        elif 'Midas' in class_name or 'MiDaS' in class_name:
            return "MiDaS"
        else:
            return class_name
    return "Unknown"

def get_segment_info_for_debug(segmenter, img_rgb):
    """Get segmentation info for debug visualization - matches original logic"""
    segments_info = []
    seg_map = None
    
    # Handle ensemble case
    if hasattr(segmenter, 'base') and hasattr(segmenter.base, 'a'):  # ensemble
        base_segmenter = segmenter.base.a
        backend_name = segmenter.backend_name
    else:
        base_segmenter = segmenter.base if hasattr(segmenter, 'base') else segmenter
        backend_name = getattr(segmenter, 'backend_name', 'unknown')
    
    # Run segmentation to get detailed info
    if backend_name == "oneformer":
        try:
            if hasattr(base_segmenter, 'model') and hasattr(base_segmenter, 'processor'):
                inputs = base_segmenter.processor(images=img_rgb, task_inputs=["panoptic"], return_tensors="pt")
                if hasattr(base_segmenter, 'device'):
                    inputs = {k: v.to(base_segmenter.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = base_segmenter.model(**inputs)
                
                result = base_segmenter.processor.post_process_panoptic_segmentation(outputs, target_sizes=[img_rgb.shape[:2]])[0]
                seg_map = result["segmentation"].cpu().numpy()
                segments_info_raw = result["segments_info"]
                
                # Convert to (id, label) format
                if hasattr(base_segmenter.model, 'config') and hasattr(base_segmenter.model.config, 'id2label'):
                    id2label = base_segmenter.model.config.id2label
                    segments_info = [(seg["id"], id2label[seg["label_id"]]) for seg in segments_info_raw]
                    
                    # Store in segmenter for later use
                    segmenter.last_segments_info = segments_info_raw
                    segmenter.last_seg_map = seg_map
                        
        except Exception as e:
            print(f"Failed to get OneFormer segment info: {e}")
                
    elif backend_name == "detectron2":
        try:
            if hasattr(base_segmenter, 'predictor'):
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                outputs = base_segmenter.predictor(img_bgr)
                
                if "panoptic_seg" in outputs:
                    panoptic_seg, segments_info_raw = outputs["panoptic_seg"]
                    seg_map = panoptic_seg.cpu().numpy()
                    
                    # Get metadata for class names
                    metadata = MetadataCatalog.get(base_segmenter.cfg.DATASETS.TRAIN[0])
                    segments_info = []
                    for seg in segments_info_raw:
                        seg_id = seg["id"]
                        cat_id = seg["category_id"]
                        isthing = seg.get("isthing", False)
                        
                        if isthing and hasattr(metadata, 'thing_classes'):
                            class_name = metadata.thing_classes[cat_id]
                        elif hasattr(metadata, 'stuff_classes'):
                            class_name = metadata.stuff_classes[cat_id]
                        else:
                            class_name = f"class_{cat_id}"
                            
                        segments_info.append((seg_id, class_name))
                    
                    # Store in segmenter for later use
                    segmenter.last_segments_info = segments_info_raw
                    segmenter.last_seg_map = seg_map
                    segmenter.metadata = metadata
                        
        except Exception as e:
            print(f"Failed to get Detectron2 segment info: {e}")
    
    return seg_map, segments_info

def _label(sid: int, segmenter, seg_info_list=None) -> str:
    """Enhanced label lookup function matching original logic"""
    
    # 1) First try seg_info list if provided
    if seg_info_list:
        seg_id_to_name = {item[0]: item[1] for item in seg_info_list}
        if sid in seg_id_to_name:
            return seg_id_to_name[sid]
    
    # 2) Try stored segments info
    if hasattr(segmenter, 'last_segments_info'):
        for seg in segmenter.last_segments_info:
            if seg["id"] == sid:
                cat_id = seg["category_id"]
                isthing = seg.get("isthing", False)
                if hasattr(segmenter, 'metadata'):
                    if isthing and hasattr(segmenter.metadata, 'thing_classes'):
                        return segmenter.metadata.thing_classes[cat_id]
                    elif hasattr(segmenter.metadata, 'stuff_classes'):
                        return segmenter.metadata.stuff_classes[cat_id]
    
    # 3) Handle ensemble case - check base segmenter
    base_segmenter = segmenter
    if hasattr(segmenter, 'base'):
        base_segmenter = segmenter.base
        if hasattr(base_segmenter, 'a'):  # ensemble
            base_segmenter = base_segmenter.a
    
    # Get backend name
    backend_name = getattr(segmenter, 'backend_name', 'unknown')

    # 4) DeepLab-specific lookup
    if backend_name == "deeplab" or "deeplab" in backend_name:
        try:
            # Check if the base segmenter has id2label mapping
            if hasattr(base_segmenter, 'id2label') and sid in base_segmenter.id2label:
                return base_segmenter.id2label[sid]
        except Exception as e:
            print(f"DeepLab lookup error: {e}")

    # 5) Detectron2-specific lookup using divisor logic
    if backend_name == "detectron2":
        try:
            if hasattr(base_segmenter, 'cfg'):
                metadata = MetadataCatalog.get(base_segmenter.cfg.DATASETS.TRAIN[0])
                
                # Use divisor logic: category_id * divisor + instance_id
                divisor = 1000  # Standard Detectron2 divisor
                cat_id = sid // divisor
                
                # Try stuff classes first, then thing classes
                if hasattr(metadata, 'stuff_classes') and cat_id < len(metadata.stuff_classes):
                    return metadata.stuff_classes[cat_id]
                elif hasattr(metadata, 'thing_classes') and cat_id < len(metadata.thing_classes):
                    return metadata.thing_classes[cat_id]
                    
        except Exception as e:
            print(f"Detectron2 lookup error: {e}")
    
    # 6) OneFormer-specific lookup
    elif backend_name == "oneformer":
        try:
            if hasattr(base_segmenter, 'model') and hasattr(base_segmenter.model, 'config'):
                config = base_segmenter.model.config
                if hasattr(config, 'id2label') and sid in config.id2label:
                    return config.id2label[sid]
        except Exception as e:
            print(f"OneFormer lookup error: {e}")
    
    # 7) Generic fallback using stored mappings
    ID2LBL = None
    DIV = 1
    
    def _find_id2lbl(obj):
        # Check for DeepLab's id2label first
        if hasattr(obj, "id2label"):
            return getattr(obj, "id2label"), 1
        if hasattr(obj, "id2lbl"):
            return getattr(obj, "id2lbl"), getattr(obj, "label_divisor", 1000)
        for attr in ("base", "a", "b"):
            child = getattr(obj, attr, None)
            if child is not None:
                result = _find_id2lbl(child)
                if result[0] is not None:
                    return result
        # OneFormer keeps mapping in model.config.id2label
        if hasattr(obj, "model") and hasattr(obj.model, "config"):
            return getattr(obj.model.config, "id2label", None), 1
        return None, 1
    
    ID2LBL, DIV = _find_id2lbl(segmenter)
    
    if ID2LBL:
        # Try divisor-based lookup (Detectron2)
        if DIV > 1:
            cat_id = sid // DIV
            if cat_id in ID2LBL:
                return ID2LBL[cat_id]
        
        # Try direct lookup (OneFormer and DeepLab)
        if sid in ID2LBL:
            return ID2LBL[sid]
    
    # 7) Final fallback
    return f"id_{sid}"
    
def write_debug_sheet(res, pipeline, args, segmenter):
    outdir: Path = args.outdir; outdir.mkdir(exist_ok=True, parents=True)
    img_rgb      = pipeline._last_rgb
    img_bgr      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w         = img_bgr.shape[:2]
    tiles        = []

    # 1 panoptic overlay + legend
    is_ensemble = "+" in args.seg
    
    # Get detailed segmentation info for better visualization
    seg_map_debug, segments_info_debug = get_segment_info_for_debug(segmenter, img_rgb)
    
    # Use debug info if available, otherwise fallback to result
    if seg_map_debug is not None:
        seg_map_to_use = seg_map_debug
        segments_info_to_use = segments_info_debug
    else:
        seg_map_to_use = res.seg_map
        segments_info_to_use = getattr(res, "seg_info", [])
    
    if seg_map_to_use is not None:
        seg = seg_map_to_use; uniq = np.unique(seg); lut = make_palette()
        overlay = cv2.addWeighted(img_bgr,.35,lut[seg%256],.65,0)
        legend  = np.full((len(uniq)*22+10,200,3),255,np.uint8)  # Increased width for longer labels

        for i,sid in enumerate(uniq):
            label_name = _label(sid, segmenter, segments_info_to_use)
            cv2.rectangle(legend,(5,i*22+5),(25,i*22+20),lut[sid].tolist(),-1)
            cv2.putText(legend,label_name[:20],(30,i*22+18),  # Increased max length
                        cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        
        if legend.shape[0]<h:
            legend=np.vstack([legend,np.full((h-legend.shape[0],legend.shape[1],3),255,np.uint8)])
        tiles.append(add_title(np.hstack([overlay,legend]),"Panoptic overlay"))

    # 2 refined overlay
    # tiles.append(add_title(overlay_mask(img_bgr,res.sidewalk_mask.astype(bool)),"Sidewalk (refined mask)"))

    # 2 apply refined overlay / masks / ensemble
    if is_ensemble:
        if hasattr(segmenter, 'base') and hasattr(segmenter.base, 'a') and hasattr(segmenter.base, 'b'):
            backend_names = segmenter.backend_name.split("+")
            target_a = LABEL_MAP.get(backend_names[0], ["sidewalk"])
            target_b = LABEL_MAP.get(backend_names[1], ["sidewalk"])
            m1 = segmenter.base.a.segment(img_rgb, target_label=target_a)[0]
            m2 = segmenter.base.b.segment(img_rgb, target_label=target_b)[0]

            # Prepare mask tiles in the desired order
            mask_tiles = []
            for j, m in enumerate([m1, m2, res.sidewalk_mask]):
                mask_bin = (m > 0).astype(np.uint8) * 255
                mask_img = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
                if j == 0:
                    title = f"Mask A ({backend_names[0]})"
                elif j == 1:
                    title = f"Mask B ({backend_names[1]})"
                else:
                    title = "Ensemble mask"
                mask_tiles.append(add_title(mask_img, title))

            # Now append in the desired order:
            # 1. Mask A, 2. Mask B, 3. Ensemble mask, 4. Refined mask, 5. Depth
            tiles = mask_tiles  # Mask A, Mask B, Ensemble mask
            tiles.append(add_title(overlay_mask(img_bgr, res.sidewalk_mask.astype(bool)), "Sidewalk (refined mask overlay)"))
        else:
            # Fallback if ensemble structure is different
            tiles.append(add_title(cv2.cvtColor(res.sidewalk_mask.astype(np.uint8)*255,
                                                cv2.COLOR_GRAY2BGR),
                                "Sidewalk (refined mask only)"))
            tiles.append(add_title(overlay_mask(img_bgr, res.sidewalk_mask.astype(bool)), "Sidewalk (refined mask overlay)"))
    else:
        # Non-ensemble: keep original order
        tiles.append(add_title(overlay_mask(img_bgr, res.sidewalk_mask.astype(bool)), "Sidewalk (refined mask overlay)"))
        tiles.append(add_title(cv2.cvtColor(res.sidewalk_mask.astype(np.uint8)*255,
                                            cv2.COLOR_GRAY2BGR),
                            "Sidewalk (refined mask only)"))

    # 3 depth - FIX: Use correct depth model name
    depth = pipeline.depth_est.predict(img_rgb)
    vis=((depth-depth.min())/(depth.ptp()+1e-6)*255).astype(np.uint8)
    depth_model_name = get_depth_model_name(pipeline.depth_est)
    tiles.append(add_title(cv2.applyColorMap(vis,cv2.COLORMAP_INFERNO),f"Depth ({depth_model_name})"))
    
    # grid + header/footer
    tile_h, tile_w = tiles[0].shape[:2]
    cols = 3
    grid = np.zeros((tile_h * ((len(tiles)+cols-1)//cols),
                    tile_w * cols, 3), np.uint8)

    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        h, w = t.shape[:2]
        grid[r*tile_h:r*tile_h+h, c*tile_w:c*tile_w+w] = t

    # ---------- HEADER ------------------------------------------------
    hdr_h = 40
    header = np.full((hdr_h, grid.shape[1], 3), 30, np.uint8)
    text = f"{args.image.name}   |   seg={args.seg}   |   depth={depth_model_name}"
    cv2.putText(header, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    # ---------- FOOTER ------------------------------------------------
    ftr_h = 30
    footer = np.full((ftr_h, grid.shape[1], 3), 30, np.uint8)
    clear  = ", ".join(f"{c.label}:{c.total_m:.2f}m" for c in res.clearances) \
             if res.clearances else "no obstacles"
    txt2 = f"width = {res.width.width_m:.2f} +/- {res.width.margin_m:.2f} m   " \
           f"|   clearance: {clear}"
    cv2.putText(footer, txt2, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # ---------- COMPOSE ----------------------------------------------
    composite = np.vstack([header, grid, footer])

    cv2.imwrite(str(outdir/f"{args.image.stem}_{args.seg.replace('+', '_')}_{depth_model_name.lower().replace('-', '_')}.png"),
                cv2.cvtColor(composite,cv2.COLOR_RGB2BGR))
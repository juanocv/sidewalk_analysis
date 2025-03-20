import os
import cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def perform_instance_segmentation(image_path, output_path):
    # Load configuration for Cityscapes Mask R-CNN model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}. Skipping.")
        return
    
    # Run inference
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    
    # Visualize and save
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("cityscapes_fine_instance_seg_train"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    print(f"Saved: {output_path}")

def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + "_output.jpg"
            output_path = os.path.join(output_folder, output_filename)
            
            # Perform instance segmentation
            perform_instance_segmentation(input_path, output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)
    
    process_folder(input_folder, output_folder)
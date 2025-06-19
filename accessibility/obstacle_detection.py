import os
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


class SidewalkObstacleDetector:
    def __init__(self, model_name="shi-labs/oneformer_ade20k_swin_large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

        self.obstacle_classes = {
            "traffic light",
            "fire hydrant",
            "stop sign",
            "bench",
            "pole",
            "tree",
            "trash can",
            "curb",
            "pothole",
        }

        self.color_palette = self._create_color_palette()
        self.font = self._load_font()

    def _create_color_palette(self):
        base_colors = plt.cm.tab20.colors 
        additional_colors = plt.cm.tab20b.colors 
        
        fallback_palette = [
            tuple(int(255 * ch) for ch in color[:3]) 
            for color in (base_colors + additional_colors)
        ][::-1] 

        color_map = {
            'tree': (34, 139, 34),         
            'trash can': (128, 0, 128),   
            'fire hydrant': (255, 0, 0),   
            'traffic light': (255, 255, 0),
            'stop sign': (139, 0, 0),
            'bench': (139, 69, 19),
            'pole': (192, 192, 192),  
            'curb': (211, 211, 211),      
            'pothole': (105, 105, 105) 
        }

        return {
            'mapped': color_map,
            'fallback': fallback_palette,
            'default': (128, 128, 128)  # Gray for unknown classes
        }

    def _load_font(self, font_size=20):
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            return ImageFont.load_default()

    def get_class_color(self, class_name):
        clean_name = class_name.lower()
        return self.color_palette["mapped"].get(
            clean_name,
            self.color_palette["fallback"][hash(clean_name) % len(self.color_palette["fallback"])]
        )

    def process_folder(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        for img_path in image_paths:
            try:
                print(f"\nProcessing: {os.path.basename(img_path)}")
                self.process_single_image(img_path, output_folder)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue

    def process_single_image(self, image_path, output_folder):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image, task_inputs=["panoptic"], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        panoptic_seg = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
            threshold=0.5,
            mask_threshold=0.5,
            overlap_mask_area_threshold=0.8,
            label_ids_to_fuse=set(range(100)),
        )[0]

        segmentation_map = panoptic_seg["segmentation"].cpu().numpy()
        rgba_overlay = self.create_segmentation_overlay(
            segmentation_map, panoptic_seg["segments_info"]
        )
        obstacles = self.detect_obstacles(
            panoptic_seg["segments_info"], segmentation_map
        )

        # Create output composite
        overlay_image = Image.fromarray(rgba_overlay)
        composite = Image.alpha_composite(image.convert("RGBA"), overlay_image)
        draw = ImageDraw.Draw(composite)

        # Add obstacle labels
        for obstacle in obstacles:
            cx, cy = obstacle["centroid"]
            text = obstacle["class"].capitalize()
            
            # Calculate text bounding box with padding
            text_bbox = draw.textbbox((cx, cy), text, font=self.font, anchor="mm")
            padding = 5
            expanded_bbox = (
                text_bbox[0] - padding,
                text_bbox[1] - padding,
                text_bbox[2] + padding,
                text_bbox[3] + padding,
            )
            
            # Draw background and text
            draw.rectangle(expanded_bbox, fill=(0, 0, 0, 191))
            draw.text(
                (cx, cy),
                text,
                fill="white",
                font=self.font,
                anchor="mm"
            )

        # Save result
        output_path = os.path.join(
            output_folder,
            f"seg_{os.path.splitext(os.path.basename(image_path))[0]}.png"
        )
        composite.convert("RGB").save(output_path)
        print(f"Saved result to: {output_path}")

    def create_segmentation_overlay(self, segmentation_map, segments_info):
        color_seg = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
        alpha = np.zeros(segmentation_map.shape, dtype=np.uint8)

        for segment in segments_info:
            class_id = segment["label_id"]
            class_name = self.model.config.id2label.get(class_id, "unknown")
            color = self.get_class_color(class_name)
            mask = segmentation_map == segment["id"]

            if class_name.lower() in self.obstacle_classes:
                color_seg[mask] = color
                alpha[mask] = int(0.9 * 255)

        return np.dstack((color_seg, alpha))

    def detect_obstacles(self, segments_info, segmentation_map):
        obstacles = []
        for segment in segments_info:
            class_id = segment["label_id"]
            class_name = self.model.config.id2label.get(class_id, "unknown").lower()

            if class_name in self.obstacle_classes:
                mask = segmentation_map == segment["id"]
                y_indices, x_indices = np.where(mask)

                if len(y_indices) > 0:
                    cx = np.mean(x_indices)
                    cy = np.mean(y_indices)
                    obstacles.append({"class": class_name, "centroid": (cx, cy)})
        return obstacles


if __name__ == "__main__":
    detector = SidewalkObstacleDetector()
    detector.process_folder(
        input_folder="generic/images", 
        output_folder="generic/output/v5"
    )
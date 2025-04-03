import os
import torch
from PIL import Image
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
            "fence",
            "tree",
            "trash can",
            "curb",
            "pothole",
        }

        self.color_palette = self._create_color_palette()

    def _create_color_palette(self):
        base_colors = plt.cm.tab20.colors 
        additional_colors = plt.cm.tab20b.colors 
        

        fallback_palette = [
            tuple(int(255 * ch) for ch in color[:3]) 
            for color in (base_colors + additional_colors)
        ][::-1] 

        color_map = {
            'tree': (34, 139, 34),         # Forest Green
            'trash can': (128, 0, 128),    # Purple
            'fire hydrant': (255, 0, 0),   # Red
            'traffic light': (255, 255, 0),# Yellow
        }

        return {
            'mapped': color_map,
            'fallback': fallback_palette,
            'default': (128, 128, 128)  # Gray for unknown classes
        }

    def get_class_color(self, class_name):
        clean_name = class_name.lower()

        if clean_name in self.color_palette["mapped"]:
            return self.color_palette["mapped"][clean_name]

        try:
            index = hash(clean_name) % len(self.color_palette["fallback"])
            return self.color_palette["fallback"][index]
        except:
            return self.color_palette["default"]

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
                result = self.process_single_image(img_path, output_folder)
                self.display_results(result)
            except Exception as e:
                print(f"Error processing image: {str(e)}")
                continue

    def process_single_image(self, image_path, output_folder):
        image = Image.open(image_path)
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

        return {
            "original_image": image,
            "rgba_overlay": rgba_overlay,
            "obstacles": obstacles,
            "output_path": os.path.join(
                output_folder,
                f"seg_{os.path.splitext(os.path.basename(image_path))[0]}.png",
            ),
        }

    def display_results(self, result):
        plt.figure(figsize=(24, 12), facecolor="white")
        plt.suptitle(
            f"Analysis Results: {os.path.basename(result['output_path'])}", fontsize=16
        )

        # Left panel: Image with overlay
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(result["original_image"])
        plt.imshow(result["rgba_overlay"])
        self.add_labels_to_plot(result["obstacles"])
        plt.title("Obstacle Detection Overlay")
        plt.axis("off")

        # Right panel: Text information
        ax2 = plt.subplot(1, 2, 2)
        self.create_text_panel(result["obstacles"])

        plt.savefig(
            result["output_path"],
            bbox_inches="tight",
            facecolor="white",
            dpi=300,
            transparent=False,
        )
        plt.close()

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

    def create_text_panel(self, obstacles):
        plt.axis("off")
        ax = plt.gca()

        # Add semi-transparent background
        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                facecolor=(1, 1, 1, 0.85),  # 85% white
                edgecolor="none",
                zorder=-1,
            )
        )

        obstacle_text = self.generate_obstacle_text(obstacles)
        plt.text(
            0.05,
            0.5,
            obstacle_text,
            fontsize=12,
            color="black",
            family="monospace",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    def add_labels_to_plot(self, obstacles):
        for obstacle in obstacles:
            cx, cy = obstacle["centroid"]
            plt.text(
                cx,
                cy,
                obstacle["class"].capitalize(),
                fontsize=10,
                color="white",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor=(0, 0, 0, 0.75),  # 75% black
                    edgecolor="none",
                    boxstyle="round,pad=0.3",
                ),
            )

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

    def generate_obstacle_text(self, obstacles):
        if not obstacles:
            return "No obstacles detected"

        text = "Detected Obstacles:\n\n"
        for idx, obs in enumerate(obstacles, 1):
            text += f"{idx}. {obs['class'].capitalize()}\n"
        return text


if __name__ == "__main__":
    detector = SidewalkObstacleDetector()
    detector.process_folder(
        input_folder="generic/images", output_folder="generic/output/oneformer_ade20k"
    )

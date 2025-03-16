import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


class SidewalkObstacleDetector:
    def __init__(self, model_name="shi-labs/oneformer_ade20k_swin_large"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            model_name).to(self.device)
        self.model.eval()

        # Gambiarra para cores diferentes por classe
        self.ade20k_palette = np.array([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                                        [4, 200, 3], [120, 120, 80], [
                                            140, 140, 140], [204, 5, 255],
                                        [230, 230, 230], [4, 250, 7], [
                                            224, 5, 255], [235, 255, 7],
                                        [150, 5, 61], [120, 120, 70], [
                                            8, 255, 51], [255, 6, 82],
                                        [143, 255, 140], [204, 255, 4], [
                                            255, 51, 7], [204, 70, 3],
                                        [0, 102, 200], [61, 230, 250], [
                                            255, 6, 51], [11, 102, 255],
                                        [255, 7, 71], [255, 9, 224], [
                                            9, 7, 230], [220, 220, 220],
                                        [255, 9, 92], [112, 9, 255], [
                                            8, 255, 214], [7, 255, 224],
                                        [255, 184, 6], [10, 255, 71], [
                                            255, 41, 10], [7, 255, 255],
                                        [224, 255, 8], [102, 8, 255], [
                                            255, 61, 6], [255, 194, 7],
                                        [255, 122, 8], [0, 255, 20], [
                                            255, 8, 41], [255, 5, 153],
                                        [6, 51, 255], [235, 12, 255], [
                                            160, 150, 20], [0, 163, 255],
                                        [140, 140, 140], [250, 10, 15], [
                                            20, 255, 0], [31, 255, 0],
                                        [255, 31, 0], [255, 224, 0], [
                                            153, 255, 0], [0, 0, 255],
                                        [255, 71, 0], [0, 235, 255], [
                                            0, 173, 255], [31, 0, 255],
                                        [11, 200, 200], [25, 25, 25], [
                                            180, 120, 120], [6, 230, 230],
                                        [80, 50, 50], [4, 200, 3], [
                                            120, 120, 80], [140, 140, 140],
                                        [204, 5, 255], [230, 230, 230], [
                                            4, 250, 7], [224, 5, 255],
                                        [235, 255, 7], [150, 5, 61], [
                                            120, 120, 70], [8, 255, 51],
                                        [255, 6, 82], [143, 255, 140], [
                                            204, 255, 4], [255, 51, 7],
                                        [204, 70, 3], [0, 102, 200], [
                                            61, 230, 250], [255, 6, 51],
                                        [11, 102, 255], [255, 7, 71], [
                                            255, 9, 224], [9, 7, 230],
                                        [220, 220, 220], [255, 9, 92], [
                                            112, 9, 255], [8, 255, 214],
                                        [7, 255, 224], [255, 184, 6], [
                                            10, 255, 71], [255, 41, 10],
                                        [7, 255, 255], [224, 255, 8], [
                                            102, 8, 255], [255, 61, 6],
                                        [255, 194, 7], [255, 122, 8], [
                                            0, 255, 20], [255, 8, 41],
                                        [255, 5, 153], [6, 51, 255], [
                                            235, 12, 255], [160, 150, 20],
                                        [0, 163, 255], [140, 140, 140], [
                                            250, 10, 15], [20, 255, 0],
                                        [31, 255, 0], [255, 31, 0], [
                                            255, 224, 0], [153, 255, 0],
                                        [0, 0, 255], [255, 71, 0], [
                                            0, 235, 255], [0, 173, 255],
                                        [31, 0, 255], [11, 200, 200], [
                                            25, 25, 25], [180, 120, 120],
                                        [6, 230, 230], [80, 50, 50], [
                                            4, 200, 3], [120, 120, 80],
                                        [140, 140, 140], [204, 5, 255], [
                                            230, 230, 230], [4, 250, 7],
                                        [224, 5, 255], [235, 255, 7], [
                                            150, 5, 61], [120, 120, 70],
                                        [8, 255, 51], [255, 6, 82], [
                                            143, 255, 140], [204, 255, 4],
                                        [255, 51, 7], [204, 70, 3], [
                                            0, 102, 200], [61, 230, 250],
                                        [255, 6, 51], [11, 102, 255], [
                                            255, 7, 71], [255, 9, 224],
                                        [9, 7, 230], [220, 220, 220], [
                                            255, 9, 92], [112, 9, 255],
                                        [8, 255, 214], [7, 255, 224], [
                                            255, 184, 6], [10, 255, 71],
                                        [255, 41, 10], [7, 255, 255], [
                                            224, 255, 8], [102, 8, 255],
                                        [255, 61, 6], [255, 194, 7], [
                                            255, 122, 8], [0, 255, 20],
                                        [255, 8, 41], [255, 5, 153], [
                                            6, 51, 255], [235, 12, 255],
                                        [160, 150, 20], [0, 163, 255], [
                                            140, 140, 140], [250, 10, 15],
                                        [20, 255, 0], [31, 255, 0], [
                                            255, 31, 0], [255, 224, 0],
                                        [153, 255, 0], [0, 0, 255], [
                                            255, 71, 0], [0, 235, 255],
                                        [0, 173, 255], [31, 0, 255], [11, 200, 200], [25, 25, 25]])

    def process_folder(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        image_extensions = ['.jpg', '.jpeg', '.png']
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
        inputs = self.processor(images=image, task_inputs=[
                                "panoptic"], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        panoptic_seg = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
            threshold=0.5,
            mask_threshold=0.5,
            overlap_mask_area_threshold=0.8,
        )[0]

        output_path = os.path.join(
            output_folder, f"seg_{os.path.basename(image_path)}")
        return {
            'original_image': image,
            'segmentation': panoptic_seg,
            'output_path': output_path,
            'obstacles': self.detect_obstacles(panoptic_seg["segments_info"], image.size)
        }

    def display_results(self, result):
        # Create figure
        fig = plt.figure(figsize=(24, 12))
        fig.suptitle(
            f"Analysis Results: {os.path.basename(result['output_path'])}", fontsize=16)

        # Original Image
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(result['original_image'])
        ax1.set_title("Original Image", fontsize=12)
        ax1.axis('off')

        # Segmentation Result
        ax2 = fig.add_subplot(1, 3, 2)
        segmentation_map = result['segmentation']["segmentation"].cpu().numpy()
        segments_info = result['segmentation']["segments_info"]
        color_seg = self.create_segmentation_overlay(
            segmentation_map, segments_info)
        ax2.imshow(color_seg)
        ax2.set_title("Segmentation Result", fontsize=12)
        ax2.axis('off')

        # Obstacle Information
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.axis('off')
        if result['obstacles']:
            obstacle_text = "Detected Obstacles:\n\n"
            for idx, obs in enumerate(result['obstacles'], 1):
                obstacle_text += (
                    f"{idx}. {obs['class']}\n"
                )
        else:
            obstacle_text = "No obstacles detected"

        ax3.text(0, 0.5, obstacle_text,
                 fontsize=12,
                 family='monospace',
                 verticalalignment='center')

        # Save and show
        plt.savefig(result['output_path'], bbox_inches='tight')
        plt.show()
        plt.close()

    def create_segmentation_overlay(self, segmentation_map, segments_info):
        color_seg = np.zeros(
            (segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
        for segment in segments_info:
            class_id = segment["label_id"]
            mask = segmentation_map == segment["id"]
            color_seg[mask] = self.ade20k_palette[class_id %
                                                  len(self.ade20k_palette)]
        return color_seg

    def visualize_and_save(self, image, segmentation_results, output_path):
        segmentation_map = segmentation_results["segmentation"].cpu().numpy()
        segments_info = segmentation_results["segments_info"]
        height, width = image.size[1], image.size[0]

        color_seg = np.zeros((height, width, 3), dtype=np.uint8)

        for segment in segments_info:
            class_id = segment["label_id"]
            mask = segmentation_map == segment["id"]
            color_seg[mask] = self.ade20k_palette[class_id %
                                                  len(self.ade20k_palette)]

        plt.figure(figsize=(20, 10))
        plt.imshow(color_seg)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def detect_obstacles(self, segments_info, image_size):
        obstacles = []
        obstacle_classes = {
            'person', 'bicycle', 'car', 'motorcycle', 'traffic light',
            'fire hydrant', 'stop sign', 'bench', 'pole', 'fence',
            'tree', 'trash can', 'dog', 'cat', 'curb', 'pothole'
        }

        total_pixels = image_size[0] * image_size[1]

        for segment in segments_info:
            class_name = self.model.config.id2label.get(
                segment["label_id"], "unknown")

            if class_name.lower() in obstacle_classes:
                obstacles.append({
                    'class': class_name,
                })

        return obstacles


if __name__ == "__main__":
    detector = SidewalkObstacleDetector()
    detector.process_folder(
        input_folder="/generic/images",
        output_folder="/generic/output/oneformer_ade20k_swin_large"
    )

import torch
from PIL import Image
import numpy as np
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


class ObstacleDetector:
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

    def detect_obstacles(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image, task_inputs=["panoptic"], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Generate label_ids_to_fuse (all non-obstacle classes)
        label2id = {v: k for k, v in self.model.config.id2label.items()}
        label_ids_to_fuse = {
            label2id[class_name]
            for class_name in self.model.config.id2label.values()
            if class_name.lower() not in self.obstacle_classes
        }

        panoptic_seg = self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
            threshold=0.5,
            label_ids_to_fuse=label_ids_to_fuse, 
        )[0]

        return {
            "image": image,
            "segmentation_map": panoptic_seg["segmentation"].cpu().numpy(),
            "segments_info": panoptic_seg["segments_info"],
            "obstacles": self._process_segments(
                panoptic_seg["segments_info"], image.size
            ),
        }

    def _process_segments(self, segments_info, image_size):
        obstacles = []
        total_pixels = image_size[0] * image_size[1]

        for segment in segments_info:
            class_name = self.model.config.id2label.get(segment["label_id"], "unknown")
            if class_name.lower() in self.obstacle_classes:
                obstacles.append(
                    {
                        "class": class_name,
                        "area": segment.get("area", 0) / total_pixels * 100,
                        "mask": segment["id"],
                    }
                )
        return obstacles

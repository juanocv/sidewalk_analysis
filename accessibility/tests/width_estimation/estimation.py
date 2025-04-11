import torch
import numpy as np
from torchvision import transforms as T
from PIL import Image


class WidthEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device)
        self.midas.eval()

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def estimate_widths(self, image_path, segmentation_map, obstacles):
        depth_map = self._get_depth_map(image_path)
        focal_length = self._calculate_focal_length(segmentation_map.shape)

        results = []
        for obstacle in obstacles:
            mask = segmentation_map == obstacle["mask"]
            width, margin = self._calculate_obstacle_width(
                mask, depth_map, focal_length
            )
            results.append({**obstacle, "width_m": width, "margin_m": margin})
        return results

    def _get_depth_map(self, image_path):
        img_pil = Image.open(image_path).convert("RGB")
        original_size = img_pil.size

        input_batch = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth_map = self.midas(input_batch)
            depth_map = (
                torch.nn.functional.interpolate(
                    depth_map.unsqueeze(1),
                    size=original_size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        return depth_map

    def _calculate_focal_length(self, img_shape):
        return img_shape[1] / (2 * np.tan(np.radians(75 / 2)))

    def _calculate_obstacle_width(self, mask, depth_map, focal_length):
        y_coords, x_coords = np.where(mask)

        if len(y_coords) > 0:
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            y_threshold = max_y - (max_y - min_y) * 0.3  # Bottom 20%

            # Filter coordinates
            bottom_mask = y_coords >= y_threshold
            y_coords = y_coords[bottom_mask]
            x_coords = x_coords[bottom_mask]

        if len(x_coords) == 0:
            return 0.0, 0.0

        z_values = depth_map[y_coords, x_coords]
        x_real = (x_coords - mask.shape[1] / 2) * z_values / focal_length

        x_sorted = np.sort(x_real)
        low_idx = int(0.02 * len(x_sorted))
        high_idx = int(0.98 * len(x_sorted))
        trimmed = x_sorted[low_idx:high_idx]

        width = (trimmed[-1] - trimmed[0]) / 10
        margin = (0.25 * width) / 10
        return round(width, 2), round(margin, 2)

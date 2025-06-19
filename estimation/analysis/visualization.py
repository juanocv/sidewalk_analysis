import numpy as np
import os
import matplotlib.pyplot as plt

def plot_3d_point_cloud(points_3d, title="3D Point Cloud"):
    """
    points_3d: a NumPy array of shape (N, 3) containing [X, Y, Z] coordinates
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        points_3d[:, 0], 
        points_3d[:, 1], 
        points_3d[:, 2],
        s=2  # marker size
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.show()

    # If you want to save as an image file (e.g. PNG), you can do:
    # plt.savefig("my_point_cloud.png")

def draw_horizontal_legend(ax, legend_data):
    """
    legend_data: list of (class_name, [R,G,B]) for each segment
    ax: a matplotlib Axes on which to draw the legend in a horizontal row.
    """
    import matplotlib.patches as mpatches
    import numpy as np

    # remove duplicates if multiple segments share the same class
    unique_legend = {}
    for cls, color in legend_data:
        if cls not in unique_legend:
            unique_legend[cls] = color

    # We place squares horizontally, each 0.2 wide, 0.2 tall, with some margin
    x_offset = 0.0
    for cls, color in unique_legend.items():
        # color is [R,G,B] in 0-255
        color_rgb = np.array(color)/255.0 if np.max(color)>1 else color

        # Draw a small rectangle patch
        rect = mpatches.Rectangle((x_offset, 0.0), 0.2, 0.2,
                                  edgecolor='none', facecolor=color_rgb)
        ax.add_patch(rect)
        # Put text to the right of the square
        ax.text(x_offset+0.25, 0.1, cls, va='center', fontsize=9)

        x_offset += 1.0  # move horizontally for the next patch

    # set x-limits to fit all classes
    ax.set_xlim(0, max(2, x_offset))
    ax.set_ylim(0, 0.3)  # just enough vertical space for squares + text
    ax.axis('off')
    ax.set_title("Segment Legend")

def oneformer_visualize(output_path, img_rgb, seg_map, segments_info, model_name):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from transformers import OneFormerForUniversalSegmentation

    oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(model_name)

    # Build a palette
    palette = []
    np.random.seed(42)
    for i in range(150):
        palette.append(np.random.randint(0, 256, size=3))

    # Create color overlay
    height, width = seg_map.shape
    color_seg = np.zeros((height, width, 3), dtype=np.uint8)

    legend_data = []
    for seg_dict in segments_info:
        seg_id = seg_dict["id"]
        label_id = seg_dict["label_id"]
        class_name = oneformer_model.config.id2label[label_id]
        color = palette[label_id % len(palette)]

        mask = (seg_map == seg_id)
        color_seg[mask] = color
        legend_data.append((class_name, color))

    final_overlay = cv2.addWeighted(img_rgb, 0.5, color_seg, 0.5, 0)

    # Create figure with 2 rows and 1 column:
    #   Row 1: the overlay
    #   Row 2: the legend (shorter)
    fig, (ax_img, ax_legend) = plt.subplots(
        nrows=2, 
        figsize=(10, 8), 
        gridspec_kw={"height_ratios": [5,1]}  # bigger for top, smaller for bottom
    )

    # Top row: show the overlay
    ax_img.imshow(final_overlay)
    ax_img.axis('off')
    ax_img.set_title("OneFormer Panoptic Overlay")

    # Bottom row: draw horizontal legend
    draw_horizontal_legend(ax_legend, legend_data)

    plt.tight_layout()
    plt.savefig(output_path)

def detectron2_visualize(output_path, img_rgb, panoptic_seg, segments_info, cfg):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from detectron2.data import MetadataCatalog

    # Build an overlay + legend_data similarly
    color_seg = np.zeros_like(img_rgb, dtype=np.uint8)
    legend_data = []

    for seg in segments_info:
        seg_id = seg["id"]
        cat_id = seg["category_id"]
        isthing = seg.get("isthing", False)

        if isthing:
            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[cat_id]
        else:
            class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes[cat_id]

        color = np.random.randint(0,256,size=3,dtype=np.uint8)
        mask = (panoptic_seg == seg_id)
        color_seg[mask] = color

        legend_data.append((class_name, color))

    final_overlay = cv2.addWeighted(img_rgb,0.5,color_seg,0.5,0)

    fig, (ax_img, ax_legend) = plt.subplots(
        nrows=2,
        figsize=(10,8),
        gridspec_kw={"height_ratios": [5,1]}
    )

    ax_img.imshow(final_overlay)
    ax_img.axis('off')
    ax_img.set_title("Detectron2 Panoptic Overlay")

    draw_horizontal_legend(ax_legend, legend_data)

    plt.tight_layout()
    plt.savefig(output_path)

def midas_visualize(img_path, img_rgb, panoptic_seg, segments_info, backend=None, oneformer_model_name=None, cfg=None):
    # set main output file
    output_file = os.path.splitext(os.path.basename(img_path))[0] + '.png'
    
    if backend.lower() == "detectron2":
        #----------------------------------
        # DETECTRON2 VISUALIZATION
        #----------------------------------
        from detectron2.utils.visualizer import Visualizer, ColorMode
        from detectron2.data import MetadataCatalog

        if cfg is None:
            raise ValueError("cfg is required for detectron2 visualization")

        # 'panoptic_seg' is typically a torch.Tensor of shape [H, W]
        # or we might do panoptic_seg.cpu() above.
        # Ensure we have a CPU tensor if needed:
        if hasattr(panoptic_seg, "cpu"):
            panoptic_seg = panoptic_seg.cpu().numpy()

        # specialize path for folder
        output_path = os.path.join("estimation\output\detectron2", output_file)
        detectron2_visualize(output_path, img_rgb, panoptic_seg, segments_info, cfg)

    elif backend.lower() == "oneformer":
        #----------------------------------
        # ONEFORMER VISUALIZATION
        #----------------------------------
        if oneformer_model_name is None:
            raise ValueError("oneformer_model_name is required for OneFormer visualization")

        # If 'panoptic_seg' is a dict from post_process (like {"segmentation": array, "segments_info":...}),
        # we might need to unify it. Check if it's a dict or an array.
        if isinstance(panoptic_seg, dict) and "segmentation" in panoptic_seg:
            seg_map = panoptic_seg["segmentation"]
        else:
            seg_map = panoptic_seg  # assume it's already the raw segmentation array

        # Convert to numpy if needed
        if hasattr(seg_map, "cpu"):
            seg_map = seg_map.cpu().numpy()
        seg_map = np.array(seg_map, dtype=np.int32)

        # specialize path for folder
        output_path = os.path.join("estimation\output\oneformer", output_file)
        oneformer_visualize(output_path, img_rgb, seg_map, segments_info, oneformer_model_name)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
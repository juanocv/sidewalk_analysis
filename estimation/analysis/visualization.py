import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

def visualize_open3d(points_3d):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Optional: color the points, e.g. white
    # pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points_3d) * 0.9)

    o3d.visualization.draw_geometries([pcd])

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

def midas_visualize(img, cfg, panoptic_seg, segments_info):
        # Visualization
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        v = Visualizer(
            img_rgb,
            metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            scale=1.0,
            instance_mode=ColorMode.SEGMENTATION
        )
        # Draw instances and return image
        out = v.draw_panoptic_seg_predictions(panoptic_seg.cpu(), segments_info)
        plt.imshow(out.get_image())
        plt.show()
from estimation import analysis
from analysis import *

def estimate_width_3dr(img1_path, img2_path, predictor, cfg):
    # Capture each mask for each image
    mask1, _, _, _ = analysis.segment_sidewalk_mask(img1_path, predictor, cfg)
    mask2, _, _, _ = analysis.segment_sidewalk_mask(img2_path, predictor, cfg)
    # Apply 3D Reconstruction with these two images
    R, t_scaled, points_3d, pts1_in, pts2_in = analysis.two_view_reconstruction(img1_path, img2_path, mask1, mask2)

    #print("Recovered rotation:\n", R)
    #print("Recovered (scaled) translation:\n", t_scaled)
    #print("Triangulated 3D points shape:", points_3d.shape)
    #print("Triangulated sidewalk features (in meters):", points_3d)

    # Estimate sidewalk width based on generated 3D cloud
    sidewalk_width = analysis.estimate_sidewalk_width(points_3d)

    #print("t_scaled:", t_scaled.ravel())
    #print("||t_scaled|| =", np.linalg.norm(t_scaled))
    analysis.plot_3d_point_cloud(points_3d, title="Reconstructed Sidewalk Points")

    return sidewalk_width
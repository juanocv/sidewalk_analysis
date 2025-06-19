import cv2
from .visualization import plot_3d_point_cloud
from utils import *

def fit_plane_least_squares(points):
    """
    points: (N,3) array of [X, Y, Z]
    returns (a,b,c,d) such that aX + bY + cZ + d = 0
    """
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    # We'll solve for (a,b,c,d) in the sense of minimizing sum of distances^2
    # One approach: stack columns [X, Y, Z, 1]
    A = np.column_stack((X, Y, Z, np.ones_like(X)))
    # We want to find vector p = [a,b,c,d] that leads to p·[X,Y,Z,1] ~ 0
    # In practice, do the SVD on A and the solution is the last right singular vector.
    # That gives A*p = 0 solution.
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    p = vh[-1]  # the vector corresponding to smallest singular value
    a, b, c, d = p
    # Usually we normalize so that sqrt(a^2+b^2+c^2) = 1
    norm_abc = np.linalg.norm([a,b,c])
    if norm_abc < 1e-12:
        return None
    a, b, c, d = a/norm_abc, b/norm_abc, c/norm_abc, d/norm_abc
    return (a,b,c,d)

def estimate_width_3d_cloud(points_3d):
    """
    points_3d: Nx3 array (in meters) for sidewalk
    Returns an approximate width in the plane of these points.
    """
    # 1) Fit plane
    a, b, c, d = fit_plane_least_squares(points_3d)
    n = np.array([a,b,c])
    
    # 2) Get centroid
    centroid = np.mean(points_3d, axis=0)
    
    # 3) Compute plane axes (u, v)
    aux = np.array([1,0,0])
    if abs(n.dot(aux)) > 0.9:
        aux = np.array([0,1,0])
    n /= np.linalg.norm(n)
    u = np.cross(n, aux)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    
    # 4) Project onto (u,v)
    uv_coords = []
    for p in points_3d:
        p_shifted = p - centroid
        u_coord = p_shifted.dot(u)
        v_coord = p_shifted.dot(v)
        uv_coords.append([u_coord, v_coord])
    uv_coords = np.array(uv_coords)
    
    # 5) PCA
    X = uv_coords - np.mean(uv_coords, axis=0)
    cov = np.cov(X.T)
    e_vals, e_vecs = np.linalg.eig(cov)
    idx = np.argsort(e_vals)[::-1]
    e1 = e_vecs[:, idx[0]]  # largest var
    e2 = e_vecs[:, idx[1]]  # second var
    
    # 6) measure extent along e2 => width
    coords_along_e2 = X.dot(e2)
    w_min, w_max = coords_along_e2.min(), coords_along_e2.max()
    width = w_max - w_min
    
    return width

def two_view_reconstruction(img1_path, img2_path, mask1, mask2):
    # Read both images 
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Could not load one or both images.")
    
    # Crop out Google's watermark (may interfere in possible keypoint matches)
    img1 = img1[:400-20,:]
    img2 = img2[:400-20,:]

    img1_height, img1_width = img1.shape[:2]
    #img2_height, img2_width = img2.shape[:2] - unnecessary

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints & descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    print(f"Image1: {len(keypoints1)} keypoints")
    print(f"Image2: {len(keypoints2)} keypoints")

    # Create a brute force matcher (L2 norm for SIFT)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by their distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Total matches: {len(matches)}")

    sidewalk_matches = []

    for m in matches:
        # Keypoint in image1
        (x1, y1) = keypoints1[m.queryIdx].pt  # floating coords
        # Keypoint in image2
        (x2, y2) = keypoints2[m.trainIdx].pt

        # Round or cast to int
        ix1, iy1 = int(round(x1)), int(round(y1))
        ix2, iy2 = int(round(x2)), int(round(y2))

        # Check if inside the mask and not out of bounds
        if (0 <= iy1 < mask1.shape[0] and 0 <= ix1 < mask1.shape[1]
            and 0 <= iy2 < mask2.shape[0] and 0 <= ix2 < mask2.shape[1]):
            if mask1[iy1, ix1] == 1 and mask2[iy2, ix2] == 1:
                sidewalk_matches.append(m)

    print(f"All matches: {len(matches)}  Sidewalk matches: {len(sidewalk_matches)}")

    matched_img = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        sidewalk_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("SIFT Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in sidewalk_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in sidewalk_matches])

    #Essential matrix (3.3)
    # Example: Suppose image width = W, height = H, horizontal FOV = fov
    # Then fx = (W / 2) / tan(fov/2)  (approx.)
    # Principal point ~ (cx, cy) = (W/2, H/2)

    # 1) Intrinsic approximation
    fx = (img1_width/2.) / np.tan(np.radians(70 / 2.))  # from your FOV
    cx, cy = (img1_width/2., img1_height/2.)
    baseline_meters = 7

    # 2) Find essential matrix via RANSAC
    E, mask = cv2.findEssentialMat(
        points1, points2,
        focal=fx, 
        pp=(cx, cy),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    # 3) Filter inliers
    inlier_points1 = points1[mask.ravel() == 1]
    inlier_points2 = points2[mask.ravel() == 1]

    # 4) Recover pose
    retval, R, t, mask_pose = cv2.recoverPose(
        E,
        inlier_points1, 
        inlier_points2,
        focal=fx, 
        pp=(cx, cy),
    )

    # 5) Current length of t
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-8:  # avoid div by zero
        scale_factor = baseline_meters / t_norm
        t_scaled = t * scale_factor
    else:
        t_scaled = t  # fallback, won't be correct scale

    # 6) Build Projection Matrices for Triangulation
    # K intrinsics:
    K = np.array([[fx,  0, cx],
                  [ 0, fx, cy],
                  [ 0,  0,  1]], dtype=np.float64)
    
    # First camera assumed at origin:
    # P1 = K [I | 0]
    I = np.eye(3, dtype=np.float64)
    zero = np.zeros((3,1), dtype=np.float64)
    P1 = K @ np.hstack((I, zero))

    # Second camera at (R, t_scaled)
    P2 = K @ np.hstack((R, t_scaled))

    # Inlier sets
    points1_2d = inlier_points1
    points2_2d = inlier_points2

    # Prepare for cv2.triangulatePoints
    points1_h = np.vstack((points1_2d.T, np.ones((1, points1_2d.shape[0]))))
    points2_h = np.vstack((points2_2d.T, np.ones((1, points2_2d.shape[0]))))

    # 7) Triangulate Points (in homogeneous coords)
    points_4d = cv2.triangulatePoints(P1, P2, points1_h[:2], points2_h[:2])
    # shape = (4, N)

    # 8) Convert from homogeneous to 3D: X = x/w, Y = y/w, Z = z/w
    points_3d = (points_4d[:3] / points_4d[3]).T  # shape (N, 3)
    
    # Return triangulated 3D points, 
    return R, t_scaled, points_3d, inlier_points1, inlier_points2

def estimate_width_3dr(img1_path, img2_path, backend=None, detectron_predictor=None, detectron_cfg=None, 
                       detectron_label_name=None, oneformer_model_name=None, oneformer_label_name=None, device="cuda"):
    img1_rgb = read_rgbimg(img1_path)
    img2_rgb = read_rgbimg(img2_path)
    mask1, _, _, _ = segment_sidewalk_mask(
        img1_rgb,
        backend=backend,
        detectron_predictor=detectron_predictor,
        detectron_cfg=detectron_cfg,
        detectron_label_name=detectron_label_name,
        oneformer_model_name=oneformer_model_name,
        oneformer_label_name=oneformer_label_name,
        device=device
    )
    mask2, _, _, _ = segment_sidewalk_mask(
        img2_rgb,
        backend=backend,
        detectron_predictor=detectron_predictor,
        detectron_cfg=detectron_cfg,
        detectron_label_name=detectron_label_name,
        oneformer_model_name=oneformer_model_name,
        oneformer_label_name=oneformer_label_name,
        device=device
    )
    # Apply 3D Reconstruction with these two images
    R, t_scaled, points_3d, pts1_in, pts2_in = two_view_reconstruction(img1_path, img2_path, mask1, mask2)

    #print("Recovered rotation:\n", R)
    #print("Recovered (scaled) translation:\n", t_scaled)
    #print("Triangulated 3D points shape:", points_3d.shape)
    #print("Triangulated sidewalk features (in meters):", points_3d)

    # Estimate sidewalk width based on generated 3D cloud
    sidewalk_width = estimate_width_3d_cloud(points_3d)

    #print("t_scaled:", t_scaled.ravel())
    #print("||t_scaled|| =", np.linalg.norm(t_scaled))
    plot_3d_point_cloud(points_3d, title="Reconstructed Sidewalk Points")

    return sidewalk_width
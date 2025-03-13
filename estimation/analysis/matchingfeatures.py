import cv2
import numpy as np

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

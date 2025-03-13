#import cv2
import numpy as np

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

def estimate_sidewalk_width(points_3d):
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
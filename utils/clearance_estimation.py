import cv2
import numpy as np
from skimage.morphology import medial_axis
from utils import segment_sidewalk_and_obstacles
from estimation.analysis.midas_estimation import get_depth_map



def scale_from_ground(S, Z_raw, fx, fy, cx, cy, Hcam=2.70, N=8000, frac_good=0.08):

    ys, xs = np.where(S)
    M = xs.size

    rng = np.random.default_rng(0)
    sel = rng.choice(M, size=min(N, M), replace=False)
    u, v = xs[sel], ys[sel]
    Zr = Z_raw[v, u].astype(np.float32)

    Xr = (u - cx) * Zr / fx
    Yr = (v - cy) * Zr / fy
    P = np.column_stack([Xr, Yr, Zr])

    try:
        iqr = np.subtract(*np.percentile(Zr, [75, 25]))
        eps = 0.01 * iqr if iqr > 1e-3 else 0.005
    except IndexError:
        return 1.0

    best = {"cnt": 0}
    iters = 1500
    for _ in range(iters):
        a, b, c = P[rng.choice(P.shape[0], 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -np.dot(n, a)
        dist = np.abs(P @ n + d)
        inl = dist < eps
        cnt = inl.sum()
        if cnt > best["cnt"]:
            best.update(dict(cnt=cnt, n=n, d=d))
            if cnt > frac_good * P.shape[0]:
                break

    if best["cnt"] < frac_good * P.shape[0]:

        raise RuntimeError(f"RANSAC found only {best['cnt']} in-liers")

    alpha = abs(Hcam / best["d"])
    return alpha


def contact_band(S, O, h_up=12, max_gap=25):
    """
    Return bool mask with obstacle pixels that sit <max_gap px above sidewalk,
    grown h_up px upward.  None if no contact.
    """
    H, W = S.shape
    band = np.zeros_like(O, bool)
    cols = np.unique(np.where(O)[1])
    for x in cols:
        y_bot = np.where(O[:, x])[0].max()
        for dy in range(max_gap + 1):
            y = y_bot + dy
            if y >= H:
                break
            if S[y, x]:
                band[max(0, y_bot - h_up + 1) : y_bot + 1, x] = True
                break
    return band if band.any() else None


def clearance_px(S, band):
    """
    Returns (left_px, right_px, total_px) using distance-transform trick.
    """
    free = (S & ~band).astype(np.uint8)
    distL = cv2.distanceTransform(free, cv2.DIST_L1, 3)
    distR = cv2.distanceTransform(free[:, ::-1], cv2.DIST_L1, 3)[:, ::-1]

    rows = np.unique(np.where(band)[0])
    if rows.size == 0:
        return None
    l_px = r_px = t_px = 1e9
    for y in rows:
        xs = np.where(band[y])[0]
        uL, uR = xs.min(), xs.max()
        l_px = min(l_px, distL[y, uL])
        r_px = min(r_px, distR[y, uR])
        t_px = min(t_px, l_px + r_px + (uR - uL + 1))
    return l_px, r_px, t_px


def clearance_m(S, O, Z_raw, α, fx, cx, h_up=12, max_gap=25):
    band = contact_band(S, O, h_up=h_up, max_gap=max_gap)
    if band is None:
        return None

    out = clearance_px_scan(S, band)
    if out is None:
        return None
    l_px, r_px, t_px = out

    rows = np.where(band)[0]
    Z_band = α * Z_raw[rows].mean()
    scale = Z_band / fx
    return dict(L=l_px * scale, R=r_px * scale, total=t_px * scale)


def clearance_px_scan(S, band):
    """
    S     : bool (H,W)  – sidewalk mask
    band  : bool (H,W)  – contact strip of the obstacle
    return (left_px, right_px, total_px)  OR  None
    """
    H, W = S.shape
    rows = np.unique(np.where(band)[0])
    if rows.size == 0:
        return None

    l_px = r_px = t_px = 1e9
    for y in rows:
        walk_cols = np.where(S[y])[0]
        obs_cols = np.where(band[y])[0]
        if walk_cols.size < 2 or obs_cols.size == 0:
            continue  # skip degenerate rows

        u_walk_L, u_walk_R = walk_cols.min(), walk_cols.max()
        u_obs_L, u_obs_R = obs_cols.min(), obs_cols.max()

        l = u_obs_L - u_walk_L
        r = u_walk_R - u_obs_R
        if l < 0 or r < 0:
            continue

        l_px = min(l_px, l)
        r_px = min(r_px, r)
        t_px = min(t_px, l + r + (u_obs_R - u_obs_L + 1))

    return None if l_px == 1e9 else (l_px, r_px, t_px)


def analyse_frame(img_rgb, S, obstacles, Z_raw, FOV_deg=75, device="cuda"):

    H, W = S.shape
    fx = W / (2 * np.tan(np.radians(FOV_deg / 2)))
    cx = W / 2

    α = scale_from_ground(S, Z_raw, fx, fx, cx, H / 2)

    results = []
    for label, O in obstacles:
        c = clearance_m(S, O, Z_raw, α, fx, cx)
        if c:
            results.append(dict(label=label, **c))
    if not results:
        results.append(dict(label="none", L=-1.0, R=-1.0, total=-1.0))

    return results

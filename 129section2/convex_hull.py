import numpy as np
from scipy.spatial import ConvexHull

# ----------------------------------------
# 2D Graham Scan (using SciPy's ConvexHull)
# ----------------------------------------
def graham_scan(points):
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except:
        return points  # Fallback for edge cases

# ------------------
# Jarvis March (Gift Wrapping)
# ------------------
def jarvis_march(points):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    if len(points) < 3:
        return points
    
    hull = []
    point_on_hull = points[np.argmin(points[:, 0])]
    
    while True:
        hull.append(point_on_hull)
        next_point = points[0]
        for p in points:
            if np.array_equal(p, point_on_hull):
                continue
            val = cross(point_on_hull, next_point, p)
            if (np.array_equal(next_point, point_on_hull)) or (val < 0):
                next_point = p
        point_on_hull = next_point
        if np.array_equal(next_point, hull[0]):
            break
    return np.array(hull)

# -------------
# Quickhull
# -------------
def quickhull(points):
    def find_hull(points, p1, p2, hull):
        if len(points) == 0:
            return
        pts = points[np.cross(p2 - p1, points - p1) > 0]
        if len(pts) == 0:
            return
        farthest = pts[np.argmax(np.cross(p2 - p1, pts - p1))]
        hull.append(farthest)
        find_hull(pts, p1, farthest, hull)
        find_hull(pts, farthest, p2, hull)
    
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return points
    
    hull = []
    leftmost = points[np.argmin(points[:, 0])]
    rightmost = points[np.argmax(points[:, 0])]
    hull.extend([leftmost, rightmost])
    
    points = points[~np.isin(points, [leftmost, rightmost]).all(axis=1)]
    find_hull(points, leftmost, rightmost, hull)
    find_hull(points, rightmost, leftmost, hull)
    
    return np.array(hull)

# ------------------
# Monotone Chain (Andrew's Algorithm)
# ------------------
def monotone_chain(points):
    points = np.unique(points, axis=0)
    if len(points) < 3:
        return points
    
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    lower = []
    for p in points:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 0:
            upper.pop()
        upper.append(p)
    
    return np.vstack([lower[:-1], upper[:-1]])

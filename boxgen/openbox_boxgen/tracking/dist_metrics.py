"""Detection/track affinities: 3-D GIoU and corner-mean centre distance.

Both work on the KITTI camera-frame corners of ``Box3D.corners_camcoord`` (footprint in
the x-z plane, height along y; see ``box.py``), AB3DMOT style: Sutherland-Hodgman
clipping with a strict ``>`` inside test, areas via
``scipy.spatial.ConvexHull`` (``.volume`` is the 2-D area), footprint = corners 3, 2, 1, 0
(``[-5::-1]``) on columns ``[0, 2]``, height overlap between corner 0 (top) and corner 4
(bottom), and ``dist3d`` between the means of the eight corners, i.e. ``(x, y - h/2, z)``.
"""
import numpy as np
from scipy.spatial import ConvexHull


def polygon_clip(subjectPolygon, clipPolygon):
    """Sutherland-Hodgman clip of ``subjectPolygon`` by the convex ``clipPolygon`` (both CCW); None if empty."""
    def inside(p):
        """True when 2-D point ``p`` is strictly left of the CCW clip edge cp1 -> cp2."""
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        """[x, y] intersection of the lines through (cp1, cp2) and (s, e) from the enclosing scope."""
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def giou_3d(box_a, box_b):
    """Generalised 3-D IoU ``I/U - (C - U)/C`` of two ``Box3D``; ``C`` is the enclosing hull volume."""
    corners1 = box_a.corners_camcoord()
    corners2 = box_b.corners_camcoord()
    boxa_bot = corners1[-5::-1, [0, 2]]
    boxb_bot = corners2[-5::-1, [0, 2]]

    if np.linalg.norm(boxa_bot - boxb_bot) < 1e-5:  # near-identical footprints, as in the release run
        I_2D = ConvexHull(boxa_bot).volume
    else:
        inter_p = polygon_clip(boxa_bot, boxb_bot)
        # ConvexHull raises QhullError for a degenerate intersection (fewer than three
        # or collinear points, i.e. footprints touching along an edge); never hit in
        # the release run, so it is left unguarded.
        I_2D = ConvexHull(inter_p).volume if inter_p is not None else 0.0

    all_corners = np.vstack((boxa_bot, boxb_bot))
    convex_corners = all_corners[ConvexHull(all_corners).vertices]
    roll_pts = np.roll(convex_corners, -1, axis=0)  # shoelace area of the enclosing hull
    C_2D = np.abs(np.sum((convex_corners[:, 0] * roll_pts[:, 1] - convex_corners[:, 1] * roll_pts[:, 0]))) * 0.5

    overlap_height = max(0.0, min(corners1[0, 1], corners2[0, 1]) - max(corners1[4, 1], corners2[4, 1]))
    I_3D = I_2D * overlap_height
    U_3D = box_a.w * box_a.l * box_a.h + box_b.w * box_b.l * box_b.h - I_3D
    union_height = max(0.0, max(corners1[0, 1], corners2[0, 1]) - min(corners1[4, 1], corners2[4, 1]))
    C_3D = C_2D * union_height
    return I_3D / U_3D - (C_3D - U_3D) / C_3D


def dist3d(bbox1, bbox2):
    """Euclidean distance between the corner means (``(x, y - h/2, z)``) of two ``Box3D``."""
    c1 = np.average(bbox1.corners_camcoord(), axis=0)
    c2 = np.average(bbox2.corners_camcoord(), axis=0)
    return np.linalg.norm(c1 - c2)

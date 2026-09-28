"""3D box geometry of the OpenBox Waymo pseudo-label pipeline.

* ``BoundingBox``: the box record shared by all stages (``t`` = bottom centre,
  ``s`` = (l, w, h), ``r`` = yaw) with the rectangle fits (``make_box``), the
  centre <-> bottom-centre conversions (``load_gt`` / ``get_np_instance`` /
  ``get_o3d_instance``), pose transforms (``transform``) and the single-frame
  size snapping (``locate_bbox`` / ``locate_bbox_with_iou``).
* Rectangle fits on the ground plane: ``closeness_rectangle``, ``point_normal_rectangle``.
* Multi-frame extension of a box along the visible SDF-mesh faces:
  ``cal_surface`` -> ``refine_bbox`` (2D-IoU candidate vote through ``cal_bbox`` /
  ``proj_points``) -> ``refine_z``.
* Ground snapping of single-frame / deformable boxes: ``refine_z_bbox``, ``refine_z_deform``.
* Per-class rotated-BEV NMS: ``nms_gpu``.

Lidar / ego / scene frames are x forward, y left, z up; yaw is the rotation about
+z of the box length axis.  ``make_box`` runs the rectangle fits in a permuted
"camera" frame (x right, y down, z forward) and converts the result back, which
is why its yaw bookkeeping contains ``-= pi/2``, ``*= -1`` and ``pi/2 - ry``
steps: together they make ``make_box('given_angle', a).r == a`` up to
floating-point rounding.
"""
import copy
import logging
from typing import NamedTuple, Tuple

import numpy as np
import open3d as o3d
import pyquaternion as pyq
import torch  # before .box_2d_iou: the pcdet CUDA op needs torch loaded first
from scipy.spatial.transform import Rotation

from .box_2d_iou import class_wise_nms_gpu, iou_with_2d_gt

logger = logging.getLogger(__name__)

# Lidar (x fwd, y left, z up) -> camera-style axes (x right, y down, z fwd).  Not a
# calibration and not dataset specific: the rectangle fits are written for these
# axes and run on the camera [x, z] plane, i.e. lidar [-y, x].  The real camera
# conventions of the 2D-mask vote (camera frame -> OpenCV, image bounds) come from
# the config through ``SceneLoader.cameras``.
LIDAR_TO_CAMERA = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

# Added to the yaw before building the open3d rotation so the axis-angle vector is
# never exactly zero.  A numerical guard that is part of the released numerics, not
# a tunable -- hence not in the yaml.
YAW_EPSILON = 1e-10


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms_gpu(scene_idx, boxes, num_pts, classes, iou_threshold, pre_max_size):
    """Per-class rotated-BEV NMS with the number of points in a box as its score.

    ``boxes`` are N arrays ``[cx, cy, cz, l, w, h, yaw, ...]`` (first 7 columns
    used), ``num_pts`` their N integer scores (ties are broken by the unstable
    GPU sort) and ``classes`` their N class names.  At most ``pre_max_size``
    boxes per class enter NMS with BEV-IoU threshold ``iou_threshold``.

    Returns ``(boxes_after, classes_after)``: float32 (M, 8)
    ``[cx, cy, cz, l, w, h, yaw, num_points]`` concatenated class by class in
    ``np.unique(classes)`` (sorted) order, each block in descending score order,
    and the list of M class names; ``(None, None)`` when there is no box.
    """
    bbox_after_list = list()
    cls_after_list = list()
    boxes_for_nms = np.array(boxes)
    num_pts = np.array(num_pts)
    classes = np.array(classes)
    scores = torch.from_numpy(num_pts).squeeze(-1).to('cuda')
    if not scores.dim() == 0:
        if len(scores) == 0:
            logger.warning('scene %s: nms_gpu received no boxes', scene_idx)
            return None, None
    boxes_cuda = torch.from_numpy(boxes_for_nms[:, :7]).reshape(-1, 7).to('cuda')
    assert boxes_cuda.dtype == torch.float32, 'nms_gpu: the pcdet kernel reads float32 boxes'
    # Float index vector; ``keep_idx`` indexes into it and the result is cast back
    # with ``.long()``.  It is an identity map (exact for N < 2^24), kept as in the
    # release code.
    indices = torch.Tensor(range(boxes_cuda.shape[0])).to('cuda')

    for cls in np.unique(classes):
        cls_mask = torch.from_numpy(np.where(classes == cls)[0])
        cls_wise_boxes = boxes_cuda[cls_mask]

        # A single box gives a 0-dim score tensor after squeeze(-1).
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        cls_wise_scores = scores[cls_mask]
        cls_wise_class = classes[cls_mask]

        # A one-element torch index is consumed through __index__, so numpy returns a
        # np.str_ scalar here (multi-element indices give an ndarray).
        if isinstance(cls_wise_class, str):
            cls_wise_class = np.array([cls_wise_class])

        keep_idx = class_wise_nms_gpu(cls_wise_boxes, cls_wise_scores, iou_threshold, pre_max_size)
        selected = indices[keep_idx]

        bbox_after = cls_wise_boxes[selected.long()].cpu().detach().numpy()
        num_pts_after = cls_wise_scores[selected.long()].cpu().detach().numpy()
        cls_after = cls_wise_class[keep_idx.cpu().detach().numpy()]

        cls_bbox_after = np.concatenate((bbox_after, num_pts_after.reshape(-1, 1)), axis=1).astype(np.float32)
        cls_after_list.append(np.array(cls_after))
        bbox_after_list.append(np.array(cls_bbox_after))

    bbox_after = np.vstack(bbox_after_list)
    class_after_list = [item for sublist in cls_after_list for item in sublist]

    return bbox_after, class_after_list


# ---------------------------------------------------------------------------
# 2D boxes and image projection (candidate vote)
# ---------------------------------------------------------------------------

def cal_bbox(instance_id, bbox_2d_infos):
    """SAM 2D boxes of one instance: positional frame index -> list of ``{cam_loc: [x1, y1, x2, y2]}``.

    ``bbox_2d_infos`` maps frame key -> {str(instance id): list of
    {'cam_loc', 'bbox'}}.  The result key is the POSITION of the frame in
    ``bbox_2d_infos.values()`` (the index into the intrinsic / extrinsic lists),
    not the dict key.  This is aligned with the calibration lists only when
    ``agg_mask.json`` holds one key per frame, in frame order, starting at
    ``data.frame_start == 0`` (true for the release data).
    """
    refined_bbox_dict = dict()
    instance_id = str(int(instance_id))
    for frame_idx, bbox_2d_info in enumerate(bbox_2d_infos.values()):
        if instance_id in bbox_2d_info.keys():
            instance_bbox_info_list = bbox_2d_info[instance_id]
            refined_bbox_dict[frame_idx] = [{instance_bbox_info['cam_loc']: instance_bbox_info['bbox']}
                                            for instance_bbox_info in instance_bbox_info_list]

    return refined_bbox_dict


def _project_corners_to_image(corners_homogeneous, extrinsic, intrinsic_matrix, to_opencv, image_bounds):
    """Axis-aligned image box of the 8 ``[x, y, z, 1]`` box corners in one camera.

    ``to_opencv`` is the (4, 4) camera frame -> OpenCV frame matrix and
    ``image_bounds`` the ``[min_u, min_v, max_u, max_v]`` of this camera (both
    from :class:`~openbox_boxgen.io.CameraConventions`).  Returns
    ``[[min_u, min_v], [min_u, max_v], [max_u, min_v], [max_u, max_v]]``
    clipped to the image bounds; corners behind the camera are divided by their
    negative depth (mirrored) before clipping.
    """
    points_cam = np.dot(extrinsic, corners_homogeneous.T)
    points_cam = np.dot(to_opencv, points_cam).T[:, :3]

    points_cam_div = (points_cam / points_cam[:, 2].reshape(-1, 1))
    points_img = np.dot(intrinsic_matrix, points_cam_div.T).T

    min_u, max_u = min(points_img[:, 0]), max(points_img[:, 0])
    min_v, max_v = min(points_img[:, 1]), max(points_img[:, 1])
    corners_2d = [[min_u, min_v], [min_u, max_v], [max_u, min_v], [max_u, max_v]]

    bound_min_u, bound_min_v, bound_max_u, bound_max_v = image_bounds
    return [[min(max(u, bound_min_u), bound_max_u), min(max(v, bound_min_v), bound_max_v)]
            for u, v in corners_2d]


def proj_points(bbox_candidates, intrinsic_dict_list, extrinsic_dict_list, bbox_2d_list, cameras):
    """Project two candidate open3d boxes into every camera that has a 2D box for the instance.

    ``intrinsic_dict_list`` / ``extrinsic_dict_list`` are per-frame dicts
    cam_loc -> (3, 3) reshaped Waymo intrinsic 9-vector / (4, 4) extrinsic;
    ``bbox_2d_list`` (output of ``cal_bbox``) decides which frames and cameras
    are projected; ``cameras`` is the loader's :class:`~openbox_boxgen.io.CameraConventions`.
    Raises ``KeyError`` when a 2D box names a camera the frame has no calibration
    entry for (``io.py`` skips missing camera files); the release data is
    complete, so this is treated as a data error.
    Returns ``(projected_1, projected_2)``: frame index -> list
    (same camera order as ``bbox_2d_list[frame]``) of ``{cam_loc: 4 clipped
    [u, v] corners}`` for candidate 1 and 2.
    """
    frame_idx_list = list(bbox_2d_list.keys())
    projected_bbox_1_list_dict = dict()
    projected_bbox_2_list_dict = dict()

    corners_1 = np.array(bbox_candidates[0].get_box_points())
    corners_2 = np.array(bbox_candidates[1].get_box_points())

    corners_1 = np.concatenate([corners_1, np.ones((8, 1))], axis=1)
    corners_2 = np.concatenate([corners_2, np.ones((8, 1))], axis=1)

    for frame_idx in frame_idx_list:
        cam_loc_list = [list(bbox_2d.keys())[0] for bbox_2d in bbox_2d_list[frame_idx]]
        per_frame_pp1_list = list()
        per_frame_pp2_list = list()

        for cam_loc in cam_loc_list:
            extrinsic = extrinsic_dict_list[frame_idx][cam_loc]
            # Waymo packs [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]; after the (3, 3)
            # reshape f_u = [0, 0], f_v = [0, 1], c_u = [0, 2], c_v = [1, 0].
            # Distortion is ignored.
            intrinsic = intrinsic_dict_list[frame_idx][cam_loc]
            intrinsic_matrix = np.array([
                [intrinsic[0, 0], 0, intrinsic[0, 2]],
                [0, intrinsic[0, 1], intrinsic[1, 0]],
                [0, 0, 1]
            ])

            bounds = cameras.image_bounds[cam_loc]
            per_frame_pp1_list.append({cam_loc: _project_corners_to_image(corners_1, extrinsic, intrinsic_matrix,
                                                                          cameras.to_opencv, bounds)})
            per_frame_pp2_list.append({cam_loc: _project_corners_to_image(corners_2, extrinsic, intrinsic_matrix,
                                                                          cameras.to_opencv, bounds)})

        projected_bbox_1_list_dict[frame_idx] = per_frame_pp1_list
        projected_bbox_2_list_dict[frame_idx] = per_frame_pp2_list

    return projected_bbox_1_list_dict, projected_bbox_2_list_dict


def _pick_candidate_by_2d_iou(bbox_candidates, instance_id, bbox_2d_info,
                              intrinsic_dict_list, extrinsic_dict_list, cameras):
    """``(box, index)`` of the candidate whose projections overlap the SAM 2D boxes best.

    Ties go to candidate 2; an instance without any SAM 2D box scores 0 vs 0 and
    therefore also falls to candidate 2.
    """
    bbox_2d_list = cal_bbox(instance_id, bbox_2d_info)
    projected_points_1, projected_points_2 = proj_points(bbox_candidates, intrinsic_dict_list,
                                                         extrinsic_dict_list, bbox_2d_list, cameras)
    chosen_index = 0 if iou_with_2d_gt(projected_points_1, projected_points_2, bbox_2d_list) == 1 else 1
    return bbox_candidates[chosen_index], chosen_index


# ---------------------------------------------------------------------------
# Multi-frame box extension along the visible mesh faces
# ---------------------------------------------------------------------------

def cal_surface(sub_mesh, thres_parallel, thres_num_normal_vectors):
    """Ascending ids of the box side faces that the instance sub-mesh shows as visible.

    ``sub_mesh`` (open3d TriangleMesh with triangle normals) must already be
    rotated into the yaw-unrotated box frame, whose face ids are::

                       (2)
                        x
                        ^
        (1)  y <------- + ------->  (0)
                        v
                       (3)

    A triangle votes for a face when ``dot(normal, face axis) > thres_parallel``;
    a face is visible when its votes exceed ``thres_num_normal_vectors``.
    """
    surface_list = []
    face_axes = np.array([[0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0]])  # outward axis of face 0..3
    triangle_normals = np.asarray(sub_mesh.triangle_normals)

    for face_id in range(4):
        num_votes = 0
        # Per-triangle np.dot, not one matrix product: the rounding differs.
        for normal in triangle_normals:
            if np.dot(normal, face_axes[face_id]) > thres_parallel:
                num_votes += 1

        if num_votes > thres_num_normal_vectors:
            surface_list.append(face_id)

    return surface_list


class _CornerExtension(NamedTuple):
    """How to grow a yaw-unrotated box from one corner (1 = +x/-y, 2 = +x/+y,
    3 = -x/+y, 4 = -x/-y) to the class-prior size.

    ``length_dir`` / ``width_dir`` are the unit steps from ``start_corner``
    towards the corner that ends the length / width edge; the fallback corner is
    used instead when the current extent already reaches the prior.  ``rotation``
    keys ``_CANDIDATE_ROTATIONS``; ``swap_lw`` compares the current extents as
    (w, l) instead of (l, w).
    """
    start_corner: int
    length_dir: Tuple[int, int]
    width_dir: Tuple[int, int]
    length_fallback_corner: int
    width_fallback_corner: int
    rotation: str
    swap_lw: bool


_CANDIDATE_ROTATIONS = {
    'identity': np.eye(3),
    'plus_90': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    # Negated +90 matrix (det = -1) of the length/width-swapped candidate; only
    # R[1,0]/R[0,0] and the symmetric corner set are consumed downstream.
    'minus_90_reflected': -np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
}

# (candidate 1, candidate 2) of the 2D vote per ascending tuple of visible
# faces.  Missing keys are the invalid cases: faces 2 or 3 alone, two opposite
# faces, none or all four; ``refine_bbox`` then keeps the initial box.  A lone
# face 2 or 3 does not occur in practice: the closeness fit puts the longer
# extent on x, so the spread of a single visible face becomes the length axis
# and that face is a long side (0 or 1).  All four visible (a mesh that wraps
# the whole object) is the common unhandled case and needs no growth.
#
# Yaw-unrotated box seen from above (x = length axis, y = width axis; face ids
# as in ``cal_surface``, corner ids as in ``refine_bbox``)::
#
#                    face 2 (+x, short end)
#              pt2 ---------------------- pt1          x
#               |                          |           ^
#   face 1 (+y) |                          | face 0    |
#   long side   |                          | (-y)      y <----+
#               |                          |
#              pt3 ---------------------- pt4
#                    face 3 (-x, short end)
#
# The mesh sits on the visible faces, so both candidates start at the corner
# the visible faces share (``start_corner``) and grow AWAY from them; the
# visible faces stay where the mesh is.  The two entries are the two 90-degree
# hypotheses that the 2D vote decides between:
#
#   candidate 1  the visible long side is really the object's short end:
#                ``length_dir`` points away from that side (along y),
#                ``width_dir`` runs along it (along x); ``rotation`` is the
#                swapped-axes matrix and ``swap_lw`` tests the extents as (w, l).
#   candidate 2  the visible long side is the long side: ``length_dir`` along
#                x, ``width_dir`` away from the side (along y); ``identity``.
#
# The three-face rows hold the same two hypotheses, but not always in this
# order, and never swap the extent test.
#
# Example, key (0, 2): faces 0 (-y) and 2 (+x) visible, shared corner pt1.
#
#              pt2 ====================== pt1     == mesh on face 2
#               .                          #
#               .        grows this        #      #  mesh on face 0
#               .        way (-x, +y)      #
#              pt3 . . . . . . . . . . .  pt4     .. hidden faces, moved
#
#   candidate 2: from pt1 go -x by ``stat_l`` (or stop at pt4 when the current
#                l already reaches it) and +y by ``stat_w`` (or stop at pt2).
#   candidate 1: from pt1 go +y by ``stat_l`` and -x by ``stat_w``, i.e. the
#                same corner with the axes swapped.
#
# Each ``_CornerExtension`` row below is exactly that: start corner, the two
# unit directions, the two "already big enough" fallback corners, the rotation
# tag and the swap flag.
_EXTENSIONS = {
    (0,): (_CornerExtension(1, (0, 1), (-1, 0), 2, 4, 'minus_90_reflected', True),
           _CornerExtension(1, (-1, 0), (0, 1), 4, 2, 'identity', False)),
    # Fallback corners of candidate 2 are pt1 / pt3, swapped relative to the
    # mirrored face-0 case (not a typo).
    (1,): (_CornerExtension(2, (0, -1), (-1, 0), 1, 3, 'minus_90_reflected', True),
           _CornerExtension(2, (-1, 0), (0, -1), 1, 3, 'identity', False)),
    (0, 2): (_CornerExtension(1, (0, 1), (-1, 0), 2, 4, 'plus_90', True),
             _CornerExtension(1, (-1, 0), (0, 1), 4, 2, 'identity', False)),
    (0, 3): (_CornerExtension(4, (0, 1), (1, 0), 3, 1, 'minus_90_reflected', True),
             _CornerExtension(4, (1, 0), (0, 1), 1, 3, 'identity', False)),
    (1, 2): (_CornerExtension(2, (0, -1), (-1, 0), 1, 3, 'minus_90_reflected', True),
             _CornerExtension(2, (-1, 0), (0, -1), 3, 1, 'identity', False)),
    (1, 3): (_CornerExtension(3, (0, -1), (1, 0), 4, 2, 'plus_90', True),
             _CornerExtension(3, (1, 0), (0, -1), 2, 4, 'identity', False)),
    # Three visible faces (keyed by the visible ones; the comment names the
    # invisible face): both candidates compare the current (l, w) unswapped.
    (1, 2, 3): (_CornerExtension(2, (0, -1), (-1, 0), 1, 3, 'minus_90_reflected', False),  # face 0 hidden
                _CornerExtension(2, (-1, 0), (0, -1), 3, 1, 'identity', False)),
    (0, 2, 3): (_CornerExtension(1, (0, 1), (-1, 0), 2, 4, 'minus_90_reflected', False),  # face 1 hidden
                _CornerExtension(1, (-1, 0), (0, 1), 4, 2, 'identity', False)),
    (0, 1, 3): (_CornerExtension(3, (1, 0), (0, -1), 2, 4, 'identity', False),  # face 2 hidden
                _CornerExtension(3, (0, -1), (1, 0), 4, 2, 'minus_90_reflected', False)),
    (0, 1, 2): (_CornerExtension(1, (-1, 0), (0, 1), 4, 2, 'identity', False),  # face 3 hidden
                _CornerExtension(1, (0, 1), (-1, 0), 2, 4, 'minus_90_reflected', False)),
}


def _extend_from_corner(spec, corners, current_l, current_w, stat_l, stat_w, center_height, height):
    """Yaw-unrotated open3d box grown as ``spec`` says from ``corners`` (id -> [x, y]) towards the prior."""
    start_point = corners[spec.start_corner]
    length_dir = np.array(spec.length_dir)
    width_dir = np.array(spec.width_dir)
    compared_l, compared_w = (current_w, current_l) if spec.swap_lw else (current_l, current_w)

    new_length_point = start_point + length_dir * stat_l if compared_l < stat_l else corners[spec.length_fallback_corner]
    new_width_point = start_point + width_dir * stat_w if compared_w < stat_w else corners[spec.width_fallback_corner]
    new_center = [np.mean([new_length_point[0], new_width_point[0]]),
                  np.mean([new_length_point[1], new_width_point[1]]),
                  center_height]
    new_length = np.abs(np.linalg.norm(np.array(new_length_point) - np.array(start_point)))
    new_width = np.abs(np.linalg.norm(np.array(new_width_point) - np.array(start_point)))
    return o3d.geometry.OrientedBoundingBox(np.array(new_center), _CANDIDATE_ROTATIONS[spec.rotation],
                                            np.array([new_length, new_width, height]))


def refine_bbox(cfg, init_box, stat_box_size, sub_mesh, instance_id, intrinsic_dict_list,
                extrinsic_dict_list, bbox_2d_info, cameras, scene_idx, full_pcs):
    """Extend a multi-frame box to the class-prior size along its visible mesh faces.

    ``cfg`` is the full config: ``cfg.multi_frame`` (face voting, height gate) and
    ``cfg.box_fit`` (aspect-ratio gate, ground quantile) are read.

    Corner / axis sketch of the yaw-unrotated box (``pt`` ids of ``_EXTENSIONS``)::

        pt2 ------- pt1        x
         |           |         ^
         |           |         |
        pt3 ------- pt4   y <--+

    Steps: unrotate the box and the sub-mesh by the initial yaw -> visible faces
    (``cal_surface`` with ``multi_frame.thres_parallel`` / ``thres_num_normal_vectors``)
    -> two candidates per case, each grown from the corner shared by the visible
    faces towards the class prior ``stat_box_size`` where the current extent is
    smaller -> aspect-ratio gate -> 2D-IoU vote against the SAM 2D boxes
    (projected with ``cameras``, the loader's :class:`~openbox_boxgen.io.CameraConventions`)
    -> ground snap on ``full_pcs`` (``refine_z`` with ``multi_frame.z_refine_threshold``).

    ``init_box`` (open3d ``OrientedBoundingBox``) is mutated in place in the
    two-opposite-faces case.  Returns the refined ``OrientedBoundingBox``.
    """
    stat_l, stat_w, _ = stat_box_size

    init_center = init_box.get_center()
    init_rotation = init_box.R
    # rotate() without ``center`` rotates about the box's own centre; np.linalg.inv
    # (not the transpose) because the two differ in the last bits.
    unrotated_box = copy.deepcopy(init_box)
    unrotated_box = unrotated_box.rotate(np.linalg.inv(init_rotation))

    l, w, h = unrotated_box.extent
    unrotated_mesh = copy.deepcopy(sub_mesh)
    unrotated_mesh = unrotated_mesh.rotate(np.linalg.inv(init_rotation), center=init_center)
    bbox_corners = np.array(unrotated_box.get_box_points())
    center_height = unrotated_box.get_center()[2]

    # xy of the four top-view corners of the yaw-unrotated box (ids of the
    # sketch above / ``_EXTENSIONS``), taken as the extremes of the 8 open3d
    # corner points so the z coordinate drops out.
    corners = {
        1: [max(bbox_corners[:, 0]), min(bbox_corners[:, 1])],  # +x, -y  where faces 2 and 0 meet
        2: [max(bbox_corners[:, 0]), max(bbox_corners[:, 1])],  # +x, +y  where faces 2 and 1 meet
        3: [min(bbox_corners[:, 0]), max(bbox_corners[:, 1])],  # -x, +y  where faces 3 and 1 meet
        4: [min(bbox_corners[:, 0]), min(bbox_corners[:, 1])],  # -x, -y  where faces 3 and 0 meet
    }

    surface_list = cal_surface(unrotated_mesh, cfg.multi_frame.thres_parallel,
                               cfg.multi_frame.thres_num_normal_vectors)
    extensions = _EXTENSIONS.get(tuple(surface_list))

    if extensions is None:
        candidate_1 = init_box
        candidate_2 = init_box
        if len(surface_list) == 2:
            # Two opposite faces: both candidates are the same init_box object,
            # rotated twice in place (R becomes yaw^3; the caller's box is mutated).
            logger.info('scene %s instance %s: invalid surface case %s', scene_idx, instance_id, surface_list)
            candidate_1.rotate(init_rotation, center=init_center)
            candidate_2.rotate(init_rotation, center=init_center)
    else:
        candidate_1 = _extend_from_corner(extensions[0], corners, l, w, stat_l, stat_w, center_height, h)
        candidate_1.rotate(init_rotation, center=init_center)
        candidate_2 = _extend_from_corner(extensions[1], corners, l, w, stat_l, stat_w, center_height, h)
        candidate_2.rotate(init_rotation, center=init_center)

    # Aspect-ratio gate: when exactly one candidate is elongated (l/w > gate) and the
    # other is not (l/w < gate) it wins outright; only the ambiguous case (both,
    # neither, or a ratio exactly at the gate) goes to the 2D vote against the SAM 2D boxes.
    gate = cfg.box_fit.aspect_ratio_gate
    l_1, w_1, _ = candidate_1.extent
    l_2, w_2, _ = candidate_2.extent
    if l_1 / w_1 < gate < l_2 / w_2:
        chosen_box = candidate_2
    elif l_2 / w_2 < gate < l_1 / w_1:
        chosen_box = candidate_1
    else:
        chosen_box, _ = _pick_candidate_by_2d_iou([candidate_1, candidate_2], instance_id, bbox_2d_info,
                                                  intrinsic_dict_list, extrinsic_dict_list, cameras)
    bbox_candidates = [chosen_box]

    try:
        bbox_candidates = refine_z(bbox_candidates, full_pcs, stat_box_size,
                                   cfg.multi_frame.z_refine_threshold, cfg.box_fit.ground_quantile)
    except Exception as error:  # noqa: BLE001  (any failure leaves the box un-snapped)
        logger.warning('scene %s instance %s: height refinement failed (%s: %s), box kept',
                       scene_idx, instance_id, type(error).__name__, error)

    return bbox_candidates[0]


# ---------------------------------------------------------------------------
# Ground snapping
# ---------------------------------------------------------------------------

def _ground_z(full_pcs, box_center, size, ground_quantile):
    """z of the local ground under a box: the ``int(ground_quantile * N)``-th largest z of the
    ``full_pcs`` points within half the class-prior (l, w) diagonal of ``box_center``
    (0.99: roughly the lowest 1 %).  Raises ``IndexError`` when no point lies in the region.
    """
    region_radius = np.linalg.norm(size[:2]) / 2
    region_mask = np.linalg.norm(full_pcs[:, :2] - box_center[:2], axis=1) < region_radius
    region_pcs = full_pcs[region_mask]
    zs = region_pcs[:, 2]
    zs.sort()  # in place on a column of the boolean-mask copy, so full_pcs is untouched
    zs = zs[::-1]
    return zs[int(zs.shape[0] * ground_quantile)]


def refine_z(bbox_candidates, full_pcs, size, z_refine_threshold, ground_quantile):
    """Snap open3d boxes to the class-prior height ``size[2]`` resting on the local ground.

    The ground is ``_ground_z`` of the ``full_pcs`` points around
    ``bbox_candidates[0]``.
    Candidates with ``extent[2] > size[2] * z_refine_threshold`` are skipped; the
    input list is returned unchanged when every candidate was skipped.  Raises
    ``IndexError`` when no point lies in the region.
    """
    new_bbox_candidates = list()
    for bbox in bbox_candidates:
        if bbox.extent[2] > size[2] * z_refine_threshold:
            continue
        z_min = _ground_z(full_pcs, bbox_candidates[0].get_center(), size, ground_quantile)

        new_center = copy.deepcopy(bbox.center)
        new_extent = copy.deepcopy(bbox.extent)
        new_center[2] = z_min + size[2]/2
        new_extent[2] = size[2]
        new_bbox = o3d.geometry.OrientedBoundingBox(np.array(new_center), bbox.R, np.array(new_extent))
        new_bbox_candidates.append(new_bbox)
    if len(new_bbox_candidates) == 0:
        new_bbox_candidates = bbox_candidates
    return new_bbox_candidates


def refine_z_bbox(bbox_candidates, full_pcs, size, ground_quantile, scene_idx, instance_id):
    """Snap single-frame ``BoundingBox`` es (bottom z = ground, height = ``size[2]``).

    The ground is estimated as in ``refine_z`` (``ground_quantile``) around ``bbox_candidates[0].t``;
    no box is skipped.  Returns ``None`` when no point lies in the region (the
    caller drops the instance).
    """
    new_bbox_candidates = list()
    for bbox in bbox_candidates:
        try:
            z_min = _ground_z(full_pcs, bbox_candidates[0].t, size, ground_quantile)
        except IndexError:
            logger.warning('scene %s instance %s: no points around the box for ground refinement, instance dropped',
                           scene_idx, instance_id)
            return None
        new_center = copy.deepcopy(bbox.t)
        new_extent = copy.deepcopy(bbox.s)
        new_center[2] = z_min
        new_extent[2] = size[2]
        new_bbox = BoundingBox(new_center, new_extent, bbox.r)
        new_bbox_candidates.append(new_bbox)
    return new_bbox_candidates


def refine_z_deform(bbox, full_pcs, size, ground_quantile):
    """Snap a deformable-class ``BoundingBox`` (bottom z = ground, height = ``size[2]``).

    The ground is estimated as in ``refine_z`` (``ground_quantile``) around ``bbox.t``; no box is
    skipped and the input box is returned unchanged when no point lies in the
    region.  The bottom z is ``(ground + h/2) - h/2``, a float round trip that
    is part of the numerics.
    """
    try:
        z_min = _ground_z(full_pcs, bbox.t, size, ground_quantile)
    except IndexError:  # no point around the box
        return bbox

    new_center = copy.deepcopy(bbox.t)
    new_extent = copy.deepcopy(bbox.s)
    new_center[2] = z_min + size[2]/2
    new_extent[2] = size[2]
    new_bbox = copy.deepcopy(bbox)
    new_bbox.t = np.array(new_center)
    new_bbox.t[2] -= size[2] / 2
    new_bbox.s = np.array(new_extent)
    return new_bbox


# ---------------------------------------------------------------------------
# Rectangle fits (camera-frame [x, z] plane = lidar [-y, x])
# ---------------------------------------------------------------------------

def _project_onto_yaw(points_2d, angle):
    """Rotate (N, 2) points into the frame whose x axis lies along yaw ``angle``.

    Returns ``(components, projection, min_x, max_x, min_y, max_y)``: the 2x2
    rotation ``[[cos, sin], [-sin, cos]]``, ``points_2d @ components.T`` and the
    per-axis extents of that projection.
    """
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = points_2d @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
    return components, projection, min_x, max_x, min_y, max_y


def _rectangle_from_angle(points_2d, angle):
    """``(corners (4, 2), angle, area)`` of the bounding rectangle of ``points_2d`` at yaw ``angle``.

    Corner order [max_x, min_y], [min_x, min_y], [min_x, max_y], [max_x, max_y]
    in the rotated frame, mapped back to the input frame: corners[0]-corners[1]
    is the length edge and corners[0]-corners[-1] the width edge.
    """
    components, _, min_x, max_x, min_y, max_y = _project_onto_yaw(points_2d, angle)
    area = (max_x - min_x) * (max_y - min_y)
    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area


def _flip_if_wider_than_long(points_2d, angle):
    """``angle + pi/2`` when the rectangle at ``angle`` is wider than long, else ``angle``."""
    _, _, min_x, max_x, min_y, max_y = _project_onto_yaw(points_2d, angle)
    if (max_x - min_x) < (max_y - min_y):
        return angle + np.pi / 2
    return angle


def closeness_rectangle(cluster_ptc, fit_cfg):
    """Closeness-to-edge rectangle fit (Zhang et al. 2017) of (N, 2) points by a brute-force yaw sweep.

    Every yaw in [0, 90] degrees (``fit_cfg.closeness_yaw_step_deg`` steps) is
    scored by the summed inverse point-to-nearest-edge distance (clamped below at
    ``fit_cfg.closeness_min_edge_dist``); ties keep the first angle.  Returns
    ``(corners, angle, area)`` as ``_rectangle_from_angle`` with the angle
    flipped by +90 degrees so that the x extent is >= the y extent.
    """
    max_beta = -float('inf')
    choose_angle = None
    step = fit_cfg.closeness_yaw_step_deg
    for angle in np.arange(0, 90 + step, step):
        angle = angle / 180. * np.pi
        _, projection, min_x, max_x, min_y, max_y = _project_onto_yaw(cluster_ptc, angle)
        dist_x = np.vstack((projection[:, 0] - min_x, max_x - projection[:, 0])).min(axis=0)
        dist_y = np.vstack((projection[:, 1] - min_y, max_y - projection[:, 1])).min(axis=0)
        beta = np.vstack((dist_x, dist_y)).min(axis=0)
        beta = np.maximum(beta, fit_cfg.closeness_min_edge_dist)
        beta = 1 / beta
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
    angle = _flip_if_wider_than_long(cluster_ptc, choose_angle)
    return _rectangle_from_angle(cluster_ptc, angle)


def point_normal_rectangle(src, fit_cfg):
    """Rectangle whose yaw is the dominant direction of the vertical-surface normals of ``src``.

    ``src`` is an open3d PointCloud in the camera-style frame with normals; a
    normal is vertical when ``|normal_y| < fit_cfg.vertical_normal_max_abs_y``.
    The yaw is the lower edge of the most populated
    ``fit_cfg.normal_hist_bin_deg``-degree bin of the normal directions folded
    into [0, pi/2) (first maximum wins), flipped by +90 degrees when the
    rectangle is wider than long.  Returns ``(corners, angle, area)`` on the
    [x, z] plane.  With 2.5-degree bins the 36 bin edges cover [0, 87.5)
    degrees only, so normals folded into [87.5, 90] degrees cast no vote
    (release behaviour).
    """
    src = src.normalize_normals()
    normals = np.array(src.normals)
    normals = normals[np.abs(normals[:, 1]) < fit_cfg.vertical_normal_max_abs_y]
    if len(normals) == 0:
        normals = np.array([[1, 0, 0]])
    normals = normals[:, [0, 2]]
    angles = np.arctan2(normals[:, 1], normals[:, 0])
    angles = np.where(angles < 0, angles + np.pi, angles)
    angles = np.where(angles >= np.pi / 2, angles - np.pi / 2, angles)
    # written as pi / (180 / deg) so that 2.5 degrees gives exactly the pi / 72 of the release
    bins = np.arange(0, np.pi / 2, np.pi / (180.0 / fit_cfg.normal_hist_bin_deg))
    hist, _ = np.histogram(angles, bins=bins)
    angle = bins[np.argmax(hist)]
    points_2d = np.array(src.points)[:, [0, 2]]
    angle = _flip_if_wider_than_long(points_2d, angle)
    return _rectangle_from_angle(points_2d, angle)


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

def face_center(obj):
    """Side-face centres of a ``BoundingBox`` and their visibility from the origin.

    Returns ``(face_centers, dot_product)``: (4, 2) xy of the faces at yaw
    ``r``, ``r + pi/2``, ``r + pi``, ``r + 3pi/2`` (front, left, back, right)
    and the (4,) dot product between the unit outward direction of each face
    and the unit direction of its centre from the origin (the sensor is assumed
    at the origin; >= 0 means invisible).
    """
    face_centers = np.array([[obj.t[0] + obj.s[0]/2 * np.cos(obj.r), obj.t[1] + obj.s[0]/2 * np.sin(obj.r)],
                             [obj.t[0] + obj.s[1]/2 * np.cos(obj.r + np.pi/2), obj.t[1] + obj.s[1]/2 * np.sin(obj.r + np.pi/2)],
                             [obj.t[0] + obj.s[0]/2 * np.cos(obj.r + np.pi), obj.t[1] + obj.s[0]/2 * np.sin(obj.r + np.pi)],
                             [obj.t[0] + obj.s[1]/2 * np.cos(obj.r + 3*np.pi/2), obj.t[1] + obj.s[1]/2 * np.sin(obj.r + 3*np.pi/2)]])

    face_looking = face_centers - obj.t[:2]
    nface_looking = face_looking / np.linalg.norm(face_looking, axis=1).reshape(-1, 1)
    nface_centers = face_centers / np.linalg.norm(face_centers, axis=1).reshape(-1, 1)
    dot_product = np.sum(nface_looking*nface_centers, axis=1)

    return face_centers, dot_product


class BoundingBox:
    """Oriented 3D box: ``t`` = (3,) bottom centre, ``s`` = (3,) [l, w, h], ``r`` = yaw about +z.

    ``bbox.t`` / ``.s`` / ``.r`` are read and written by every stage;
    ``get_np_instance`` / ``load_gt`` convert to / from the centre-anchored
    7-vector used on disk.
    """

    def __init__(self, t=None, s=None, r=0):
        """t: (3,) bottom centre; s: (3,) [l, w, h]; r: float yaw (defaults: zero arrays / 0)."""
        # Fresh int arrays per instance; every fit or load replaces them before use.
        self.t = np.array([0, 0, 0]) if t is None else t
        self.s = np.array([0, 0, 0]) if s is None else s
        self.r = r

    def make_box(self, pcd, fit_method, fit_cfg, angle=None):
        """Fit this box to an open3d PointCloud in the lidar frame; returns ``self``.

        ``fit_method`` is ``'closeness_to_edge'``, ``'point_normal'`` (normals
        estimated here with ``fit_cfg.normal_search_radius`` / ``normal_max_nn``)
        or ``'given_angle'`` (rectangle at the lidar-frame yaw ``angle``; the box
        then has ``r == angle``); ``fit_cfg`` is the ``box_fit`` config section.
        ``h`` spans the min..max height of the points.  Raises ``ValueError`` on
        an empty cloud.
        """
        points = np.array(pcd.points)
        if len(points) == 0:
            raise ValueError('make_box: empty point cloud')
        cam_coord_pcd = points @ LIDAR_TO_CAMERA.T
        if fit_method == 'closeness_to_edge':
            corners, ry, _ = closeness_rectangle(cam_coord_pcd[:, [0, 2]], fit_cfg)
        elif fit_method == 'point_normal':
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(cam_coord_pcd)
            src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=fit_cfg.normal_search_radius, max_nn=fit_cfg.normal_max_nn))
            src.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
            corners, ry, _ = point_normal_rectangle(src, fit_cfg)
        elif fit_method == 'given_angle':
            if angle is None:
                raise ValueError('angle is None')
            angle -= np.pi / 2
            corners, ry, _ = _rectangle_from_angle(cam_coord_pcd[:, [0, 2]], angle)
        else:
            raise NotImplementedError(fit_method)
        ry *= -1
        l = np.linalg.norm(corners[0] - corners[1])
        w = np.linalg.norm(corners[0] - corners[-1])
        c = (corners[0] + corners[2]) / 2
        bottom = cam_coord_pcd[:, 1].max()
        h = bottom - cam_coord_pcd[:, 1].min()

        self.t = np.array([c[0], bottom, c[1]])
        self.t = LIDAR_TO_CAMERA.T @ self.t
        self.s = np.array([l, w, h])
        self.r = np.pi / 2 - ry
        return self

    def load_gt(self, arr):
        """Load from a (deep-copied) centre-anchored 7-vector ``[cx, cy, cz, l, w, h, yaw]``; returns ``self``."""
        arr = copy.deepcopy(arr)
        self.t = arr[:3]
        self.s = arr[3:6]
        self.t[2] -= self.s[2] / 2
        self.r = arr[6]
        return self

    def get_o3d_instance(self):
        """The box as an open3d ``OrientedBoundingBox`` centred at ``t + (0, 0, h/2)``.

        ``R = Rz(r + YAW_EPSILON)``: the epsilon keeps the axis-angle vector
        non-zero and is part of the numerics.
        """
        center = copy.deepcopy(self.t)
        center[2] += self.s[2] / 2
        axis_angles = np.array([0, 0, self.r + YAW_EPSILON])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        return o3d.geometry.OrientedBoundingBox(center, rot, self.s)

    def get_np_instance(self):
        """Centre-anchored 7-vector ``[cx, cy, cz, l, w, h, yaw]`` (``cz = t_z + h/2``)."""
        t = copy.deepcopy(self.t)
        t[2] += self.s[2] / 2
        return np.concatenate((t, self.s, [self.r]), axis=0)

    def transform(self, transformation_matrix):
        """Apply the INVERSE of the 4x4 ``transformation_matrix`` in place; returns ``self``.

        Only the yaw of the inverse (``arctan2(M[1,0], M[0,0])``) rotates the box;
        the translation is composed through the homogeneous box centre so that
        the centre lands where the full inverse would put it.  Callers pass
        ``np.linalg.inv(pose)`` to apply ``pose``.
        """
        tr_matrix = np.linalg.inv(transformation_matrix)
        new_tr_mat = np.eye(4)
        rotation = np.arctan2(tr_matrix[1, 0], tr_matrix[0, 0])
        new_tr_mat[:3, :3] = Rotation.from_euler('z', rotation).as_matrix()
        homo_t = np.concatenate((copy.deepcopy(self.t + [0, 0, self.s[2]/2]), [1]))
        new_tr_mat[:3, 3] = (tr_matrix @ homo_t)[:3] - (new_tr_mat @ homo_t)[:3]
        self.t = (new_tr_mat @ homo_t)[:3] - [0, 0, self.s[2]/2]
        self.r += rotation
        return self

    def locate_bbox(self, t, s, r, bbox_size):
        """Grow the box to the class prior ``bbox_size`` away from its faces visible from the origin.

        ``t`` is used IN PLACE (``box.locate_bbox(box.t, ...)`` shifts the
        caller's array).  One visible face: shift along the face normal by
        ``(prior - current) / 2`` (even when negative); two adjacent visible
        faces: shift along both; other patterns only resize.  ``s`` becomes
        ``max(s, prior)`` in l and w; ``r`` is unchanged.  Returns ``self``.
        ``t``, ``s``, ``r`` are this box's own fields at every call site; ``s``
        and ``r`` only drive the visibility test, the size update reads ``self.s``.
        """
        obj = BoundingBox(t, s, r)

        face_centers, dot_product = face_center(obj)

        invis = dot_product >= 0
        vis = np.logical_not(invis)

        if np.sum(invis) == 3:
            if vis[0] or vis[2]:
                direction = (obj.t[:2] - face_centers[vis])
                normalized_direction = direction / np.linalg.norm(direction)
                l = bbox_size[0] / 2 - obj.s[0] / 2
                obj.t[:2] += (normalized_direction * l).reshape(-1)
            else:
                direction = (obj.t[:2] - face_centers[vis])
                normalized_direction = direction / np.linalg.norm(direction)
                l = bbox_size[1] / 2 - obj.s[1] / 2
                obj.t[:2] += (normalized_direction * l).reshape(-1)
        elif np.sum(invis) == 2:
            # Adjacent faces share a corner.  Two OPPOSITE visible faces (viewpoint
            # inside the box's slab) make cor the box centre and the shift below
            # degenerate -- kept as in the release code.
            cor = np.sum(face_centers[vis], axis=0) - obj.t[:2]
            indices = np.where(vis)[0]
            # Even face ids (0, 2) are the length ends: ind1 is the length-end face.
            if indices[0] % 2 == 0:
                ind1, ind2 = indices[0], indices[1]
            else:
                ind1, ind2 = indices[1], indices[0]
            dir1 = face_centers[ind1] - cor
            dir2 = face_centers[ind2] - cor
            dir1 = dir1 / np.linalg.norm(dir1) * (bbox_size[1] / 2 - obj.s[1] / 2)
            dir2 = dir2 / np.linalg.norm(dir2) * (bbox_size[0] / 2 - obj.s[0] / 2)
            obj.t[:2] = obj.t[:2] + dir1 + dir2
        else:
            logger.debug('locate_bbox: %d invisible faces, box only resized', int(np.sum(invis)))
        self.t = obj.t
        new_size = np.array([max(self.s[0], bbox_size[0]), max(self.s[1], bbox_size[1]), self.s[2]])
        self.s = new_size
        self.r = obj.r
        return self

    def locate_bbox_with_iou(self, bbox_size, instance_id, bbox_2d_info, intrinsic_dict_list,
                             extrinsic_dict_list, cameras):
        """``locate_bbox`` for both 90-degree hypotheses, keeping the better 2D-IoU one; returns ``self``.

        The box must be in the frame the extrinsics expect; ``cameras`` is the
        loader's :class:`~openbox_boxgen.io.CameraConventions`.  Two quirks are part
        of the numerics: the candidates are projected with ``t`` (the BOTTOM
        centre) as the open3d box centre, and the yaw is recovered with
        ``pyquaternion.Quaternion(matrix=R).angle``, which wraps it into (-pi, pi]
        and, for wrapped yaws in (-pi/2, 0), returns the mirrored value ``-yaw``
        (pyquaternion flips the axis to -z there); yaws in (-pi, -pi/2] keep
        their sign.  Position and size come from the correctly rotated candidate,
        so the mirrored case is a yaw-only quirk of the released boxes.
        """
        cand1, cand2 = copy.deepcopy(self), copy.deepcopy(self)
        cand1 = cand1.locate_bbox(cand1.t, cand1.s, cand1.r, bbox_size)
        r1 = pyq.Quaternion(axis=[0, 0, 1], angle=cand1.r).rotation_matrix
        cand1 = o3d.geometry.OrientedBoundingBox(cand1.t, r1, cand1.s)
        cand2.r += np.pi / 2
        cand2.s[0], cand2.s[1] = cand2.s[1], cand2.s[0]
        cand2 = cand2.locate_bbox(cand2.t, cand2.s, cand2.r, bbox_size)
        r2 = pyq.Quaternion(axis=[0, 0, 1], angle=cand2.r).rotation_matrix
        cand2 = o3d.geometry.OrientedBoundingBox(cand2.t, r2, cand2.s)
        bbox_candidates = [cand1, cand2]

        chosen_box, chosen_index = _pick_candidate_by_2d_iou(bbox_candidates, instance_id, bbox_2d_info,
                                                             intrinsic_dict_list, extrinsic_dict_list, cameras)
        chosen_rotation = r1 if chosen_index == 0 else r2
        angle = pyq.Quaternion(matrix=chosen_rotation).angle
        self.t = chosen_box.center
        self.s = chosen_box.extent
        self.r = angle
        return self

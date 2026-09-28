"""Per-frame box generation for instances the multi-frame path cannot handle.

The multi-frame (aggregated + HDBSCAN) generator only sees SAM points that
landed on the static SDF mesh.  Dynamic instances and SDF-outlier instances are
boxed here, one frame and one instance at a time:

1. estimate a motion heading from the centroid shift to the previous / next
   frame (:func:`get_angle_if_far`); ``NO_HEADING`` when the instance is static
   or has no usable neighbour-frame observation;
2. fit a rectangle in the frame's ego coordinates and drop the instance when
   the fitted size exceeds the class prior by ``too_big_ratio``;
3. re-fit with the heading or, without one, with the point-normal fit, then
   snap the box to the statistical class size with ``BoundingBox.locate_bbox``
   or, for small / orientation-ambiguous footprints, ``locate_bbox_with_iou``
   (2D-mask IoU vote between the two 90-degree candidates);
4. re-set the box bottom from the raw ego-frame LiDAR (``refine_z_bbox``) and
   express the result in world (frame-0) coordinates.

Boxes are split into ``dynamic_boxes`` (heading known) and ``single_iou_boxes``
(heading unknown); the caller concatenates them, dynamic first.
"""
import numpy as np
import open3d as o3d

from .boxes import BoundingBox, refine_z_bbox
from .point_filters import points_rigid_transform

# Sentinel heading: "no heading could be estimated" (compared with ``==`` exactly as the original).
NO_HEADING = -1e6


def get_angle_if_far(current_pts, target_pts, center_diff_threshold, pose):
    """Ego-frame yaw of the centroid shift from ``current_pts`` to ``target_pts``, or
    ``NO_HEADING`` when the world-frame shift is not above ``center_diff_threshold`` metres.

    Both centroids go through ``points_rigid_transform`` (float32) on purpose.
    """
    current_center = np.mean(current_pts, axis=0)
    target_center = np.mean(target_pts, axis=0)
    if np.linalg.norm(current_center - target_center) > center_diff_threshold:
        tr_cur_center = points_rigid_transform(current_center.reshape(1, 3), np.linalg.inv(pose))
        tr_target_center = points_rigid_transform(target_center.reshape(1, 3), np.linalg.inv(pose))
        return np.arctan2(tr_target_center[0, 1] - tr_cur_center[0, 1],
                          tr_target_center[0, 0] - tr_cur_center[0, 0])
    return NO_HEADING


def single_frame_box_gen(cfg, loader, points_per_frame, ids_per_frame):
    """One box per (frame, instance) for the dynamic and SDF-outlier instances.

    ``cfg`` is the full config (uses ``single_frame`` and ``classes``); ``loader`` the
    SceneLoader (poses, raw LiDAR for the ground estimate, instance classes, 2D boxes
    and camera calibration for the 2D-IoU orientation vote).  ``points_per_frame`` /
    ``ids_per_frame`` hold the world-frame points and instance ids of every frame.

    Returns ``(dynamic_boxes, single_iou_boxes)``: two per-frame lists of
    ``(class_name, float32 (7,))`` world-frame boxes ``[cx, cy, cz, l, w, h, yaw]``
    (cz = box centre) in ascending instance-id order within a frame.
    """
    statistical_box_size = cfg.classes.statistical_box_size
    deformable_classes = cfg.classes.deformable
    pose_list = loader.pose_list
    full_pc_list = loader.raw_pc_sensor_list
    instance_cls_dict = loader.instance_cls_dict
    bbox_2d_info = loader.bbox_2d_info
    intrinsic_dict_list = loader.intrinsic_list
    extrinsic_dict_list = loader.projection_mat_list
    cameras = loader.cameras
    scene_idx = loader.scene_idx
    fit_cfg = cfg.box_fit
    cfg = cfg.single_frame

    num_frames = len(points_per_frame)
    dynamic_boxes = [[] for _ in range(num_frames)]
    single_iou_boxes = [[] for _ in range(num_frames)]

    for frame_idx in range(num_frames):
        pose = pose_list[frame_idx]
        frame_points = points_per_frame[frame_idx]
        frame_ids = ids_per_frame[frame_idx]
        for instance_id in np.unique(frame_ids):
            cls = instance_cls_dict[instance_id]
            if cls in deformable_classes:
                continue
            stat_size = np.array(statistical_box_size[cls])
            instance_pts = frame_points[frame_ids == instance_id]
            if len(instance_pts) < cfg.min_points:
                continue

            # Motion heading from the neighbouring frames.  The next-frame estimate
            # overwrites the previous-frame one (even back to NO_HEADING).
            heading = NO_HEADING
            if frame_idx > 0:
                prev_pts = points_per_frame[frame_idx - 1][ids_per_frame[frame_idx - 1] == instance_id]
                if len(prev_pts) > cfg.neighbour_frame_min_points:
                    heading = get_angle_if_far(prev_pts, instance_pts, cfg.center_diff_threshold, pose)
            if frame_idx < num_frames - 1:
                next_pts = points_per_frame[frame_idx + 1][ids_per_frame[frame_idx + 1] == instance_id]
                if len(next_pts) > cfg.neighbour_frame_min_points:
                    heading = get_angle_if_far(instance_pts, next_pts, cfg.center_diff_threshold, pose)

            instance_pts_ego = points_rigid_transform(instance_pts, np.linalg.inv(pose))
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(instance_pts_ego))

            # Too-large gate; this fit is discarded afterwards.
            bbox = BoundingBox().make_box(cloud, 'closeness_to_edge', fit_cfg)
            if (bbox.s > stat_size * cfg.too_big_ratio).any():
                continue

            if heading == NO_HEADING:
                bbox = BoundingBox().make_box(cloud, 'point_normal', fit_cfg)
                if bbox.s[:2].max() > stat_size[:2].min() * cfg.fix_ratio:
                    if bbox.s[:2].argmax() == bbox.s[:2].argmin():  # only when l == w exactly
                        bbox.r += np.pi / 2
                    bbox = bbox.locate_bbox(bbox.t, bbox.s, bbox.r, stat_size)
                else:
                    # The 2D-mask vote runs on the world-frame box (``transform`` applies
                    # the inverse of its argument) and the winner is mapped back to ego.
                    bbox = bbox.transform(np.linalg.inv(pose))
                    # Because locate_bbox runs inside locate_bbox_with_iou on this
                    # world-frame box, its face-visibility test (face_center: sensor at
                    # the origin) takes the frame-0 ego position as the viewpoint, not
                    # this frame's sensor -- unlike the ego-frame locate_bbox calls above.
                    bbox = bbox.locate_bbox_with_iou(
                        stat_size, instance_id, bbox_2d_info,
                        intrinsic_dict_list, extrinsic_dict_list, cameras).transform(pose)
            else:
                bbox = BoundingBox().make_box(cloud, 'given_angle', fit_cfg, heading)
                bbox = bbox.locate_bbox(bbox.t, bbox.s, bbox.r, stat_size)

            refined = refine_z_bbox([bbox], full_pc_list[frame_idx], stat_size, fit_cfg.ground_quantile,
                                    scene_idx, instance_id)
            if refined is None:  # no raw LiDAR around the box: ground unknown, instance dropped
                continue
            bbox = refined[0].transform(np.linalg.inv(pose))  # ego -> world (frame 0)
            box_entry = (cls, bbox.get_np_instance().astype(np.float32))
            if heading != NO_HEADING:
                dynamic_boxes[frame_idx].append(box_entry)
            else:
                single_iou_boxes[frame_idx].append(box_entry)

    return dynamic_boxes, single_iou_boxes

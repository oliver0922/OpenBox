"""Per-frame pseudo-label infos shared by the info-building stages (2 and 3) and the CLIs.

Both stages read the per-frame OpenPCDet info list of a Waymo segment, replace each frame's
``annos`` with pseudo labels and write the list back as ``<segment>/<segment>_<version>.pkl``.
The two point counts differ on purpose: stage 2 issues one batched kernel call, which assigns
every point to at most one box (overlapping boxes split the points they share), while stage 3
queries one box at a time, so a point inside two boxes is counted by both -- those counts
are the scores stage-3 NMS sorts by.
"""
import logging
import os
import pickle

import numpy as np
import torch  # before pcdet: its compiled ops need torch loaded first
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

logger = logging.getLogger(__name__)


def parse_scene_list(scenes):
    """Parse the ``--scenes`` value ``"0,1,125"`` into ``[0, 1, 125]``."""
    return [int(token) for token in scenes.split(',') if token.strip()]


def count_points_in_boxes_batched(points, boxes):
    """Stage-2 count: one batched kernel call over (M, >=3) points and (N, >=7) boxes; N ``np.int64``."""
    if len(points) == 0:  # a zero-block kernel launch aborts the whole process
        return [np.int64(0)] * len(boxes)
    points = torch.from_numpy(points[:, :3]).unsqueeze(0).float().cuda()
    boxes_cuda = torch.from_numpy(boxes[:, :7]).unsqueeze(0).float().cuda()
    box_idxs_of_pts = points_in_boxes_gpu(points, boxes_cuda).squeeze(0).cpu().numpy()
    return [np.sum(box_idxs_of_pts == box_idx) for box_idx in range(boxes.shape[0])]


def count_points_in_boxes_per_box(points, boxes):
    """Stage-3 count: one kernel call per box, so shared points count for every box; N ``int``."""
    if len(points) == 0:  # a zero-block kernel launch aborts the whole process
        return [0] * len(boxes)
    points = torch.from_numpy(points[:, :3]).unsqueeze(dim=0).float().cuda()
    counts = list()
    for box in boxes:
        box_idxs_of_pts = points_in_boxes_gpu(
            points, torch.from_numpy(box[:7]).reshape(1, 1, 7).float().cuda(),
        ).long().squeeze(dim=0).cpu().numpy()
        counts.append(len(np.where(box_idxs_of_pts != -1)[0]))
    return counts


def build_frame_annos(class_names, boxes, num_points_in_box, tracking_id=None):
    """One frame's ``annos`` record in the OpenPCDet Waymo layout.

    ``boxes`` is (N, >=7) ``[x, y, z, l, w, h, yaw, ...]``; only the first seven columns are
    written.  The dtype of ``num_points_in_box`` decides that of ``difficulty`` and
    ``num_points_in_gt`` (int64 in stage 2, float32 in stage 3).  ``difficulty``, ``obj_ids``,
    ``tracking_difficulty``, ``speed_global`` and ``accel_global`` are constant placeholders
    Waymo infos must carry; ``tracking_id`` is stored only when given (stage 3).
    """
    heading_angles = boxes[:, 6]
    speed_global = np.ones((len(num_points_in_box), 2))
    frame_annos = {
        'name': np.array(class_names),
        'difficulty': np.zeros_like(num_points_in_box),
        'dimensions': boxes[:, 3:6],
        'location': boxes[:, :3],
        'heading_angles': heading_angles,
        'obj_ids': np.ones_like(heading_angles),
        'tracking_difficulty': np.ones_like(heading_angles),
        'num_points_in_gt': np.array(num_points_in_box),
        'speed_global': speed_global,
        'accel_global': np.ones((len(num_points_in_box), 2)),
        'gt_boxes_lidar': np.concatenate((boxes[:, :7], speed_global), axis=1),
    }
    if tracking_id is not None:
        frame_annos['tracking_id'] = tracking_id
    return frame_annos


def save_infos(root, segment_name, version, infos):
    """Pickle ``infos`` to ``<root>/<segment>/<segment>_<version>.pkl``, creating the directory."""
    os.makedirs(os.path.join(root, segment_name), exist_ok=True)
    path = os.path.join(root, segment_name, '%s_%s.pkl' % (segment_name, version))
    with open(path, 'wb') as handle:
        pickle.dump(infos, handle)
    logger.info('wrote %d frame infos to %s', len(infos), path)

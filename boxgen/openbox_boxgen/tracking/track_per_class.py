"""Run AB3DMOT forward and backward over one scene for one category.

Per frame the category's boxes are taken from ``infos[i]['annos']`` (ego frame of frame
``i``), moved into the scene-centre frame (ego frame of ``center_pose``), tracked, and the
Kalman-smoothed boxes are moved back and appended to ``bbox_list[i]`` as
``(cat, float32 [x, y, z, l, w, h, yaw, track_id])``.  Two independent trackers run --
forward over ``0..N-1`` and backward over ``N-1..0`` -- both seeded with ``first_track_id``;
both outputs are appended (forward first) and de-duplicated by the caller's NMS.  The id
counter returned is advanced from the backward tracker only.
"""
import logging

import numpy as np

from ..boxes import BoundingBox
from .model import AB3DMOT

logger = logging.getLogger(__name__)


def track_per_class(num_frame, infos, center_pose, sequence_name, bbox_list, cfg, kalman_cfg, cat,
                    first_track_id):
    """Track ``cat`` through the scene; returns ``(bbox_list, first_track_id of the next category)``.

    num_frame: int; infos: list of per-frame info dicts (``annos``, ``pose``);
    center_pose: float64 (4, 4); bbox_list: per-frame lists, appended to and
    returned; cat: str category; first_track_id: int.
    ``cfg`` is ``tracking.categories[cat]``, ``kalman_cfg`` is ``tracking.kalman``;
    ``sequence_name`` is used in log messages only.
    """
    for direction, frame_indices in (('forward', range(num_frame)),
                                     ('backward', reversed(range(num_frame)))):
        logger.info('%s: tracking %s %s over %d frames', sequence_name, cat, direction, num_frame)
        tracker = AB3DMOT(cfg, kalman_cfg, first_track_id)
        for frame_idx in frame_indices:
            annos = infos[frame_idx]['annos']
            cls_mask = np.array([cls == cat for cls in annos['name']])
            if len(cls_mask) == 0:
                # A frame with no box of ANY class is skipped WITHOUT stepping the tracker:
                # its tracks neither predict nor age here, while a frame that only lacks
                # ``cat`` boxes does step it with empty ``cls_dets``.  Release behaviour;
                # it also avoids the float-index error the empty mask would raise below.
                continue
            cls_dets = annos['gt_boxes_lidar'][cls_mask]
            pose = np.linalg.inv(center_pose) @ infos[frame_idx]['pose']
            # BoundingBox.transform(M) applies inv(M): inv(pose) moves a box from this frame's
            # ego frame into the scene-centre frame and pose moves it back.
            pose_inv = np.linalg.inv(pose)
            new_cls_dets = []
            for anno in cls_dets:
                new_cls_dets.append(BoundingBox().load_gt(anno).transform(pose_inv).get_np_instance())
            results = tracker.track(np.array(new_cls_dets))

            for box_with_id in results:
                box = BoundingBox().load_gt(box_with_id[:7]).transform(pose).get_np_instance()
                trk_id = box_with_id[-1]
                bbox_list[frame_idx].append((cat, np.hstack((box.astype(np.float32), trk_id.astype(np.float32))).astype(np.float32)))

    return bbox_list, max(first_track_id, tracker.next_track_id)

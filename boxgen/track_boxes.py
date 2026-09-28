#!/usr/bin/env python
"""Stage 3 of the OpenBox box generation: multi-object tracking of the pseudo boxes.

Stage 2 (``build_pseudo_infos.py``) produces one independent set of boxes per
frame.  This stage links them over time with AB3DMOT and replaces them by the
Kalman-smoothed tracks, which removes flicker and recovers boxes in frames where
the box fit failed.

Per scene and per class (``classes.tracked`` of the config, in that order) two
trackers are run over the whole segment -- one forward, one backward -- and both
of their outputs are kept.  The union is then de-duplicated per frame by
rotated-BEV NMS whose score is the number of LiDAR points inside a box, so the
better supported of two overlapping boxes survives.

Inputs: ``{infos_root}/{segment}/{segment}_{input_version}.pkl`` (the stage-2
infos; each frame's ``annos`` holds ``name`` and ``gt_boxes_lidar``, each info
its 4x4 ``pose``) and ``{processed_root}/{segment}/masked_points/{f:04d}.npy``
(per-frame LiDAR).  Output: ``{out_root}/{segment}/{segment}_{output_version}.pkl``,
the same info list with tracked boxes and one extra ``annos`` key, ``tracking_id``.

Example::

    python track_boxes.py --scenes 0,1,2 \\
        --infos-root  /data/waymo/waymo_processed_data_v0_5_0_static \\
        --processed-root /data/waymo/waymo_processed_data_v0_5_0 \\
        --split-file  /data/waymo/ImageSets/train.txt \\
        --out-root    /data/waymo/waymo_processed_data_v0_5_0_static \\
        --workers 2 --gpu 0 --cfg configs/waymo.yaml
"""
import argparse
import logging
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import torch  # must be imported before the pcdet CUDA ops (openbox_boxgen.boxes / .infos)
from tqdm import tqdm

from openbox_boxgen.boxes import nms_gpu
from openbox_boxgen.config import load_config
from openbox_boxgen.infos import (build_frame_annos, count_points_in_boxes_per_box, parse_scene_list,
                                  save_infos)
from openbox_boxgen.io import read_segment_name
from openbox_boxgen.tracking import track_per_class

logger = logging.getLogger(__name__)


def track_scene(cfg, scene_idx, segment_name, infos_root, processed_root, out_root, input_version,
                output_version, gpu):
    """Track one scene's pseudo boxes and write the tracked infos (runs in a worker process).

    cfg: loaded config namespace; scene_idx: int; segment_name: str;
    infos_root/processed_root/out_root: str dirs; input_version/output_version:
    str tags; gpu: int CUDA device -> None (writes the tracked pkl).
    A scene without a stage-2 file is logged and skipped.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    torch.cuda.set_device(gpu)
    start = time.time()

    try:
        infos_path = os.path.join(infos_root, segment_name, '%s_%s.pkl' % (segment_name, input_version))
        with open(infos_path, 'rb') as handle:
            infos = pickle.load(handle)
    except FileNotFoundError as err:
        logger.error('scene-%d (%s): no input infos, skipped (%s)', scene_idx, segment_name, err)
        return
    num_frame = len(infos)

    # Per category, forward and backward through the scene in the scene-centre frame (the
    # ego frame of frame 0).  Per frame the forward pass of a category is appended before
    # its backward pass, and the track-id counter is threaded from one category to the
    # next, so the category order decides the ids and must not be reordered.
    bbox_list = [[] for _ in range(num_frame)]
    next_track_id = cfg.tracking.first_track_id
    # infos[0]['pose'] is the same vehicle-to-global matrix stage 2 read from pose/000000.bin,
    # so the boxes are tracked in the frame they were generated in.
    for category in cfg.classes.tracked:
        bbox_list, next_track_id = track_per_class(
            num_frame, infos, infos[0]['pose'], segment_name, bbox_list,
            cfg.tracking.categories[category], cfg.tracking.kalman, category, next_track_id)

    scene_infos = list()
    for frame_idx in tqdm(range(num_frame), desc='scene-%d annos' % scene_idx):
        columns = list(zip(*bbox_list[frame_idx]))  # (class, float32 [box7, track_id]) pairs
        if len(columns) == 0:
            class_names, boxes = list(), np.zeros((0, 8))
        else:
            class_names, boxes = columns[0], np.array(columns[1])

        # Point counts of the forward and backward boxes BEFORE de-duplication:
        # they are the scores NMS ranks the overlapping duplicates by.
        points = np.load(os.path.join(processed_root, segment_name, 'masked_points', '%04d.npy' % frame_idx))
        num_points_in_box = count_points_in_boxes_per_box(points, boxes)

        if len(boxes) != 0:
            boxes_with_num_points, class_names = nms_gpu(scene_idx, boxes, num_points_in_box, class_names,
                                                         cfg.nms.tracking.bev_iou_threshold, cfg.nms.tracking.pre_max_size)
            num_points_in_box = boxes_with_num_points[:, -1]
            boxes = boxes_with_num_points[:, :-1]
        else:
            boxes = np.zeros((0, 7))
            num_points_in_box = np.zeros((0, ))

        # Quirk of the released files, kept on purpose: NMS rebuilds every row as
        # [box7, num_points] and drops the track id the trackers appended, so
        # 'tracking_id' carries the heading angle instead of an id.  Nothing in
        # OpenPCDet reads it, and correcting it would change the released pkl.
        tracking_id = boxes[:, -1]

        infos[frame_idx]['annos'] = build_frame_annos(class_names, boxes, num_points_in_box,
                                                      tracking_id=tracking_id)
        scene_infos.append(infos[frame_idx])

    save_infos(out_root, segment_name, output_version, scene_infos)
    logger.info('scene-%d (%s): %d frames in %.1f s', scene_idx, segment_name, num_frame,
                time.time() - start)


def parse_args():
    """Parse the CLI arguments -> argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description='Track the stage-2 pseudo boxes per class (forward and backward) and '
                    'de-duplicate them with NMS.')
    parser.add_argument('--scenes', type=str, required=True,
                        help="comma separated scene indices to process, e.g. '0,1,125'; scene N "
                             'uses the segment named on line N of --split-file')
    parser.add_argument('--infos-root', type=str, required=True,
                        help='directory holding {segment}/{segment}_{input_version}.pkl')
    parser.add_argument('--processed-root', type=str, required=True,
                        help='processed Waymo data root holding '
                             '{segment}/masked_points/{frame:04d}.npy')
    parser.add_argument('--split-file', type=str, required=True,
                        help='ImageSets/train.txt whose line N names the segment of scene N')
    parser.add_argument('--out-root', type=str, required=True,
                        help='output directory; one {segment}/{segment}_{output_version}.pkl per scene')
    parser.add_argument('--input-version', type=str, default='openbox_untracked',
                        help='version tag of the stage-2 input file (default: %(default)s)')
    parser.add_argument('--output-version', type=str, default='openbox',
                        help='version tag of the tracked output file (default: %(default)s)')
    parser.add_argument('--cfg', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'waymo.yaml'),
                        help='YAML with every tunable of the pipeline (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of scenes tracked in parallel (default: %(default)s)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='CUDA device index used by the workers (default: %(default)s)')
    return parser.parse_args()


def track_scene_or_log(*task):
    """``track_scene`` that logs a failing scene with its traceback and lets the batch continue."""
    try:
        track_scene(*task)
    except Exception:  # noqa: BLE001  one broken scene must not stop the batch (as in generate_boxes)
        logger.exception('scene-%d (%s) failed', task[1], task[2])


def main():
    """Run stage 3 over the requested scenes."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    cfg = load_config(args.cfg)

    tasks = [(cfg, scene_idx, read_segment_name(args.split_file, scene_idx), args.infos_root,
              args.processed_root, args.out_root, args.input_version, args.output_version, args.gpu)
             for scene_idx in parse_scene_list(args.scenes)]

    # 'spawn', not the default 'fork': each worker builds its own CUDA context,
    # which cannot be inherited across a fork.
    pool = mp.get_context('spawn').Pool(args.workers)
    pool.starmap(track_scene_or_log, tasks)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

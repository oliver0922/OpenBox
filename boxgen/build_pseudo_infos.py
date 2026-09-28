#!/usr/bin/env python
"""Stage 2 of the OpenBox box generation: scene boxes -> per-frame pseudo-label infos.

Stage 1 (``generate_boxes.py``) writes one set of boxes per scene in the scene's
world frame (the ego frame of frame 0).  This stage turns them into the
per-frame OpenPCDet info format that training consumes: for every frame of the
segment it moves every box into that frame's ego frame, maps the fine SAM class
names to the coarse Waymo classes (``classes.fine_to_coarse`` of the config),
counts the LiDAR points inside each box and replaces the frame's ``annos``
record with the result.

Inputs, per scene ``N`` (segment name = line ``N`` of ``--split-file``): the four
stage-1 artifacts and the ``pose/{f:06d}.bin`` ego poses under
``{scene_root}/scene-N/``, ``{processed_root}/{segment}/{segment}_fov.pkl`` (the
pre-processed info list) and ``{processed_root}/{segment}/masked_points/{f:04d}.npy``
(per-frame LiDAR).  Output: ``{out_root}/{segment}/{segment}_{version}.pkl``.

Example::

    python build_pseudo_infos.py --scenes 0,1,2 \\
        --scene-root  /data/waymo/waymo_sam2 \\
        --processed-root /data/waymo/waymo_processed_data_v0_5_0 \\
        --split-file  /data/waymo/ImageSets/train.txt \\
        --out-root    /data/waymo/waymo_processed_data_v0_5_0_static \\
        --workers 2 --gpu 0 --cfg configs/waymo.yaml
"""
import argparse
import itertools
import logging
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import torch  # must be imported before the pcdet CUDA ops (openbox_boxgen.infos)
from tqdm import tqdm

from openbox_boxgen.boxes import BoundingBox
from openbox_boxgen.config import load_config
from openbox_boxgen.infos import (build_frame_annos, count_points_in_boxes_batched, parse_scene_list,
                                  save_infos)
from openbox_boxgen.io import read_segment_name
from openbox_boxgen.save import STAGE1_FILE

logger = logging.getLogger(__name__)


def load_stage1_boxes(scene_dir, num_frame):
    """Read ``<scene_dir>/openbox_boxes.pkl`` (written by ``openbox_boxgen.save``).

    Returns ``(static_boxes, static_classes, deformable_per_frame, single_frame_per_frame)``:
    (K, 7) float32 world-frame static boxes, their K fine class names, and two per-frame
    containers of ``(fine class, float32 (7,))`` pairs -- or None when the scene has no
    stage-1 output.  A scene without any static box gets zero static boxes AND empty
    deformable lists: the original run lost the deformable boxes of such scenes, and the
    released labels reflect that, so it is reproduced here explicitly.  The per-frame
    containers are indexed by FRAME index while stage 1 appended one entry per LOADED
    frame; the two agree only when stage 1 saw every frame of the segment (the release
    run did).
    """
    path = os.path.join(scene_dir, STAGE1_FILE)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as handle:
        result = pickle.load(handle)

    single_frame_per_frame = result['single_frame_boxes']
    if len(single_frame_per_frame) != num_frame:  # per-frame containers are indexed by frame index
        raise ValueError(f'{path}: stage 1 loaded {len(single_frame_per_frame)} frames but the info list '
                         f'has {num_frame}; stage 1 must see every frame of the segment')
    if result['static_boxes'] is None:
        logger.warning('%s: no static boxes; the deformable boxes are dropped as well (release behaviour)',
                       scene_dir)
        static_boxes = np.zeros((0, 7))
        static_classes = list()
        deformable_per_frame = [[] for _ in range(num_frame)]
    else:
        # rows are [box7, num_points]; the count is dropped here and recomputed per frame
        static_boxes = result['static_boxes'][:, :7]
        static_classes = result['static_classes']
        deformable_per_frame = result['deformable_boxes']

    return static_boxes, static_classes, deformable_per_frame, single_frame_per_frame


def build_scene_infos(cfg, scene_idx, segment_name, scene_root, processed_root, out_root, version, gpu):
    """Build and write the pseudo-label infos of one scene (runs in a worker process).

    cfg: loaded config namespace; scene_idx: int; segment_name: str;
    scene_root/processed_root/out_root: str dirs; version: str tag; gpu: int
    CUDA device -> None (writes {out_root}/{segment}/{segment}_{version}.pkl).
    Scenes without stage-1 output or without a pre-processed info list are logged and skipped.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    torch.cuda.set_device(gpu)
    start = time.time()

    scene_dir = os.path.join(scene_root, 'scene-%d' % scene_idx)
    try:
        with open(os.path.join(processed_root, segment_name, '%s_fov.pkl' % segment_name), 'rb') as handle:
            infos = pickle.load(handle)
    except FileNotFoundError as err:
        logger.error('scene-%d (%s): no info list, skipped (%s)', scene_idx, segment_name, err)
        return
    num_frame = len(infos)

    stage1_boxes = load_stage1_boxes(scene_dir, num_frame)
    if stage1_boxes is None:
        logger.error('scene-%d (%s): no stage-1 boxes in %s, skipped', scene_idx, segment_name, scene_dir)
        return
    static_boxes, static_classes, deformable_per_frame, single_frame_per_frame = stage1_boxes

    # The world frame is the ego frame of pose/000000.bin, so each frame's transform is
    # inv(pose_0) @ pose_f in float64.
    center_pose = np.fromfile(os.path.join(scene_dir, 'pose', '000000.bin'), dtype=np.float64).reshape(4, 4)
    fine_to_coarse = cfg.classes.fine_to_coarse

    scene_infos = list()
    for frame_idx in tqdm(range(num_frame), desc='scene-%d annos' % scene_idx):
        pose = np.fromfile(os.path.join(scene_dir, 'pose', '%06d.bin' % frame_idx), dtype=np.float64).reshape(4, 4)
        pose = np.linalg.inv(center_pose) @ pose

        # Static (multi-frame) boxes first, then deformable, then single-frame: this row
        # order is inherited by the info file and decides stage-3 track birth order and
        # NMS tie-breaks.  BoundingBox.transform applies the INVERSE of its argument, so
        # the ego -> world pose moves a box from the world frame into this frame's ego frame.
        frame_boxes = []
        for fine_class, box in itertools.chain(zip(static_classes, static_boxes),
                                               deformable_per_frame[frame_idx],
                                               single_frame_per_frame[frame_idx]):
            box_in_frame = BoundingBox().load_gt(box).transform(pose).get_np_instance().astype(np.float32)
            frame_boxes.append((fine_to_coarse[fine_class], box_in_frame))

        columns = list(zip(*frame_boxes))
        if len(columns) == 0:
            class_names, boxes = list(), np.zeros((0, 7))
        else:
            class_names, boxes = columns[0], np.array(columns[1])
        points = np.load(os.path.join(processed_root, segment_name, 'masked_points', '%04d.npy' % frame_idx))
        num_points_in_box = count_points_in_boxes_batched(points, boxes)

        infos[frame_idx]['annos'] = build_frame_annos(class_names, boxes, num_points_in_box)
        scene_infos.append(infos[frame_idx])

    save_infos(out_root, segment_name, version, scene_infos)
    logger.info('scene-%d (%s): %d frames in %.1f s', scene_idx, segment_name, num_frame, time.time() - start)


def parse_args():
    """Parse the CLI arguments -> argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description='Build per-frame pseudo-label infos from the stage-1 scene boxes.')
    parser.add_argument('--scenes', type=str, required=True,
                        help="comma separated scene indices to process, e.g. '0,1,125'; scene N "
                             'uses {scene_root}/scene-N and the segment on line N of --split-file')
    parser.add_argument('--scene-root', type=str, required=True,
                        help='directory holding the scene-N folders written by stage 1')
    parser.add_argument('--processed-root', type=str, required=True,
                        help='processed Waymo data root holding {segment}/{segment}_fov.pkl and '
                             '{segment}/masked_points/{frame:04d}.npy')
    parser.add_argument('--split-file', type=str, required=True,
                        help='ImageSets/train.txt whose line N names the segment of scene N')
    parser.add_argument('--out-root', type=str, required=True,
                        help='output directory; one {segment}/{segment}_{version}.pkl per scene')
    parser.add_argument('--version', type=str, default='openbox_untracked',
                        help='tag of the output file <segment>_<version>.pkl (default: %(default)s; '
                             'stage 3 reads it and writes the final <segment>_openbox.pkl)')
    parser.add_argument('--cfg', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'waymo.yaml'),
                        help='YAML with every tunable of the pipeline (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=2,
                        help='number of scenes processed in parallel (default: %(default)s)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='CUDA device index used by the workers (default: %(default)s)')
    return parser.parse_args()


def build_scene_infos_or_log(*task):
    """``build_scene_infos`` that logs a failing scene with its traceback and lets the batch continue."""
    try:
        build_scene_infos(*task)
    except Exception:  # noqa: BLE001  one broken scene must not stop the batch (as in generate_boxes)
        logger.exception('scene-%d (%s) failed', task[1], task[2])


def main():
    """Run stage 2 over the requested scenes, one worker process per scene at a time."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    cfg = load_config(args.cfg)

    tasks = [(cfg, scene_idx, read_segment_name(args.split_file, scene_idx), args.scene_root,
              args.processed_root, args.out_root, args.version, args.gpu)
             for scene_idx in parse_scene_list(args.scenes)]

    # 'spawn', not the default 'fork': each worker builds its own CUDA context,
    # which cannot be inherited across a fork.
    pool = mp.get_context('spawn').Pool(args.workers)
    pool.starmap(build_scene_infos_or_log, tasks)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

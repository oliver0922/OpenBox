"""Stage 1 of the OpenBox pseudo-label pipeline: per-scene 3-D box generation (Waymo).

Algorithm (one scene at a time; the order of the steps is load-bearing because
each step fixes the point order the next one sees)
--------------------------------------------------------------------------
1. **Load** (``openbox_boxgen.io.SceneLoader``): static SDF mesh, per-frame raw
   LiDAR, refined SAM instance points, poses, per-camera intrinsics /
   projection matrices, instance classes and 2-D boxes.  While loading, every
   frame's SAM points are split into a *static* and a *dynamic* set with the
   per-point ppscore of the raw LiDAR and the proximity to the static surface.
   All point lists except the raw LiDAR are expressed in the *scene frame* =
   the ego frame of ``pose/000000.bin``.
2. **Deformable boxes** (``openbox_boxgen.deformable``): person and bicycle
   instances do not aggregate across frames, so they are boxed per frame and
   removed from the per-frame SAM point lists.  This step also defines the
   frame-major, instance-sorted concatenation order that HDBSCAN sees later.
3. **SDF filter + single-frame boxes** (``openbox_boxgen.point_filters``,
   ``openbox_boxgen.single_frame``): the aggregated static SAM points are
   associated with the static mesh; points off the surface, and instances that
   lost all of their points, are diverted to the single-frame path and boxed
   per frame together with the dynamic instances.
4. **Cluster and match** (``openbox_boxgen.clustering``): HDBSCAN on the
   surviving aggregated static points, then a majority vote that relabels every
   cluster with the SAM instance id of most of its points.
5. **Multi-frame boxes** (``openbox_boxgen.multi_frame``): one refined box per
   static instance -- initial rectangle fit, extension along the visible mesh
   faces, 2-D IoU vote between the orientation candidates, height refinement.
6. **NMS and save** (``openbox_boxgen.boxes.nms_gpu``, ``openbox_boxgen.save``):
   per-class rotated-BEV NMS scored by the number of instance points, then
   ``{out_root}/scene-N/openbox_boxes.pkl`` is written (static boxes and their
   class names, per-frame deformable boxes, per-frame single-frame boxes);
   stage 2 (``build_pseudo_infos.py``) reads exactly this file.

Every tunable comes from the YAML given with ``--cfg`` (``configs/waymo.yaml``
holds the values of the released run).  Its ``*_sq_threshold`` values are
compared against SQUARED L2 distances (the faiss ``IndexFlatL2`` semantics of
the release run), so ``0.15`` is an effective radius of ``sqrt(0.15) = 0.387`` m.
Per-frame lists are indexed by LOAD POSITION, not by frame number: frames
without a raw point cloud are skipped without a placeholder.  ``hdbscan`` and
the CUDA NMS make the box set reproducible for a fixed environment, not across
library builds.

Example (``pcdet`` and this directory must be importable)::

    python generate_boxes.py --scenes 0,1,2 \\
        --scene-root  /data/waymo/waymo_sam2 \\
        --processed-root /data/waymo/waymo_processed_data_v0_5_0 \\
        --split-file  /data/waymo/ImageSets/train.txt \\
        --out-root    /data/waymo/waymo_sam2 \\
        --cfg configs/waymo.yaml --gpu 0
"""
import argparse
import logging
import os
import time

import torch  # must be imported before the pcdet CUDA ops used by nms_gpu

from openbox_boxgen.boxes import nms_gpu
from openbox_boxgen.clustering import ClusterMatcher, HDBSCANClusterer
from openbox_boxgen.config import load_config
from openbox_boxgen.deformable import DeformableBoxGen
from openbox_boxgen.infos import parse_scene_list
from openbox_boxgen.io import SceneLoader
from openbox_boxgen.multi_frame import MultiFrameBoxGen
from openbox_boxgen.point_filters import SDFPointFilter
from openbox_boxgen.save import save_stage1_outputs
from openbox_boxgen.single_frame import single_frame_box_gen

logger = logging.getLogger("openbox.generate_boxes")


def process_scene(cfg, scene_idx, scene_root, processed_root, split_file, out_root):
    """Run the six steps on one scene; returns the paths written under ``{out_root}/scene-{scene_idx}``.

    cfg: loaded config namespace; scene_idx: int; scene_root/processed_root/
    split_file/out_root: str paths -> list of str written file paths.
    A scene whose SDF filter leaves no aggregated static point produces no static
    boxes; its deformable and single-frame artifacts are still written.
    """
    start_time = time.time()
    out_dir = os.path.join(out_root, "scene-{}".format(scene_idx))

    # 1. load
    loader = SceneLoader(cfg, scene_idx=scene_idx, scene_root=scene_root,
                         processed_root=processed_root, split_file=split_file)

    # 2. deformable boxes.  MUTATES the loader's per-frame static SAM lists in place;
    #    the regrouped order it leaves behind is what the SDF filter and HDBSCAN see.
    split = DeformableBoxGen(cfg, loader).run()  # deformable boxes + the remaining static points

    # 3. SDF filter, then single-frame boxes for the dynamic and off-surface instances
    #    (the two ``_`` are colour arrays: carried for visualisation only, unused by every box stage)
    (static_points, static_ids, _,
     single_frame_pcd, _, single_frame_id, recorded_frame_idx) = SDFPointFilter(cfg, loader, split).filter()
    points_per_frame, ids_per_frame = loader.get_dynamic_frame_pcds(
        single_frame_pcd, single_frame_id, recorded_frame_idx)
    single_frame_boxes, single_iou_boxes = single_frame_box_gen(cfg, loader, points_per_frame, ids_per_frame)
    # One list per frame, motion-heading boxes first: this is the row order the later
    # stages (pseudo-label infos, tracking, NMS ties) inherit.
    for frame_pos in range(len(points_per_frame)):
        single_frame_boxes[frame_pos] += single_iou_boxes[frame_pos]

    if static_points is None or len(static_points) == 0:
        logger.warning("scene-%d: no aggregated static point survived the SDF filter; "
                       "writing the deformable and single-frame boxes only", scene_idx)
        return save_stage1_outputs(out_dir, None, None, split.deformable_boxes, single_frame_boxes)

    # 4. cluster the static points and relabel every cluster by its majority instance id
    clusterer = HDBSCANClusterer(cfg.hdbscan)
    clusterer.fit(static_points[:, :3])
    cluster_points = static_points[clusterer.inlier_mask]
    point_instance_ids = ClusterMatcher(static_ids[clusterer.inlier_mask],
                                        clusterer.labels[clusterer.inlier_mask]).match()

    # 5. one refined box per static instance
    multi_frame = MultiFrameBoxGen(cfg, loader, split.non_sam_pc, static_points,
                                   cluster_points, point_instance_ids).run()

    # 6. NMS and save
    boxes_after, classes_after = nms_gpu(scene_idx, multi_frame.boxes, multi_frame.num_points,
                                         multi_frame.class_names, cfg.nms.stage1.bev_iou_threshold,
                                         cfg.nms.stage1.pre_max_size)
    written = save_stage1_outputs(out_dir, boxes_after, classes_after,
                                  split.deformable_boxes, single_frame_boxes)
    logger.info("scene-%d complete: %d static boxes after NMS (%d before), %.2f s",
                scene_idx, 0 if boxes_after is None else len(boxes_after),
                len(multi_frame.boxes), time.time() - start_time)
    return written


def parse_args(argv=None):
    """Parse CLI arguments. argv: list of str or None (sys.argv) -> argparse.Namespace."""
    parser = argparse.ArgumentParser(
        description="OpenBox stage 1: generate pseudo 3-D boxes for one or more Waymo scenes.")
    parser.add_argument("--scenes", type=str, required=True,
                        help="comma separated scene indices to process, e.g. '0,1,125'; "
                             "scene N lives in <scene-root>/scene-N and is line N of --split-file")
    parser.add_argument("--scene-root", type=str, required=True,
                        help="directory holding the scene-N folders with the SAM point clouds, "
                             "poses, camera files and the static SDF mesh")
    parser.add_argument("--processed-root", type=str, required=True,
                        help="processed Waymo data root holding <segment>/{frame:04d}.npy and "
                             "<segment>/ppscore/{frame:04d}.npy")
    parser.add_argument("--split-file", type=str, required=True,
                        help="ImageSets/train.txt whose line N names the Waymo segment of scene-N")
    parser.add_argument("--out-root", type=str, required=True,
                        help="output root; scene-N/openbox_boxes.pkl is written under it")
    parser.add_argument("--cfg", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "waymo.yaml"),
                        help="YAML with every tunable of the pipeline (default: %(default)s)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device of the box NMS; the faiss neighbour search always uses GPU 0 "
                             "(neighbors.py) -- set CUDA_VISIBLE_DEVICES to move both (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv=None):
    """Process every requested scene; exit code 0 when all succeeded, 1 otherwise.

    A scene that raises is logged with its traceback and the run continues with the next one.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.cfg)
    torch.cuda.set_device(args.gpu)  # the CUDA NMS op runs on the current device

    failed = []
    for scene_idx in parse_scene_list(args.scenes):
        try:
            process_scene(cfg, scene_idx, args.scene_root, args.processed_root, args.split_file, args.out_root)
        except Exception:  # one broken scene must not stop a batch
            logger.exception("scene-%d failed", scene_idx)
            failed.append(scene_idx)

    if failed:
        logger.error("%d scene(s) failed: %s", len(failed), failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

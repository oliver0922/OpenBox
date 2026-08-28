"""Context-aware refinement of the merged SAM instance point clouds (step 4).

For every frame the raw lidar sweep (<seq_data_dir>/<segment>/NNNN.npy) is
cleaned with Patchwork++ ground removal and a z threshold, the remaining
points are clustered with HDBSCAN, and each geometric cluster is matched by
majority vote (nearest-neighbour overlap) to one of the merged SAM instances
(<sam_data_dir>/scene-N/merged_sam_pc/<aug_name>/NNNNNN.bin). Clusters that
match an instance are written out with that instance id; unmatched clusters
and the noisy points introduced by 2D-to-3D projection are dropped.

Inputs
  --seq_data_dir  processed lidar frames, <segment>/NNNN.npy, float32 (N,6)
  --sam_data_dir  merged instance clouds, scene-N/merged_sam_pc/<aug_name>/NNNNNN.bin
  --split_file    ImageSets/train.txt (line N = segment name of scene-N)

Output
  <out_dir>/scene-N/refined_sam_pc/<aug_name>/NNNNNN.bin
  float64 (R,4): x, y, z, instance_id

Usage
  python refine_sam_pcd.py \
      --seq_data_dir $OUT/processed --sam_data_dir $OUT/sam \
      --split_file $WAYMO/ImageSets/train.txt \
      --aug_name adaptive1200_30_50_10_2 \
      --scene-start 0 --scene-end 797 --out_dir $OUT/refined --num_workers 16

--scene-start/--scene-end are inclusive (0..797 = all scenes). --num_workers > 1 splits the scene
range into contiguous batches, one process per batch.
"""
import argparse
import numpy as np
import os
import open3d as o3d
import pypatchworkpp
from hdbscan import HDBSCAN
from tqdm import tqdm
import faiss
import scipy
import time
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


def load_pcd(args, scene_idx, frame_idx):
    seq_name_path = args.split_file
    if seq_name_path is None:
        seq_name_path = os.path.join(args.seq_data_dir, '../ImageSets_range/train.txt')
        if not os.path.exists(seq_name_path):
            seq_name_path = os.path.join(args.seq_data_dir, '../ImageSets/train.txt')
    with open(seq_name_path) as f:
        seq_name_list = [x.strip().split('.')[0] for x in f.readlines()]
    seq_name = seq_name_list[scene_idx]
    seq_path = os.path.join(args.seq_data_dir, seq_name)
    full_pc = np.load(os.path.join(seq_path, f'{str(frame_idx).zfill(4)}.npy'))[:, :3]
    return full_pc


def ground_plane_removal(args, full_pc):
    params = pypatchworkpp.Parameters()
    params.enable_RNR = False
    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    PatchworkPLUSPLUS.estimateGround(full_pc)
    groundless_pc = PatchworkPLUSPLUS.getNonground()
    groundless_pc = groundless_pc[groundless_pc[:, 2] > args.z_threshold]
    return groundless_pc


def load_instance_pcd(args, scene_idx, frame_idx):
    try:
        instance_pc_all = (np.fromfile(os.path.join(args.sam_data_dir, f'scene-{scene_idx}', 'merged_sam_pc', args.aug_name, f'{str(frame_idx).zfill(6)}.bin'), dtype=np.float32).reshape(-1, 4))
        # remove with z threshold
        instance_pc_all = instance_pc_all[instance_pc_all[:, 2] > args.z_threshold]
        instance_pcd_list = []
        instance_pcd_id_list = np.unique(instance_pc_all[:, 3])
        for instance_id in instance_pcd_id_list:
            instance_pc = instance_pc_all[instance_pc_all[:, 3] == instance_id][:, :3]
            instance_pcd_list.append(instance_pc)
        return instance_pcd_list, instance_pcd_id_list
    except:
        return [], []


def match_segment_instances(args, segment_list, instance_pcd_list, instance_pcd_id_list):
    segment_id_list = []
    for s_id, segment in enumerate(segment_list):
        for i_id, instance_pcd in zip(instance_pcd_id_list, instance_pcd_list):
            tree = scipy.spatial.cKDTree(instance_pcd)
            D, I = tree.query(segment)
            seg_in_inst = np.sum(D < args.match_dist)
            if seg_in_inst / len(segment) > args.seg_overlap and seg_in_inst / len(instance_pcd) > args.inst_overlap:
                segment_id_list.append(i_id)
                break
        if len(segment_id_list) <= s_id:
            segment_id_list.append(-1)
    return segment_id_list


def merge_segments_with_id(args, segment_id_list, segment_list):
    refined_instace_pcd_with_id_list = []
    for i, segment_id in enumerate(segment_id_list):
        if segment_id == -1:
            continue
        segment_with_id = np.concatenate([segment_list[i], np.ones((len(segment_list[i]), 1)) * segment_id], axis=1)
        refined_instace_pcd_with_id_list.append(segment_with_id)
    # stack into one (R, 4) array; empty when no cluster matched an instance
    try:
        refined_instace_pcd_with_id_list = np.concatenate(refined_instace_pcd_with_id_list, axis=0)
    except:
        refined_instace_pcd_with_id_list = np.array([])
    return refined_instace_pcd_with_id_list


def hdbscan_areas(args, pc):
    hdbscaner = HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                        gen_min_span_tree=True, leaf_size=100, metric='euclidean',
                        min_cluster_size=args.min_cluster_size,
                        min_samples=args.min_samples,
                        cluster_selection_method='eom')
    cluster_idx = hdbscaner.fit_predict(pc)
    outlier_mask = hdbscaner.outlier_scores_ > args.outlier_threshold
    cluster_idx = cluster_idx[~outlier_mask]
    pc = pc[~outlier_mask]
    cluster_list = []
    for i in range(cluster_idx.max() + 1):
        cluster = pc[cluster_idx == i]
        if len(cluster) < args.min_segment_pc_size:
            continue
        cluster_list.append(cluster)
    return cluster_list


def scene_thing(args, scene_range):
    out_root = args.sam_data_dir if args.in_place else args.out_dir
    for scene_idx in scene_range:
        save_dir = os.path.join(out_root, f'scene-{scene_idx}', 'refined_sam_pc', args.aug_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for frame_idx in range(args.frame_start, args.frame_end):
            save_path = os.path.join(save_dir, f'{str(frame_idx).zfill(6)}.bin')
            if args.skip_existing and os.path.exists(save_path):
                continue
            try:
                full_pc = load_pcd(args, scene_idx, frame_idx)
            except:
                continue
            print(f'scene-{scene_idx} frame-{frame_idx}')
            t = time.time()
            full_pc = load_pcd(args, scene_idx, frame_idx)
            groundless_pc = ground_plane_removal(args, full_pc)

            segment_list = hdbscan_areas(args, groundless_pc)

            instance_pcd_list, instance_pcd_id_list = load_instance_pcd(args, scene_idx, frame_idx)
            segment_id_list = match_segment_instances(args, segment_list, instance_pcd_list, instance_pcd_id_list)
            refined_instace_pcd_list = merge_segments_with_id(args, segment_id_list, segment_list)

            refined_instace_pcd_list.tofile(save_path)
            print(f'total {time.time() - t}')


def main(args):
    scene_list = range(args.scene_start, args.scene_end + 1)

    if args.num_workers <= 1:
        scene_thing(args, scene_list)
        return

    ps = []
    multiprocessing.set_start_method('spawn')
    batch_size = max(1, (len(scene_list) + args.num_workers - 1) // args.num_workers)
    for i in range(0, len(scene_list), batch_size):
        ed = min(i + batch_size, len(scene_list))
        p = multiprocessing.Process(target=scene_thing, args=(args, scene_list[i:ed]))
        ps.append(p)
        p.start()
    for p in ps:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Context-aware refinement of merged SAM instance point clouds (step 4)')
    parser.add_argument('--seq_data_dir', type=str, required=True,
                        help='processed lidar root ($OUT/processed): <segment>/NNNN.npy')
    parser.add_argument('--sam_data_dir', type=str, required=True,
                        help='merged instance root ($OUT/sam): scene-N/merged_sam_pc/<aug_name>/')
    parser.add_argument('--split_file', type=str, default=None,
                        help='ImageSets/train.txt (line N = scene-N). Default: '
                             '<seq_data_dir>/../ImageSets_range/train.txt if present, '
                             'else <seq_data_dir>/../ImageSets/train.txt')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='output root (required unless --in_place); refined pcs go to '
                             '<out_dir>/scene-N/refined_sam_pc/<aug_name>/')
    parser.add_argument('--in_place', action='store_true',
                        help='write into sam_data_dir (overwrites existing results there)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='skip frames whose output file already exists')
    parser.add_argument('--scene-start', type=int, default=0,
                        help='first scene index (inclusive)')
    parser.add_argument('--scene-end', type=int, default=797,
                        help='last scene index (inclusive; 797 = all 798 scenes)')
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=200)

    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of scene-level worker processes')

    parser.add_argument('--min_segment_pc_size', type=int, default=15,
                        help='drop HDBSCAN clusters with fewer points than this')
    parser.add_argument('--min_cluster_size', type=int, default=15,
                        help='HDBSCAN min_cluster_size')
    parser.add_argument('--min_samples', type=int, default=10,
                        help='HDBSCAN min_samples')
    parser.add_argument('--outlier_threshold', type=float, default=0.5,
                        help='drop points with HDBSCAN outlier score above this')
    parser.add_argument('--match_dist', type=float, default=0.1,
                        help='nearest-neighbour distance (m) that counts a cluster point as inside an instance')
    parser.add_argument('--seg_overlap', type=float, default=0.3,
                        help='min fraction of a cluster covered by an instance to accept the match')
    parser.add_argument('--inst_overlap', type=float, default=0.2,
                        help='min fraction of the instance covered by the cluster to accept the match')
    parser.add_argument('--z_threshold', type=float, default=0,
                        help='drop points at or below this height after ground removal')

    parser.add_argument('--aug_name', type=str, default='adaptive1200_30_50_10_2',
                        help='mask variant to refine: no_aug or adaptive1200_30_50_10_2')

    args = parser.parse_args()

    if args.in_place:
        assert args.out_dir is None, 'use either --out_dir or --in_place, not both'
    else:
        assert args.out_dir is not None, 'pass --out_dir (or --in_place to write into sam_data_dir)'
        out_abs = os.path.realpath(args.out_dir)
        sam_abs = os.path.realpath(args.sam_data_dir)
        assert not out_abs.startswith(sam_abs + os.sep) and out_abs != sam_abs, \
            'out_dir points inside sam_data_dir; use --in_place to write there explicitly'

    s = time.time()
    main(args)
    print(time.time() - s)

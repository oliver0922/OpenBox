"""Merge per-camera SAM instance ids into scene-consistent ids (step 3).

Pass 1 links instance ids whose point clouds overlap across cameras, pass 2
rewrites the per-camera point clouds into single per-frame files using the
merged ids.

Input:  $OUT/sam/scene-N/<CAM>/[<aug>/]visualization/uppc_{,color_}continuous_sam/NNNNNN.bin
Output: $OUT/sam/scene-N/merged_sam_pc/<aug>/NNNNNN.bin      float32 (M,4) x,y,z,instance_id
        $OUT/sam/scene-N/merged_sam_color/<aug>/NNNNNN.bin   float32 (M,3) rgb
        $OUT/sam/scene-N/conn_dict_<aug>.pkl                 {instance_id: merged_into_id}
Writes into $OUT/sam itself. Run once per mask variant (no_aug, adaptive1200_30_50_10_2).

Usage:
    python merge_instances.py \
        --data_root $OUT/sam --augname adaptive1200_30_50_10_2 \
        --scene-start 0 --scene-end 797 --num_workers 32
"""
import os
import argparse
import functools
import multiprocessing
import pickle

import numpy as np
import open3d as o3d
from tqdm import tqdm

CAM_LIST = ['SIDE_LEFT', 'FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_RIGHT']


def sam_frame_path(data_path, cam, augname, dir_name, frame_idx):
    """Path to a per-camera SAM result file; 'no_aug' results live outside the aug subdir."""
    if augname == 'no_aug':
        return os.path.join(data_path, cam, 'visualization', dir_name, f'{frame_idx:06d}.bin')
    return os.path.join(data_path, cam, augname, 'visualization', dir_name, f'{frame_idx:06d}.bin')


def load_bin(path, num_cols):
    """Load an Nx{num_cols} float32 binary file, or None if it does not exist."""
    try:
        return np.fromfile(path, dtype=np.float32).reshape(-1, num_cols)
    except FileNotFoundError:
        return None


def count_shared_points(points_a, points_b):
    """Number of exactly-duplicated points between two point sets."""
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.concatenate([points_a, points_b], axis=0))
    deduped = merged.remove_duplicated_points()
    return len(points_a) + len(points_b) - len(deduped.points)


def resolve_id(cluster_id, conn_dict):
    """Follow child -> parent links to the final merged instance id."""
    while cluster_id in conn_dict:
        cluster_id = conn_dict[cluster_id]
    return cluster_id


def find_connected_instances(data_path, augname, frame_range, overlap_threshold):
    """Pass 1: link instance ids that overlap in 3D across cameras.

    Returns (conn_dict, color_dict): conn_dict maps larger id -> smaller id for
    each overlapping pair; color_dict keeps one representative color per id.
    """
    conn_dict = {}
    color_dict = {}

    for frame_idx in tqdm(frame_range, desc='Frame loading'):
        points_per_id = {}
        for cam in CAM_LIST:
            sam_pc = load_bin(sam_frame_path(data_path, cam, augname, 'uppc_continuous_sam', frame_idx), 4)
            sam_color = load_bin(sam_frame_path(data_path, cam, augname, 'uppc_color_continuous_sam', frame_idx), 3)
            if sam_pc is None or sam_color is None:
                continue

            ids = sam_pc[:, 3].astype(np.int32)
            for cluster_id in np.unique(ids):
                mask = ids == cluster_id
                points_per_id[cluster_id] = sam_pc[mask, :3]
                color_dict[cluster_id] = sam_color[mask][0]

        frame_ids = sorted(points_per_id)
        for i, id_a in enumerate(frame_ids):
            for id_b in frame_ids[i + 1:]:
                shared = count_shared_points(points_per_id[id_a], points_per_id[id_b])
                if shared > overlap_threshold:
                    conn_dict[id_b] = id_a

    return conn_dict, color_dict


def save_merged_frames(data_path, augname, frame_range, conn_dict, color_dict):
    """Pass 2: rewrite per-camera point clouds with merged ids into single per-frame files."""
    pc_out_dir = os.path.join(data_path, 'merged_sam_pc', augname)
    color_out_dir = os.path.join(data_path, 'merged_sam_color', augname)
    os.makedirs(pc_out_dir, exist_ok=True)
    os.makedirs(color_out_dir, exist_ok=True)

    for frame_idx in tqdm(frame_range, desc='Frame saving'):
        pc_chunks = []
        color_chunks = []

        for cam in CAM_LIST:
            sam_pc = load_bin(sam_frame_path(data_path, cam, augname, 'uppc_continuous_sam', frame_idx), 4)
            if sam_pc is None:
                continue

            ids = sam_pc[:, 3].astype(np.int32)
            for cluster_id in np.unique(ids):
                target_id = resolve_id(cluster_id, conn_dict)
                mask = ids == cluster_id
                num_points = int(mask.sum())
                id_col = np.full((num_points, 1), target_id, dtype=np.float32)
                pc_chunks.append(np.hstack([sam_pc[mask, :3], id_col]))
                color_chunks.append(np.tile(color_dict[target_id], (num_points, 1)))

        if not pc_chunks:
            continue

        np.concatenate(pc_chunks).tofile(os.path.join(pc_out_dir, f'{frame_idx:06d}.bin'))
        np.concatenate(color_chunks).tofile(os.path.join(color_out_dir, f'{frame_idx:06d}.bin'))


def merge_instances(scene_idx, args):
    data_path = os.path.join(args.data_root, f'scene-{scene_idx}')
    frame_range = range(args.num_frames)

    conn_dict, color_dict = find_connected_instances(data_path, args.augname, frame_range, args.overlap_threshold)
    save_merged_frames(data_path, args.augname, frame_range, conn_dict, color_dict)

    with open(os.path.join(data_path, f'conn_dict_{args.augname}.pkl'), 'wb') as f:
        pickle.dump(conn_dict, f)
    print("scene_idx: ", scene_idx, "done")


def main(args):
    worker = functools.partial(merge_instances, args=args)
    pool = multiprocessing.Pool(processes=args.num_workers)
    pool.map(worker, range(args.scene_start, args.scene_end + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-start', type=int, default=0, help='first scene index')
    parser.add_argument('--scene-end', type=int, default=797, help='last scene index (inclusive; 797 = all scenes)')
    parser.add_argument('--augname', type=str, default='adaptive1200_30_50_10_2',
                        help='mask variant to merge: no_aug or adaptive1200_30_50_10_2')
    parser.add_argument('--data_root', type=str, required=True,
                        help='step-2 output root ($OUT/sam, contains scene-N/); merged results are written back into it')
    parser.add_argument('--num_frames', type=int, default=200, help='frames per scene (0 .. num_frames-1)')
    parser.add_argument('--overlap_threshold', type=int, default=20,
                        help='merge two instances if they share more than this many duplicated points')
    parser.add_argument('--num_workers', type=int, default=64, help='multiprocessing pool size')

    args = parser.parse_args()
    main(args)

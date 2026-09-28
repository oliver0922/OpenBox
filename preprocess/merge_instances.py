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
import json
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
    """Path to a per-camera SAM result file; 'no_aug' results live outside the aug subdir.

    data_path/cam/augname/dir_name: str; frame_idx: int -> str file path.
    """
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
    """Number of exactly-duplicated points between two point sets.

    points_a: float32 (A, 3) xyz; points_b: float32 (B, 3) xyz -> int count.
    """
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(np.concatenate([points_a, points_b], axis=0))
    deduped = merged.remove_duplicated_points()
    return len(points_a) + len(points_b) - len(deduped.points)


def resolve_id(cluster_id, conn_dict):
    """Follow child -> parent links to the final merged instance id.

    cluster_id: int id; conn_dict: {int: int} child -> parent -> int final id.
    """
    while cluster_id in conn_dict:
        cluster_id = conn_dict[cluster_id]
    return cluster_id


def find_connected_instances(data_path, augname, frame_range, overlap_threshold):
    """Pass 1: link instance ids that overlap in 3D across cameras.

    data_path: str scene dir; augname: str mask variant; frame_range: iterable
    of int; overlap_threshold: int shared-point count.
    Returns (conn_dict, color_dict): conn_dict {int: int} maps larger id ->
    smaller id for each overlapping pair; color_dict {int: float32 (3,) rgb}
    keeps one representative color per id.
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
    """Pass 2: rewrite per-camera point clouds with merged ids into single per-frame files.

    data_path: str scene dir; augname: str; frame_range: iterable of int;
    conn_dict: {int: int}; color_dict: {int: float32 (3,) rgb} -> None. Writes
    float32 (M, 4) x,y,z,id and float32 (M, 3) rgb .bin files per frame.
    """
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


CAM_ORDER = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']


def save_agg_mask(data_path, conn_dict, frame_range) -> None:
    """Aggregate the per-camera 2D detections into one ``agg_mask.json`` per scene.

    Reads each camera's step-2 ``erosion/json_data/mask_NNNNNN.json``
    (``labels: {instance_id: {class_name, x1, y1, x2, y2, ...}}``) and writes
    ``agg_mask.json``: ``{frame_idx: {merged_instance_id: [{cam_loc: str,
    class_name: str, bbox: [x1, y1, x2, y2]}, ...]}}`` (all int pixel coords).
    Instance ids are remapped through ``conn_dict`` by a SINGLE lookup step,
    exactly like the run that produced the released files (not the chained
    ``resolve_id`` used for the point clouds).  Frames where no camera has a
    json are skipped.  Box generation reads this file for the 2D IoU vote.

    data_path: str scene dir; conn_dict: {int: int}; frame_range: iterable of
    int -> None.
    """
    mask_per_scene = {}
    for frame_idx in frame_range:
        mask_per_frame = {}
        for cam_loc in CAM_ORDER:
            mask_path = os.path.join(data_path, cam_loc, 'erosion', 'json_data',
                                     f'mask_{frame_idx:06d}.json')
            if not os.path.exists(mask_path):
                continue
            with open(mask_path) as json_file:
                labels = json.load(json_file)['labels']
            for instance_id, info in labels.items():
                entry = {'cam_loc': cam_loc, 'class_name': info['class_name'],
                         'bbox': [info['x1'], info['y1'], info['x2'], info['y2']]}
                if int(instance_id) in conn_dict:
                    merged_id = str(conn_dict[int(instance_id)])
                    mask_per_frame.setdefault(merged_id, []).append(entry)
                else:
                    # unmapped ids REPLACE any existing list -- kept exactly as the
                    # run that produced the released agg_mask.json behaved
                    mask_per_frame[str(instance_id)] = [entry]
        if mask_per_frame:
            mask_per_scene[frame_idx] = mask_per_frame
    with open(os.path.join(data_path, 'agg_mask.json'), 'w') as json_save_file:
        json.dump(mask_per_scene, json_save_file)


def merge_instances(scene_idx, args):
    """Run both passes for one scene and dump conn_dict (+ agg_mask for adaptive).

    scene_idx: int scene number; args: parsed CLI namespace -> None.
    """
    data_path = os.path.join(args.data_root, f'scene-{scene_idx}')
    frame_range = range(args.num_frames)

    conn_dict, color_dict = find_connected_instances(data_path, args.augname, frame_range, args.overlap_threshold)
    save_merged_frames(data_path, args.augname, frame_range, conn_dict, color_dict)

    with open(os.path.join(data_path, f'conn_dict_{args.augname}.pkl'), 'wb') as f:
        pickle.dump(conn_dict, f)
    if args.augname == 'adaptive1200_30_50_10_2':
        # the 2D-box aggregate is defined on the adaptive-mask merge only
        save_agg_mask(data_path, conn_dict, frame_range)
    print("scene_idx: ", scene_idx, "done")


def main(args):
    """Map merge_instances over the scene range with a worker pool. args: parsed CLI namespace -> None."""
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

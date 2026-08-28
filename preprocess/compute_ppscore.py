"""Compute the per-point persistence score (PP-score) for Waymo sequences.

For every frame i of a sequence, gather one-frame temporal windows at
j in [i-30, i+30) step 5, transform them into frame i's ego coordinates via
the poses in <segment>/<segment>.pkl, count neighbors within 0.3 m of every
current-frame point per window (cKDTree), and store the normalized entropy H
of the per-window count distribution as float16:
    <out-root>/<segment>/ppscore/NNNN.npy   float16 (N,)
High H = point observed persistently across time = static. The downstream
static-mesh step (dynamic_removed_vdbfusion.py) thresholds H > 0.7.

Adapted from CPD (cpd/unsupervised_core/precompute_ppscore.py) with the
PPScoreConfig hyperparameters of waymo_unsupervised_cproto.yaml
(max_win_size 30, win_interval 5, max_neighbor_dist 0.3).

Input:  <processed-data-root>/<segment>/{NNNN.npy, <segment>.pkl}
        (step 0, waymo_processed_gen.py)
Output: <out-root>/<segment>/ppscore/NNNN.npy

Example:
    python compute_ppscore.py \
        --processed-data-root $OUT/processed \
        --split-file $WAYMO/ImageSets/train.txt \
        --scene-start 0 --scene-end 797 --out-root $OUT/ppscore --num-workers 10
"""

import argparse
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def count_neighbors(points, trees, max_neighbor_dist):
    counts = [
        tree.query_ball_point(points[:, :3], r=max_neighbor_dist, return_length=True)
        for tree in trees
    ]
    return np.stack(counts).T


def compute_ephe_score(count):
    num_windows = count.shape[1]
    P = count / (np.expand_dims(count.sum(axis=1), -1) + 1e-8)
    return (-P * np.log(P + 1e-8)).sum(axis=1) / np.log(num_windows)


def compute_ppscore(cur_frame, neighbor_traversals, max_neighbor_dist):
    trees = [cKDTree(points) for points in neighbor_traversals]
    count = count_neighbors(cur_frame, trees, max_neighbor_dist)
    return compute_ephe_score(count)


def points_rigid_transform(cloud, pose):
    if cloud.shape[0] == 0:
        return cloud
    mat = np.ones(shape=(cloud.shape[0], 4), dtype=np.float32)
    pose_mat = np.mat(pose)
    mat[:, 0:3] = cloud[:, 0:3]
    mat = np.mat(mat)
    transformed_mat = pose_mat * mat.T
    T = np.array(transformed_mat.T, dtype=np.float32)
    return T[:, 0:3]


def save_pp_score(
    seq_name, root_path, out_root, max_win=30, win_inte=5, max_neighbor_dist=0.3
):
    """Write ppscore .npy files for one sequence; returns the sequence name."""
    out_dir = os.path.join(out_root, seq_name, "ppscore")
    os.makedirs(out_dir, exist_ok=True)

    # <segment>.pkl: written by waymo_processed_gen.py (step 0). An OpenPCDet
    # create_waymo_infos pkl also works: its entries carry the same `pose` field.
    with open(os.path.join(root_path, seq_name, seq_name + ".pkl"), "rb") as f:
        infos = pickle.load(f)

    for i in range(len(infos)):
        pose_i = np.linalg.inv(infos[i]["pose"])

        all_traversals = []
        cur_points = None

        for j in range(i - max_win, i + max_win, win_inte):
            lidar_path = os.path.join(root_path, seq_name, str(j).zfill(4) + ".npy")
            if not os.path.exists(lidar_path):
                continue
            lidar_points = np.load(lidar_path)[:, 0:3]
            if j == i:
                cur_points = lidar_points
            transformed = points_rigid_transform(lidar_points, infos[j]["pose"])
            all_traversals.append(points_rigid_transform(transformed, pose_i))

        H = compute_ppscore(cur_points, all_traversals, max_neighbor_dist)
        np.save(
            os.path.join(out_dir, str(i).zfill(4) + ".npy"),
            np.array(H).astype(np.float16),
        )

    return seq_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute PP-scores for Waymo sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--processed-data-root",
        type=Path,
        required=True,
        help="Root with <segment>/NNNN.npy lidar frames and <segment>/<segment>.pkl "
        "pose files (step 0 output, read only).",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Split file whose line N is the segment name of scene-N "
        "($WAYMO/ImageSets/train.txt).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root; <segment>/ppscore/NNNN.npy files are written here "
        "(existing files are overwritten).",
    )
    parser.add_argument("--scene-start", type=int, default=0, help="first scene index")
    parser.add_argument(
        "--scene-end", type=int, default=797, help="last scene index (inclusive)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=10, help="number of worker processes"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with args.split_file.open("r") as stream:
        segment_names = [line.strip().split(".")[0] for line in stream if line.strip()]
    seq_list = segment_names[args.scene_start : args.scene_end + 1]

    mp.set_start_method("spawn")
    with mp.Pool(processes=args.num_workers) as pool:
        progress = tqdm(total=len(seq_list), desc="sequences")
        results = [
            pool.apply_async(
                save_pp_score,
                args=(seq, str(args.processed_data_root), str(args.out_root)),
                callback=lambda _: progress.update(1),
            )
            for seq in seq_list
        ]
        for result in results:
            result.get()
        progress.close()
    print("All done!")


if __name__ == "__main__":
    main()

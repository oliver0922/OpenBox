"""Waymo Grounded-SAM2 unified generation script.

One run produces, per scene:

    <out_dir>/scene-N/<CAM>/erosion/mask_data/mask_XXXXXX.npy   raw instance masks (uint16)
    <out_dir>/scene-N/<CAM>/erosion/json_data/mask_XXXXXX.json  instance id / class / bbox
    <out_dir>/scene-N/<CAM>/erosion/result/*.jpeg               annotated, fixed erosion
    <out_dir>/scene-N/<CAM>/result/*.jpeg                       annotated, no erosion
    <out_dir>/scene-N/<CAM>/adaptive_erosion/<P>/result/*.jpeg  annotated, adaptive erosion
    <out_dir>/scene-N/<CAM>/visualization/uppc_*_sam/*.bin      3D unprojection (no erosion)
    <out_dir>/scene-N/<CAM>/adaptive<P>/visualization/...       3D unprojection (adaptive)
    <out_dir>/scene-N/instance_classname_dict.pkl

<P> = <width_thres_1>_<width_thres_2>_<iteration_0>_<iteration_1>_<iteration_2>

Scenes are distributed over the GPUs given by --gpus, one worker process per
GPU. Models are loaded once per worker.


Example (run from the mmdetection fork root):
    python groundedsamv2/gen_multicam_samv2_data_unified.py \
        configs/grounding_dino/grounding_dino_swin-b_finetune_8xb4_1x_nus.py \
        --data_dir $OUT/scenes --out_dir $OUT/sam \
        --scene-start 0 --scene-end 797 --gpus 0,1,2,3
"""
import torch
import os
import cv2
import time
import pickle
import multiprocessing
from functools import partial
from argparse import ArgumentParser

from mmdet.apis import DetInferencer_SAMV2

from groundedsamv2.utils.mask_dictionary_model import MaskDictionaryModel
from groundedsamv2.utils.multicam_common_utils_unified import CommonUtils

from groundedsamv2.sam2.build_sam import build_sam2_video_predictor, build_sam2
from groundedsamv2.sam2.sam2_image_predictor import SAM2ImagePredictor

CAM_LOCS = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}

# do not edit the spacing of this prompt; the finetuned detector expects this exact string
TEXTS = "car . truck . construction_vehicle . bus . trailer  . motorcycle . bicycle . person . traffic_cone. barrier"

SAM2_CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'


def resolve_sam2_checkpoint(path):
    """Return a local SAM2 checkpoint path; download the official file if it is missing.

    path: str file path or None (None -> <script dir>/checkpoints/sam2_hiera_large.pt)
    -> str path to an existing checkpoint file.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'checkpoints', 'sam2_hiera_large.pt')
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'downloading SAM2 checkpoint to {path}')
        torch.hub.download_url_to_file(SAM2_CHECKPOINT_URL, path, progress=True)
    return path


def parse_args():
    """Parse the CLI arguments -> argparse.Namespace."""
    parser = ArgumentParser()
    parser.add_argument(
        'model',
        type=str,
        help='Grounding DINO config file')
    parser.add_argument(
        '--weights',
        default=None,
        help='Grounding DINO checkpoint path. Default: download '
             'groundingdinofintune.pth from HF oliver0922/GroundingDINOfintune '
             '(the finetuned detector weight).')
    parser.add_argument(
        '--sam2_checkpoint',
        default=None,
        help='SAM2 checkpoint path. Default: <this script dir>/checkpoints/sam2_hiera_large.pt, '
             'downloaded from the official SAM2 release when missing.')
    parser.add_argument('--scene-start', default=0, type=int)
    parser.add_argument('--scene-end', default=797, type=int)
    parser.add_argument('--gpus', default='0', help='comma separated GPU ids, one worker per GPU')
    parser.add_argument('--data_dir', required=True,
                        help='step-1 scene tree, e.g. $OUT/scenes (read only)')
    parser.add_argument('--out_dir', required=True,
                        help='output root, e.g. $OUT/sam. everything is written under here.')
    parser.add_argument('--pred_score_thr', type=float, default=0.4)
    parser.add_argument('--width_thres_1', type=int, default=1200)
    parser.add_argument('--width_thres_2', type=int, default=30)
    parser.add_argument('--iteration_0', type=int, default=50, help='largest object iteration')
    parser.add_argument('--iteration_1', type=int, default=10, help='large object iteration')
    parser.add_argument('--iteration_2', type=int, default=2, help='small object iteration')
    args = parser.parse_args()

    return args


# one set of models per worker process, created on first use
inferencer = None
video_predictor = None
image_predictor = None


def init_worker(gpu_queue, num_workers):
    """Pool initializer: pin this worker to one GPU and cap its CPU threads.

    gpu_queue: multiprocessing.Queue of str GPU ids; num_workers: int pool
    size -> None (sets CUDA_VISIBLE_DEVICES / thread counts as side effects).
    """
    # each worker takes one GPU from the queue. CUDA_VISIBLE_DEVICES must be
    # set before the first CUDA call (torch is already imported, that is fine).
    gpu = gpu_queue.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # keep the machine's load average sane when several workers run together
    n_threads = max(1, (os.cpu_count() or 8) // max(1, num_workers))
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    cv2.setNumThreads(n_threads)
    torch.set_num_threads(n_threads)
    print(f'worker pid={os.getpid()} using GPU {gpu}, {n_threads} cpu threads')


def load_models(args):
    """Build the SAM2 predictors and Grounding DINO inferencer once per worker.

    args: parsed CLI namespace -> None (fills the module-level inferencer,
    video_predictor and image_predictor globals).
    """
    global inferencer, video_predictor, image_predictor

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 on Ampere and newer GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = args.sam2_checkpoint
    model_cfg = "sam2_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device="cuda"))

    inferencer = DetInferencer_SAMV2(model=args.model, weights=args.weights,
                                     device='cuda:0', palette='none')
    inferencer.model.test_cfg.chunked_size = -1


def process_scene(scene_idx, args):
    """Run detection + SAM2 tracking + result drawing for one scene, all 5 cams.

    scene_idx: int scene number; args: parsed CLI namespace -> None (writes
    the per-scene output tree described in the module docstring).
    """
    global inferencer
    if inferencer is None:
        load_models(args)

    start = time.time()
    adaptive_tag = f'{args.width_thres_1}_{args.width_thres_2}_{args.iteration_0}_{args.iteration_1}_{args.iteration_2}'
    scene_dir = os.path.join(args.out_dir, f'scene-{scene_idx}')
    scene_data_dir = os.path.join(args.data_dir, f'scene-{scene_idx}')

    objects_count = 0            # instance ids continue across the 5 cameras
    instance_classname_dict = {}

    for i in range(1, 6):
        cam_name = CAM_LOCS[i]
        video_dir = os.path.join(scene_data_dir, cam_name, 'image')
        cam_dir = os.path.join(scene_dir, cam_name)

        mask_data_dir = os.path.join(cam_dir, 'erosion', 'mask_data')
        json_data_dir = os.path.join(cam_dir, 'erosion', 'json_data')
        CommonUtils.creat_dirs(mask_data_dir)
        CommonUtils.creat_dirs(json_data_dir)

        """
        Step 1: Grounding DINO detection + SAM2 mask tracking -> raw mask/json
        """
        frame_names = [os.path.join(video_dir, f)
                       for f in sorted(os.listdir(video_dir)) if f.endswith('.jpeg')]
        inference_state = video_predictor.init_state(
            video_path=video_dir, offload_video_to_cpu=True, async_loading_frames=True)
        sam2_masks = MaskDictionaryModel()

        objects_count = inferencer(frame_names, 1, args.pred_score_thr, TEXTS,
                                   False, False, image_predictor, video_predictor,
                                   inference_state, sam2_masks, frame_names,
                                   mask_data_dir, json_data_dir, objects_count)

        """
        Step 2: draw every annotated-result flavor + 3D unprojection bins
        """
        this_dict = CommonUtils.draw_all_results(
            video_dir, mask_data_dir, json_data_dir, scene_data_dir, cam_name,
            erosion_result_path=os.path.join(cam_dir, 'erosion', 'result'),
            result_path=os.path.join(cam_dir, 'result'),
            adaptive_result_path=os.path.join(cam_dir, 'adaptive_erosion', adaptive_tag, 'result'),
            vis_path=os.path.join(cam_dir, 'visualization'),
            adaptive_vis_path=os.path.join(cam_dir, 'adaptive' + adaptive_tag, 'visualization'),
            width_threshold_1=args.width_thres_1, width_threshold_2=args.width_thres_2,
            larger_object_iteration=args.iteration_0, large_object_iteration=args.iteration_1,
            small_object_iteration=args.iteration_2)
        instance_classname_dict.update(this_dict)
        print(f'scene-{scene_idx} {cam_name} done ({time.time() - start:.0f}s)')

    with open(os.path.join(scene_dir, 'instance_classname_dict.pkl'), 'wb') as f:
        pickle.dump(instance_classname_dict, f)
    print(f'scene-{scene_idx} finished in {time.time() - start:.0f}s')


def main():
    """CLI entry point: resolve checkpoints, then map scenes over a GPU pool."""
    args = parse_args()
    if args.weights is None:
        # one download in the parent, cached in ~/.cache/huggingface for the workers
        from huggingface_hub import hf_hub_download
        args.weights = hf_hub_download(
            repo_id='oliver0922/GroundingDINOfintune',
            filename='groundingdinofintune.pth')
        print(f'weights: {args.weights}')
    args.sam2_checkpoint = resolve_sam2_checkpoint(args.sam2_checkpoint)
    scene_list = list(range(args.scene_start, args.scene_end + 1))
    gpu_list = [g for g in args.gpus.split(',') if g != '']

    gpu_queue = multiprocessing.Queue()
    for gpu in gpu_list:
        gpu_queue.put(gpu)

    with multiprocessing.Pool(processes=len(gpu_list), initializer=init_worker,
                              initargs=(gpu_queue, len(gpu_list))) as pool:
        pool.map(partial(process_scene, args=args), scene_list, chunksize=1)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()

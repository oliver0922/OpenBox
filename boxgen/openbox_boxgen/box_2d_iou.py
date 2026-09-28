"""Orientation disambiguation against the SAM/GroundingDINO 2D boxes, and the
pcdet CUDA NMS wrapper.  ``torch`` must be imported before the pcdet extension.
"""
import torch
from pcdet.ops.iou3d_nms import iou3d_nms_cuda


def iou_with_2d_gt(projected_points_1, projected_points_2, bbox_2d_list):
    """Return 1 if the image projections of 3D-box hypothesis 1 overlap the 2D
    boxes better (strictly larger summed IoU) than those of hypothesis 2, else 2.

    ``projected_points_*`` map frame_idx -> list (one entry per camera) of
    {cam_loc: (pt1, pt2, pt3, pt4)}; ``bbox_2d_list`` maps frame_idx -> list of
    {cam_loc: [x1, y1, x2, y2]} in the same camera order.  Corners follow the
    image-plane convention (u right, v down)::

        pt1 ------- pt3
         |           |
        pt2 ------- pt4

    so ``[pt1.u, pt1.v, pt4.u, pt4.v]`` is a hypothesis's axis-aligned 2D box.
    IoUs are summed frame by frame, camera by camera; ties pick hypothesis 2.
    """
    iou_1 = 0
    iou_2 = 0

    assert len(bbox_2d_list) == len(projected_points_1)

    for frame_idx in projected_points_1:
        cam_locs = [list(per_cam.keys())[0] for per_cam in projected_points_1[frame_idx]]

        for idx, cam_loc in enumerate(cam_locs):
            bbox1_pt1, _, _, bbox1_pt4 = projected_points_1[frame_idx][idx][cam_loc]
            bbox2_pt1, _, _, bbox2_pt4 = projected_points_2[frame_idx][idx][cam_loc]
            bbox1 = [bbox1_pt1[0], bbox1_pt1[1], bbox1_pt4[0], bbox1_pt4[1]]
            bbox2 = [bbox2_pt1[0], bbox2_pt1[1], bbox2_pt4[0], bbox2_pt4[1]]
            gt = bbox_2d_list[frame_idx][idx][cam_loc]

            iou_1 += calculate_iou(bbox1, gt)
            iou_2 += calculate_iou(bbox2, gt)

    return 1 if iou_1 > iou_2 else 2


def calculate_iou(box1, box2):
    """IoU of two axis-aligned (x1, y1, x2, y2) boxes; 0.0 when they do not overlap."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area


def class_wise_nms_gpu(boxes, scores, thresh, pre_maxsize):
    """Rotated-BEV NMS of one class's (N, 7) CUDA boxes with pcdet's CUDA kernel.

    The top ``pre_maxsize`` boxes by ``scores`` (integer point counts, so ties are
    frequent and broken by the unstable ``Tensor.sort``) enter NMS with BEV-IoU
    threshold ``thresh``.  Returns a CUDA LongTensor of the kept indices into
    ``boxes`` in descending-score order.
    """
    assert boxes.shape[1] == 7  # x, y, z, dx, dy, dz, heading
    order = scores.sort(0, descending=True)[1][:pre_maxsize]  # None keeps all

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))  # filled in place by the kernel
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)

    return order[keep[:num_out].cuda()].contiguous()

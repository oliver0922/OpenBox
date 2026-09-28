"""Writer of the stage-1 result of a scene: one ``openbox_boxes.pkl`` (read back by
``build_pseudo_infos.load_stage1_boxes``)."""
import logging
import os
import pickle

logger = logging.getLogger(__name__)

STAGE1_FILE = "openbox_boxes.pkl"


def save_stage1_outputs(out_dir, boxes_after, class_list, deformable_boxes, single_frame_boxes):
    """Write ``<out_dir>/openbox_boxes.pkl`` and return ``[path]``.

    The pickle is a dict with four keys:

    * ``static_boxes``: float32 (K, 8) = [x, y, z, l, w, h, yaw, num_points] static
      (multi-frame) boxes after NMS, world (frame-0) frame -- None when no aggregated
      static point survived the SDF filter,
    * ``static_classes``: the K fine class names of those rows (None with the above),
    * ``deformable_boxes``: dict load position -> list of ``(class name, float32 box7)``
      deformable (person / bicycle) boxes,
    * ``single_frame_boxes``: per loaded frame, list of ``(class name, float32 box7)``
      single-frame (dynamic + off-surface) boxes.

    boxes_after: float32 (K, 8) or None; class_list: list of str or None;
    deformable_boxes / single_frame_boxes: the per-frame containers above.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, STAGE1_FILE)
    with open(path, "wb") as handle:
        pickle.dump({
            "static_boxes": boxes_after,
            "static_classes": class_list,
            "deformable_boxes": deformable_boxes,
            "single_frame_boxes": single_frame_boxes,
        }, handle)
    logger.info("wrote %s", path)
    return [path]

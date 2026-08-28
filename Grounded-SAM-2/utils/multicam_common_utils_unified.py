import os
import json
import cv2
import numpy as np
import supervision as sv
import torch

# ---------------------------------------------------------------------------
# Instance color palette. Colors are looked up by instance id, so the entry
# order determines the per-instance colors in the annotated images and in the
# 3D bins. Keep the list and its order unchanged (duplicates included).
# ---------------------------------------------------------------------------
point_color_map = 300*[
    [0.25, 0.5, 0.75],
    [0.8, 0.6, 0.4],
    [0.3, 0.7, 0.5],
    [0.9, 0.1, 0.2],
    [0.5, 0.5, 0.5],
    [0.2, 0.3, 0.9],
    [0.7, 0.4, 0.3],
    [0, 1, 1],
    [0.8, 0.2, 0.6],
    [0.4, 0.4, 0.8],
    [0.6, 0.4, 0.2],
    [0.1, 0.9, 0.7],
    [0.6, 0.2, 0.8],
    [0.4, 0.8, 0.1],
    [0.9, 0.4, 0.1],
    [0.2, 0.6, 0.8],
]
Pallete = sv.ColorPalette.from_hex(
    [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in point_color_map])

CAM_IDX = {'FRONT': 1, 'FRONT_LEFT': 2, 'FRONT_RIGHT': 3, 'SIDE_LEFT': 4, 'SIDE_RIGHT': 5}


class CommonUtils:
    @staticmethod
    def creat_dirs(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Path '{path}' did not exist and has been created.")

    @staticmethod
    def erode_fixed(object_mask):
        # fixed erosion: 3x3 cross kernel, 15 iterations
        kernel = np.ones((3, 3), np.uint8)
        kernel[0, 0] = 0
        kernel[0, 2] = 0
        kernel[2, 0] = 0
        kernel[2, 2] = 0
        return cv2.erode(object_mask.astype(np.uint8), kernel, iterations=15).astype(bool)

    @staticmethod
    def erode_adaptive(object_mask, width_threshold_1, width_threshold_2,
                       larger_object_iteration, large_object_iteration, small_object_iteration):
        # adaptive erosion: kernel/iterations depend on the object's max pixel width
        kernel = np.ones((3, 3), np.uint8)
        kernel[0, 0] = 0
        kernel[0, 2] = 0
        kernel[2, 0] = 0
        kernel[2, 2] = 0

        heihgt_idx_list = np.where(object_mask.any(axis=1))[0]
        max_width = 0
        for heihgt_idx in heihgt_idx_list:
            widht_idx_list = np.where(object_mask[heihgt_idx])[0]
            max_width = max(max_width, widht_idx_list[-1] - widht_idx_list[0])
        if max_width > width_threshold_1:
            kernel = np.ones((7, 7), np.uint8)
            for i in range(7):
                if i != 3:
                    kernel[i, 0] = 0
                    kernel[i, 1] = 0
                    kernel[i, 2] = 0
                    kernel[i, 4] = 0
                    kernel[i, 5] = 0
                    kernel[i, 6] = 0
            iterations = larger_object_iteration
        elif max_width > width_threshold_2:
            iterations = large_object_iteration
        else:
            iterations = small_object_iteration
        return cv2.erode(object_mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)

    @staticmethod
    def annotate_frame(image, all_object_boxes, all_object_ids, all_class_names,
                       all_object_masks, pallete):
        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )
        labels = [
            f"{instance_id}: {class_name}"
            for instance_id, class_name in zip(all_object_ids, all_class_names)
        ]
        box_annotator = sv.BoxAnnotator(color=pallete, color_lookup=sv.ColorLookup.CLASS)
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
        label_annotator = sv.LabelAnnotator(color=pallete, color_lookup=sv.ColorLookup.CLASS)
        annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, color=pallete)
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        return annotated_frame

    @staticmethod
    def unproject_to_pointcloud(all_object_masks, all_object_ids, scene_data_path, cam_name,
                                frame_idx, uppc_sam_path, uppc_sam_color_path):
        """Project the 2D instance masks onto the lidar pointcloud and save
        uppc_continuous_sam / uppc_color_continuous_sam bins for this frame."""
        instance_coords_idx_list = list()
        for mask in all_object_masks.astype(int):
            idx = np.where(mask != 0)
            instance_coords_idx_list.append(
                torch.stack((torch.from_numpy(idx[0]), torch.from_numpy(idx[1])), dim=1).to('cuda'))

        points = torch.from_numpy(np.fromfile(
            os.path.join(scene_data_path, 'pointcloud', f'{str(frame_idx).zfill(6)}.bin'),
            dtype=np.float32).reshape(-1, 3))
        projection = np.fromfile(
            os.path.join(scene_data_path, 'pointcloud_projection', f'{str(frame_idx).zfill(6)}.bin'),
            dtype=np.int32, count=-1).reshape([-1, 6])

        # each lidar point can project into two cameras: columns 0-2 and 3-5
        pr1 = projection[:, :3]
        pr2 = projection[:, 3:]
        cam_idx = CAM_IDX[cam_name]
        mask_1 = np.where(pr1[:, 0] == cam_idx)[0]
        mask_2 = np.where(pr2[:, 0] == cam_idx)[0]

        masked_mapping_coords_1 = projection[mask_1][:, 1:3]  # u,v coordinate
        masked_mapping_coords_2 = projection[mask_2][:, 4:6]  # u,v coordinate
        masked_mapping_coords = np.concatenate((masked_mapping_coords_1, masked_mapping_coords_2), axis=0)

        masked_points_1 = points[mask_1]
        masked_points_2 = points[mask_2]
        masked_points = torch.cat((masked_points_1, masked_points_2))
        masked_points_with_instance_idx_1 = torch.cat((masked_points_1, torch.zeros(masked_points_1.shape[0], 1)), dim=1)
        masked_points_with_instance_idx_2 = torch.cat((masked_points_2, torch.zeros(masked_points_2.shape[0], 1)), dim=1)
        masked_points_with_instance_idx = torch.cat((masked_points_with_instance_idx_1, masked_points_with_instance_idx_2))
        masked_points_with_colors = torch.ones_like(masked_points)

        masked_mapping_coords = torch.tensor(masked_mapping_coords).to('cuda')
        # row-major linear key: col < image width, so the key is collision-free
        image_width = all_object_masks.shape[-1]
        for instance_idx, instance_coords_idx in zip(all_object_ids, instance_coords_idx_list):
            new_coords = instance_coords_idx[:, 0] * image_width + instance_coords_idx[:, 1]
            new_masked_mapping_coords = masked_mapping_coords[:, 1] * image_width + masked_mapping_coords[:, 0]
            val = torch.isin(new_masked_mapping_coords, new_coords)
            masked_points_with_instance_idx[val, 3] = instance_idx
            masked_points_with_colors[val, 0] = point_color_map[instance_idx][0]
            masked_points_with_colors[val, 1] = point_color_map[instance_idx][1]
            masked_points_with_colors[val, 2] = point_color_map[instance_idx][2]

        np_masked_points_with_instance_idx = masked_points_with_instance_idx.cpu().detach().numpy()
        np_masked_points_with_colors = masked_points_with_colors.cpu().detach().numpy()
        bg_mask = np.where(np_masked_points_with_instance_idx[:, 3] != 0)

        os.makedirs(uppc_sam_path, exist_ok=True)
        os.makedirs(uppc_sam_color_path, exist_ok=True)
        np_masked_points_with_instance_idx[bg_mask].tofile(
            os.path.join(uppc_sam_path, f'{str(frame_idx).zfill(6)}.bin'))
        np_masked_points_with_colors[bg_mask].tofile(
            os.path.join(uppc_sam_color_path, f'{str(frame_idx).zfill(6)}.bin'))

    @staticmethod
    def draw_all_results(raw_image_path, mask_path, json_path, scene_data_path, cam_name,
                         erosion_result_path, result_path, adaptive_result_path,
                         vis_path, adaptive_vis_path,
                         width_threshold_1=1200, width_threshold_2=30,
                         larger_object_iteration=50, large_object_iteration=10,
                         small_object_iteration=2):
        """Generate every visualization output of one camera in a single pass
        over the saved raw masks (mask_data/json_data):

          erosion_result_path   annotated jpeg, fixed erosion      (== erosion/result)
          result_path           annotated jpeg, no erosion         (== result)
          adaptive_result_path  annotated jpeg, adaptive erosion   (== adaptive_erosion/<P>/result)
          vis_path              3D bins from the no-erosion masks  (== visualization/)
          adaptive_vis_path     3D bins from the adaptive masks    (== adaptive<P>/visualization/)

        Returns {instance_id: class_name} for every instance seen by this camera.
        """
        instance_classname_dict = {}
        for path in (erosion_result_path, result_path, adaptive_result_path):
            CommonUtils.creat_dirs(path)

        raw_image_name_list = os.listdir(raw_image_path)
        raw_image_name_list.sort()

        for raw_image_name in raw_image_name_list:
            image = cv2.imread(os.path.join(raw_image_path, raw_image_name))
            if image is None:
                raise FileNotFoundError("Image file not found.")
            frame_idx = raw_image_name.split(".")[0]
            mask = np.load(os.path.join(mask_path, "mask_" + frame_idx + ".npy"))
            unique_ids = np.unique(mask)

            # per-object bool masks, in ascending instance-id order
            plain_masks = []
            fixed_masks = []
            adaptive_masks = []
            for uid in unique_ids:
                if uid == 0:  # skip background id
                    continue
                object_mask = (mask == uid)
                plain_masks.append(object_mask[None])
                fixed_masks.append(CommonUtils.erode_fixed(object_mask)[None])
                adaptive_masks.append(CommonUtils.erode_adaptive(
                    object_mask, width_threshold_1, width_threshold_2,
                    larger_object_iteration, large_object_iteration, small_object_iteration)[None])

            if len(plain_masks) == 0:
                # nothing detected in this frame: save the raw image as-is, no 3D bins
                for path in (erosion_result_path, result_path, adaptive_result_path):
                    cv2.imwrite(os.path.join(path, raw_image_name), image)
                continue
            plain_masks = np.concatenate(plain_masks, axis=0)
            fixed_masks = np.concatenate(fixed_masks, axis=0)
            adaptive_masks = np.concatenate(adaptive_masks, axis=0)

            # box/class info from the json, sorted by instance id to match the mask order
            all_object_boxes = []
            all_object_ids = []
            all_class_names = []
            with open(os.path.join(json_path, "mask_" + frame_idx + ".json"), "r") as file:
                json_data = json.load(file)
                for obj_id, obj_item in json_data["labels"].items():
                    instance_id = obj_item["instance_id"]
                    if instance_id not in unique_ids:  # not a valid box
                        continue
                    all_object_boxes.append([obj_item["x1"], obj_item["y1"],
                                             obj_item["x2"], obj_item["y2"]])
                    class_name = obj_item["class_name"]
                    instance_classname_dict[instance_id] = class_name
                    all_object_ids.append(instance_id)
                    all_class_names.append(class_name)
            sorted_pair = sorted(zip(all_object_ids, all_class_names, all_object_boxes),
                                 key=lambda pair: pair[0])
            all_object_ids = [pair[0] for pair in sorted_pair]
            all_class_names = [pair[1] for pair in sorted_pair]
            all_object_boxes = [pair[2] for pair in sorted_pair]

            for masks, pallete, path in ((fixed_masks, Pallete, erosion_result_path),
                                         (plain_masks, Pallete, result_path),
                                         (adaptive_masks, Pallete, adaptive_result_path)):
                annotated = CommonUtils.annotate_frame(
                    image, all_object_boxes, all_object_ids, all_class_names, masks, pallete)
                cv2.imwrite(os.path.join(path, raw_image_name), annotated)

            CommonUtils.unproject_to_pointcloud(
                plain_masks, all_object_ids, scene_data_path, cam_name, frame_idx,
                os.path.join(vis_path, 'uppc_continuous_sam'),
                os.path.join(vis_path, 'uppc_color_continuous_sam'))
            CommonUtils.unproject_to_pointcloud(
                adaptive_masks, all_object_ids, scene_data_path, cam_name, frame_idx,
                os.path.join(adaptive_vis_path, 'uppc_continuous_sam'),
                os.path.join(adaptive_vis_path, 'uppc_color_continuous_sam'))
            print(f"frame {frame_idx} results saved")

        return instance_classname_dict

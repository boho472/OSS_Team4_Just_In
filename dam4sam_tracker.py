import numpy as np
import yaml
import torch
import torchvision.transforms.functional as F

from vot.region.raster import calculate_overlaps
from vot.region.shapes import Mask
from vot.region import RegionType
from sam2.build_sam import build_sam2_video_predictor
from collections import OrderedDict
import random
import os
from utils.utils import keep_largest_component, determine_tracker

from pathlib import Path
config_path = Path(__file__).parent / "dam4sam_config.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class DAM4SAMTracker():
    def __init__(self, tracker_name="sam21pp-L"):
        """
        Constructor for the DAM4SAM (2 or 2.1) tracking wrapper.

        Args:
        - tracker_name (str): Name of the tracker to use. Options are:
            - "sam21pp-L": DAM4SAM (2.1) Hiera Large
            - "sam21pp-B": DAM4SAM (2.1) Hiera Base+
            - "sam21pp-S": DAM4SAM (2.1) Hiera Small
            - "sam21pp-T": DAM4SAM (2.1) Hiera Tiny
            - "sam2pp-L": DAM4SAM (2) Hiera Large
            - "sam2pp-B": DAM4SAM (2) Hiera Base+
            - "sam2pp-S": DAM4SAM (2) Hiera Small
            - "sam2pp-T": DAM4SAM (2) Hier Tiny
        """
        self.checkpoint, self.model_cfg = determine_tracker(tracker_name)

        # Image preprocessing parameters
        self.input_image_size = 1024
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
            :, None, None]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
            :, None, None]

        self.predictor = build_sam2_video_predictor(
            self.model_cfg, self.checkpoint, device="cuda:0")
        self.tracking_times = []

    def _prepare_image(self, img_pil):
        # _load_img_as_tensor from SAM2
        img = torch.from_numpy(np.array(img_pil)).to(
            self.inference_state["device"])
        img = img.permute(2, 0, 1).float() / 255.0
        img = F.resize(img, (self.input_image_size, self.input_image_size))
        img = (img - self.img_mean) / self.img_std
        return img

    @torch.inference_mode()
    def init_state_tw(
        self,
    ):
        """Initialize an inference state."""
        compute_device = torch.device("cuda")
        inference_state = {}
        inference_state["images"] = None  # later add, step by step
        inference_state["num_frames"] = 0  # later add, step by step
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = False
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = False
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = None  # later add, step by step
        inference_state["video_width"] = None  # later add, step by step
        inference_state["device"] = compute_device
        # torch.device("cpu")
        inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["adds_in_drm_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        inference_state["frames_tracked_per_obj"] = {}

        self.img_mean = self.img_mean.to(compute_device)
        self.img_std = self.img_std.to(compute_device)

        return inference_state

    @torch.inference_mode()
    def initialize(self, image, init_masks=None, bboxes=None, obj_ids=None):
        """
        Initialize the tracker with the first frame and mask.
        Function builds the DAM4SAM (2.1) tracker and initializes it with the first frame and mask.

        Args:
        - image (PIL Image): First frame of the video.
        - init_mask (numpy array): Binary mask for the initialization

        Returns:
        - out_dict (dict): Dictionary containing the mask for the initialization frame.
        """
        self.frame_index = 0  # Current frame index, updated frame-by-frame
        # Dict to store object sizes per object {obj_id: [sizes]}
        self.object_sizes = {}
        # Dict to store last added frame to DRM per object {obj_id: frame_idx}
        self.last_added = {}

        self.img_width = image.width  # Original image width
        self.img_height = image.height  # Original image height
        self.inference_state = self.init_state_tw()
        self.inference_state["images"] = image
        video_width, video_height = image.size
        self.inference_state["video_height"] = video_height
        self.inference_state["video_width"] = video_width
        prepared_img = self._prepare_image(image)
        self.inference_state["images"] = {0: prepared_img}
        self.inference_state["num_frames"] = 1
        self.predictor.reset_state(self.inference_state)

        # warm up the model
        self.predictor._get_image_feature(
            self.inference_state, frame_idx=0, batch_size=1)

        if init_masks is None and bboxes is None:
            print('Error: initialization state (bbox or mask) is not given.')
            exit(-1)

        # Handle single object case(backward compatibility)
        if init_masks is not None and not isinstance(init_masks, list):
            init_masks = [init_masks]

        if bboxes is not None and not isinstance(bboxes, list):
            bboxes = [bboxes]

        # Determine number of objects
        num_objects = len(
            init_masks) if init_masks is not None else len(bboxes)

        # Set default obj_ids if not provided
        if obj_ids is None:
            obj_ids = list(range(num_objects))

        # Convert bboxes to masks if needed
        if init_masks is None:
            init_masks = []
            for bbox in bboxes:
                mask = self.estimate_mask_from_box(bbox, frame_idx=0)
                init_masks.append(mask)

        # Initialize each object
        out_masks = {}
        for obj_id, init_mask in zip(obj_ids, init_masks):
            _, _, out_mask_logits = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=obj_id,
                mask=init_mask,
            )

            obj_idx = self.inference_state["obj_id_to_idx"][obj_id]
            m = (out_mask_logits[obj_idx, 0] >
                 0).float().cpu().numpy().astype(np.uint8)
            out_masks[obj_id] = m

            # initialize tracking info for this object
            self.object_sizes[obj_id] = []
            self.last_added[obj_id] = -1

        self.inference_state["images"].pop(self.frame_index)

        out_dict = {'pred_masks': out_masks,
                    'obj_ids': obj_ids}

        return out_dict

    @torch.inference_mode()
    def track(self, image, init=False):
        """
        Function to track the object in the next frame.

        Args:
        - image (PIL Image): Next frame of the video.
        - init (bool): Whether the current frame is the initialization frame.

        Returns:
        - out_dict (dict): Dictionary containing the predicted mask for the current frame.
        """
        torch.cuda.empty_cache()
        # Prepare the image for input to the model
        prepared_img = self._prepare_image(image).unsqueeze(0)
        if not init:
            self.frame_index += 1
            self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img

        # Propagate the tracking to the next frame
        # return_all_masks=True returns all predicted (chosen and alternative) masks and corresponding IoUs
        for out in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=self.frame_index, max_frame_num_to_track=0, return_all_masks=True):
            if len(out) == 3:
                # There are 3 outputs when the tracking is done on the initialization frame
                out_frame_idx, out_obj_ids, out_mask_logits = out

                # Extract masks for all objects
                all_masks = {}

                for i, obj_id in enumerate(out_obj_ids):
                    m = (out_mask_logits[i][0] > 0.0).float(
                    ).cpu().numpy().astype(np.uint8)
                    all_masks[obj_id] = m

            else:
                # There are 4 outputs when the tracking is done on a non-initialization frame
                # alternative_masks_ious is a tuple containing chosen and alternative masks and corresponding predicted IoUs
                out_frame_idx, out_obj_ids, out_mask_logits, alternative_masks_ious = out

                # Extract all predicted masks (chosen and alternatives) and IoUs
                alternative_masks, out_all_ious = alternative_masks_ious

                # Process each object independently
                all_masks = {}
                for i, obj_id in enumerate(out_obj_ids):
                    m = (out_mask_logits[i][0] > 0.0).float(
                    ).cpu().numpy().astype(np.uint8)
                    all_masks[obj_id] = m

                    # Index of the chosen predicted mask for this object
                    m_idx = np.argmax(out_all_ious[i])
                    # Predicted IoU of the chosen predicted mask
                    m_iou = out_all_ious[i][m_idx]
                    # Delete the chosen predicted mask from the list of all predicted masks, leading to only alternative masks
                    alternative_masks = [mask for j, mask in enumerate(
                        alternative_masks) if j != m_idx]

                    # Determine if the object ratio between the current frame and the previous frames is within a certain range
                    n_pixels = (m == 1).sum()

                    # Initialize object_sizes for this obj_id if not exists
                    if obj_id not in self.object_sizes:
                        self.object_sizes[obj_id] = []

                    self.object_sizes[obj_id].append(n_pixels)

                    if len(self.object_sizes[obj_id]) > 1 and n_pixels >= 1:
                        obj_sizes_ratio = n_pixels / np.median([
                            size for size in self.object_sizes[obj_id][-300:] if size >= 1
                        ][-10:])
                    else:
                        obj_sizes_ratio = -1

                    # The first condition checks if:
                    #  - the chosen predicted mask has a high predicted IoU,
                    #  - the object size ratio is within a +- 20% range compared to the previous frames,
                    #  - the target is present in the current frame,
                    #  - the last added frame to DRM is more than 5 frames ago or no frame has been added yet
                    if obj_id not in self.last_added:
                        self.last_added[obj_id] = -1

                    if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and n_pixels >= 1 and (self.frame_index - self.last_added[obj_id] > 5 or self.last_added[obj_id] == -1):
                        alternative_masks = [Mask((m_[0][0] > 0.0).cpu().numpy()).rasterize((0, 0, self.img_width - 1, self.img_height - 1)).astype(np.uint8)
                                             for m_ in alternative_masks]

                        # Numpy array of the chosen mask and corresponding bounding box
                        chosen_mask_np = m.copy()
                        chosen_bbox = Mask(chosen_mask_np).convert(
                            RegionType.RECTANGLE)

                        alternative_masks_for_obj = alternative_masks

                        # Delete the parts of the alternative masks that overlap with the chosen mask
                        alternative_masks_for_obj = [np.logical_and(m_, np.logical_not(
                            chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks_for_obj]
                        # Keep only the largest connected component of the processed alternative masks
                        alternative_masks_for_obj = [keep_largest_component(
                            m_) for m_ in alternative_masks_for_obj if np.sum(m_) >= 1]

                        if len(alternative_masks_for_obj) > 0:
                            # Make the union of the chosen mask and the processed alternative masks (corresponding to the largest connected component)
                            alternative_masks_for_obj = [np.logical_or(m_, chosen_mask_np).astype(
                                np.uint8) for m_ in alternative_masks]
                            # Convert the processed alternative masks to bounding boxes to calculate the IoUs bounding box-wise
                            alternative_bboxes = [Mask(m_).convert(
                                RegionType.RECTANGLE) for m_ in alternative_masks_for_obj]
                            # Calculate the IoUs between the chosen bounding box and the processed alternative bounding boxes
                            ious = [calculate_overlaps([chosen_bbox], [bbox])[
                                0] for bbox in alternative_bboxes]

                            # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
                            # That would mean that there are significant differences between the chosen mask and the processed alternative masks,
                            # leading to possible detections of distractors within alternative masks.
                            if np.min(np.array(ious)) <= 0.7:
                                # Update the last added frame index
                                self.last_added[obj_id] = self.frame_index

                                self.predictor.add_to_drm(
                                    inference_state=self.inference_state,
                                    frame_idx=out_frame_idx,
                                    obj_id=obj_id,
                                )

            # Return the predicted mask for the current frame
            out_dict = {'pred_masks': all_masks,
                        'obj_ids': list(all_masks.keys())}

            self.inference_state["images"].pop(self.frame_index)
            return out_dict

    def estimate_mask_from_box(self, bbox, frame_idx=0):
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.predictor._get_image_feature(self.inference_state, frame_idx, 1)

        box = np.array([bbox[0], bbox[1], bbox[0] +
                       bbox[2], bbox[1] + bbox[3]])[None, :]
        box = torch.as_tensor(box, dtype=torch.float,
                              device=current_vision_feats[0].device)

        from sam2.utils.transforms import SAM2Transforms
        _transforms = SAM2Transforms(
            resolution=self.predictor.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        unnorm_box = _transforms.transform_boxes(
            box, normalize=True, orig_hw=(self.img_height, self.img_width)
        )  # Bx2x2

        box_coords = unnorm_box.reshape(-1, 2, 2)
        box_labels = torch.tensor(
            [[2, 3]], dtype=torch.int, device=unnorm_box.device)
        box_labels = box_labels.repeat(unnorm_box.size(0), 1)
        concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = []
        for i in range(2):
            _, b_, c_ = current_vision_feats[i].shape
            high_res_features.append(current_vision_feats[i].permute(
                1, 2, 0).view(b_, c_, feat_sizes[i][0], feat_sizes[i][1]))
        if self.predictor.directly_add_no_mem_embed:
            img_embed = current_vision_feats[2] + self.predictor.no_mem_embed
        else:
            img_embed = current_vision_feats[2]
        _, b_, c_ = current_vision_feats[2].shape
        img_embed = img_embed.permute(1, 2, 0).view(
            b_, c_, feat_sizes[2][0], feat_sizes[2][1])
        low_res_masks, iou_predictions, _, _ = self.predictor.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = _transforms.postprocess_masks(
            low_res_masks, (self.img_height, self.img_width)
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        masks = masks > 0

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(
            0).float().detach().cpu().numpy()

        init_mask = masks_np[0]
        return init_mask

    @torch.inference_mode()
    def add_new_object(self, image, bbox, obj_id):
        """
        Add a new object to track from the current frame (mid-tracking addition).
        Creates independent RAM/DRM memory for the new object.

        Args:
        - image (PIL Image): Current frame where the new object appears
        - bbox (tuple): Bounding box (x, y, w, h) for the new object
        - obj_id (int): Object ID for the new object

        Returns:
        - out_dict (dict): Dictionary containing the mask for the new object
            {
                'pred_mask': mask,
                'obj_id': obj_id
            }
        """
        frame_idx = self.frame_index

        # Prepare image if not already in memory
        prepared_img = self._prepare_image(image)
        if frame_idx not in self.inference_state["images"]:
            self.inference_state["images"][frame_idx] = prepared_img

        # Convert bbox to mask
        mask = self.estimate_mask_from_box(bbox, frame_idx=frame_idx)

        # Add new object to inference_state
        # This will create independent memory structures via _obj_id_to_idx
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask,
        )

        # Get the internal index for this object
        obj_idx = self.inference_state["obj_id_to_idx"][obj_id]
        m = (out_mask_logits[obj_idx, 0] >
             0).float().cpu().numpy().astype(np.uint8)

        # Initialize tracking info for the new object
        self.object_sizes[obj_id] = []
        self.last_added[obj_id] = -1

        # Clean up
        self.inference_state["images"].pop(frame_idx)

        out_dict = {
            'pred_mask': m,
            'obj_id': obj_id
        }

        return out_dict

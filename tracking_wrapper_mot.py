import os
import numpy as np
import cv2
import torch
import time

from collections import OrderedDict

import logging
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from vot.region.raster import calculate_overlaps
from vot.region.shapes import Rectangle

from sam2.utils.misc import fill_holes_in_mask_scores


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def build_sam(config_file, ckpt_path=None, device="cuda", mode="eval", hydra_overrides_extra=[], apply_postprocessing=True):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import os

    # ✅ 추가: Hydra 초기화
    GlobalHydra.instance().clear()
    config_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "sam2/configs/sam2.1"))

    # ✅ with 문으로 감싸기
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # 기존 코드 (들여쓰기만 추가)
        hydra_overrides = []
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                # "++model.fill_hole_area=8",  # [AL] commented out due to different SAM class
            ]
        hydra_overrides.extend(hydra_overrides_extra)

        # Read config and init model
        cfg = compose(config_name=config_file, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    # ✅ with 문 밖으로 (들여쓰기 제거)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def keep_largest_component(mask):
    """
    Keeps only the largest connected component from a binary mask.

    Args:
    - mask (numpy array): 2D binary mask where object pixels are non-zero and background is 0.

    Returns:
    - filtered_mask (numpy array): Binary mask with only the largest connected component.
    """
    # Perform connected components analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    # Find the index of the largest component (excluding background)
    # Skip background (index 0)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    # Create a mask that contains only the largest component
    filtered_mask = np.zeros_like(mask)
    filtered_mask[labels == largest_component] = 1
    return filtered_mask


def load_confs(chkpt_path, model_size):
    if model_size == 'large':
        checkpoint = os.path.join(chkpt_path, 'sam2.1_hiera_large.pt')
        model_cfg = "sam2.1_hiera_l"  # ✅ .yaml 제거, 경로 제거
    elif model_size == 'base':
        checkpoint = os.path.join(chkpt_path, 'sam2.1_hiera_base.pt')
        model_cfg = "sam2.1_hiera_b"  # ✅ 파일명만
    elif model_size == 'small':
        checkpoint = os.path.join(chkpt_path, 'sam2.1_hiera_small.pt')
        model_cfg = "sam2.1_hiera_s"  # ✅ 파일명만
    elif model_size == 'tiny':
        checkpoint = os.path.join(chkpt_path, 'sam2.1_hiera_tiny.pt')
        model_cfg = "sam2.1_hiera_t"  # ✅ 파일명만
    else:
        print('Error: Unknown model size:', model_size)
        exit(-1)

    return checkpoint, model_cfg


class DAM4SAMMOT():
    def __init__(self, model_size='large', checkpoint_dir=None, offload_state_to_cpu=False):

        if not checkpoint_dir:
            checkpoint_dir = './checkpoints'
        checkpoint, model_cfg = load_confs(checkpoint_dir, model_size)

        self.sam = build_sam(model_cfg, checkpoint)

        self.input_image_size = 1024
        self.fill_hole_area = 8

        self._img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
            :, None, None]
        self._img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
            :, None, None]

        self.img_width = None
        self.img_height = None

        self.frame_index = 0
        self.n_frames = None

        self.maskmem_pos_enc = None

        self.output_dict = {'cond_frame_outputs': {},
                            'non_cond_frame_outputs': {},
                            'maskmem_pos_enc': None,
                            'per_obj_dict': {}}

        self.mask_inputs_per_obj = {}
        self.output_dict_per_obj = {}
        self.temp_output_dict_per_obj = {}
        self.consolidated_frame_inds = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }

        self.obj_id_to_idx = OrderedDict()
        self.obj_idx_to_id = OrderedDict()
        self.obj_ids = []

        self.device = torch.device("cuda")
        if offload_state_to_cpu:
            self.storage_device = torch.device("cpu")
        else:
            self.storage_device = torch.device("cuda")

        self.non_overlap_masks_for_mem_enc = False
        self.binarize_mask_from_pts_for_mem_enc = True

        # MOT-specific fields
        self.per_object_outputs_all = {}
        # separate object pointers since they are updated differently
        self.per_object_obj_ptr = {}
        self.next_obj_id = 1
        self.all_obj_ids = []
        # how many objects will be processed together (should not impact tracking)
        self.max_batch_sz = 200
        self.update_delta = 5  # update every delta frames
        self.max_ram = 3
        self.max_drm = 3
        self.use_last = True  # always use last frame in RAM
        # needed for DRM update (to prevent adding twice the same frame to the memory)
        self.add_to_drm_next = {}

        # Object tracking metadata
        # - object_sizes[i]: 객체 i의 크기 히스토리 (DRM 관리용)
        # - last_added[i]: 객체 i가 마지막으로 DRM에 추가된 프레임
        # - per_object_outputs_all[obj_id]: 객체 obj_id의 메모리 (RAM + DRM)
        # - per_object_obj_ptr[obj_id]: 객체 obj_id의 object pointer 히스토리
        # Note: initialize()와 add_new_objects()에서 동기화 필요

    def _prepare_image(self, image):
        # image is RGB PIL image: values on range [0, 255]
        # normalize values, resize/pad output to (3x1024x1024)
        img = np.array(image.convert("RGB").resize(
            (self.input_image_size, self.input_image_size)))
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # normalize
        img -= self._img_mean
        img /= self._img_std
        return img.cuda()

    def _get_features(self, image, num_obj=1):
        # compute backbone features
        backbone_out = self.sam.forward_image(image)
        # vision_features = backbone_out['vision_features']  # (1, 256, 64, 64)
        # list: [(1, 256, 256, 256), (1, 256, 128, 128), (1, 256, 64, 64)]
        vision_pos_enc = backbone_out['vision_pos_enc']
        # list: [(1, 32, 256, 256), (1, 64, 128, 128), (1, 256, 64, 64)]
        backbone_fpn = backbone_out['backbone_fpn']
        # Note: vision_features is the same as backbone_fpn[-1]

        batch_size = num_obj
        for i, feat in enumerate(backbone_fpn):
            backbone_fpn[i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(vision_pos_enc):
            vision_pos_enc[i] = pos.expand(batch_size, -1, -1, -1)

        expanded_backbone_out = {
            "backbone_fpn": backbone_fpn, "vision_pos_enc": vision_pos_enc}
        features = self.sam._prepare_backbone_features(expanded_backbone_out)
        _, vision_feats, vision_pos_embeds, feat_sizes = features

        # vision_feats: [(65536, 1, 32), (16384, 1, 64), (4096, 1, 256)]
        # vision_pos_embeds: [(65536, 1, 256), (16384, 1, 256), (4096, 1, 256)]
        # feat_sizes: actual values: [(256, 256), (128, 128), (64, 64)]
        return vision_feats, vision_pos_embeds, feat_sizes

    def _get_maskmem_pos_enc(self, batch_size=1):
        expanded_maskmem_pos_enc = [
            x.expand(batch_size, -1, -1, -1) for x in self.maskmem_pos_enc
        ]
        return expanded_maskmem_pos_enc

    def _npmask2box(self, mask):
        # mask is a 2D numpy array in a np.uint8 format
        x_ = np.where(mask.sum(0) > 0)[0]
        y_ = np.where(mask.sum(1) > 0)[0]
        x0, x1 = x_.min(), x_.max()
        y0, y1 = y_.min(), y_.max()
        # convert to (x, y0, width, height) bbox format
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

    # *****************************************************************
    # **                        VOT Tracker                          **
    # *****************************************************************

    @torch.no_grad()
    def initialize(self, image, init_regions):
        self.frame_index = 0

        if self.img_width is None or self.img_height is None:
            self.img_width = image.width
            self.img_height = image.height

        # prepare image
        img = self._prepare_image(image)
        img = img.unsqueeze(0)  # (1, 3, 1024, 1024)

        # compute features
        feats, pos, feat_sizes = self._get_features(
            img)  # Note: removed number of objects

        self.object_sizes = []
        self.last_added = []

        # take all unmatched detections and put them in memory for future tracking
        for reg in init_regions:
            # support both - bbox and mask initialization
            if 'mask' in reg:
                mask = reg['mask']
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, dtype=torch.bool)
                mask_H, mask_W = mask.shape
                # add batch and channel dimension
                mask_inputs_orig = mask[None, None]
                mask_inputs_orig = mask_inputs_orig.float().to(feats[0].device)

                # resize the mask if it doesn't match the model's image size
                if mask_H != self.sam.image_size or mask_W != self.sam.image_size:
                    mask_inputs = torch.nn.functional.interpolate(
                        mask_inputs_orig,
                        size=(self.sam.image_size, self.sam.image_size),
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,  # use antialias for downsampling
                    )
                    mask_inputs_ = (mask_inputs >= 0.5).float()
                else:
                    mask_inputs_ = mask_inputs_orig

                point_inputs_ = None

                self.object_sizes.append([])
                self.last_added.append(-1)
            elif 'bbox' in reg:
                bbox = reg['bbox']
                box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

                points = torch.zeros(0, 2, dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int32)
                if points.dim() == 2:
                    points = points.unsqueeze(0)  # add batch dimension
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)  # add batch dimension

                box = torch.tensor(box, dtype=torch.float32,
                                   device=points.device)
                box_coords = box.reshape(1, 2, 2)
                box_labels = torch.tensor(
                    [2, 3], dtype=torch.int32, device=labels.device)
                box_labels = box_labels.reshape(1, 2)
                points = torch.cat([box_coords, points], dim=1)
                labels = torch.cat([box_labels, labels], dim=1)
                points = points / \
                    torch.tensor([image.width, image.height]).to(points.device)

                points = points * self.sam.image_size
                points = points.to(feats[0].device)
                labels = labels.to(feats[0].device)

                point_inputs_ = {"point_coords": points,
                                 "point_labels": labels}
                mask_inputs_ = None

                self.object_sizes.append([])
                self.last_added.append(-1)
            else:
                print('Error: Input region should be mask or rectangle.')
                exit(-1)

            output_dict_ = {'per_obj_dict': {}, 'maskmem_pos_enc': None}
            current_out = self.sam.track_step(
                frame_idx=self.frame_index,
                is_init_cond_frame=True,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs_,
                mask_inputs=mask_inputs_,
                output_dict=output_dict_,
                num_frames=self.n_frames,
                track_in_reverse=False,
                run_mem_encoder=False,  # We might need to put this on True since it is not run separately
                prev_sam_mask_logits=None,
            )
            pred_masks_gpu = current_out["pred_masks"]

            # potentially fill holes in the predicted masks
            if self.fill_hole_area > 0:
                pred_masks_gpu = fill_holes_in_mask_scores(
                    pred_masks_gpu, self.fill_hole_area
                )

            pred_masks = pred_masks_gpu.to(img.device, non_blocking=True)

            high_res_masks = torch.nn.functional.interpolate(
                pred_masks,
                size=(self.sam.image_size, self.sam.image_size),
                mode="bilinear",
                align_corners=False,
            )

            maskmem_features, maskmem_pos_enc = self.sam._encode_new_memory(
                current_vision_feats=feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=current_out['object_score_logits'],
                is_mask_from_pts=True
            )

            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(
                self.storage_device, non_blocking=True)

            if self.maskmem_pos_enc is None:
                self.maskmem_pos_enc = [x[0:1].clone()
                                        for x in maskmem_pos_enc]
                maskmem_pos_enc_ = self.maskmem_pos_enc[0].to(img.device)
                self.output_dict['maskmem_pos_enc'] = maskmem_pos_enc_

            per_obj_dict = {
                "maskmem_features": maskmem_features,  # (1, 64, 64, 64)
                "pred_masks": pred_masks,  # (1, 1, 256, 256)
                "is_init": True, "frame_idx": self.frame_index, "is_drm": False
            }

            # obj_ptr dimmension: (1, 256)
            per_obj_obj_ptr_dict = {
                "obj_ptr": current_out["obj_ptr"], "frame_idx": self.frame_index, "is_init": True}

            self.per_object_outputs_all[self.next_obj_id] = [per_obj_dict]
            self.per_object_obj_ptr[self.next_obj_id] = [per_obj_obj_ptr_dict]
            self.add_to_drm_next[self.next_obj_id] = None
            self.all_obj_ids.append(self.next_obj_id)
            self.next_obj_id += 1

        return None

    @torch.no_grad()
    def track(self, image, active_track_ids=None, dead_track_ids=None):
        """
        Track objects in the current frame.

        Args:
            image: PIL image
            active_track_ids: list of track IDs to track in this frame (None = track all)
            dead_track_ids: list of track IDs to remove from memory (None = remove none)

        Returns:
            dict with 'masks', 'active_track_ids', 'removed_track_ids'
        """
        self.frame_index += 1

        # ========================================
        # STEP 1: 메모리 정리 - Dead 객체 제거
        # ========================================
        removed_track_ids = []
        if dead_track_ids is not None and len(dead_track_ids) > 0:
            print(
                f"\n[Frame {self.frame_index}] Removing {len(dead_track_ids)} dead object(s): {dead_track_ids}")

            for dead_id in dead_track_ids:
                if dead_id in self.all_obj_ids:
                    # 모든 메모리 구조에서 제거
                    self.all_obj_ids.remove(dead_id)
                    self.per_object_outputs_all.pop(dead_id, None)
                    self.per_object_obj_ptr.pop(dead_id, None)
                    self.add_to_drm_next.pop(dead_id, None)

                    removed_track_ids.append(dead_id)

            if len(removed_track_ids) > 0:
                print(
                    f"  ✓ Removed {len(removed_track_ids)} object(s) from memory")
                print(
                    f"  Remaining objects: {len(self.all_obj_ids)} (IDs: {self.all_obj_ids})")

        # ========================================
        # STEP 2: 추적 대상 선택 - Active 객체만
        # ========================================
        if active_track_ids is not None:
            obj_ids_to_track = [
                obj_id for obj_id in self.all_obj_ids
                if obj_id in active_track_ids
            ]
            if len(obj_ids_to_track) < len(self.all_obj_ids):
                print(
                    f"[Frame {self.frame_index}] Tracking {len(obj_ids_to_track)}/{len(self.all_obj_ids)} active object(s)")
        else:
            obj_ids_to_track = self.all_obj_ids.copy()

        # ========================================
        # STEP 3: 이미지 준비 및 Feature 추출
        # ========================================
        img = self._prepare_image(image)
        img = img.unsqueeze(0)
        feats, pos, feat_sizes = self._get_features(img)

        # ========================================
        # STEP 4: SAM2 추적 (필터링된 객체만)
        # ========================================
        output_dict_ = {
            'per_obj_dict': self.per_object_outputs_all,
            'per_obj_obj_ptr_dict': self.per_object_obj_ptr,
            'maskmem_pos_enc': self.output_dict['maskmem_pos_enc'],
            'obj_ids_list': obj_ids_to_track  # ← 필터링된 리스트
        }

        n_runs = ((len(obj_ids_to_track) - 1) // self.max_batch_sz) + 1

        current_out = None
        for i in range(n_runs):
            start_obj_idx = i * self.max_batch_sz
            end_obj_idx = min(len(obj_ids_to_track),
                              i * self.max_batch_sz + self.max_batch_sz)

            obj_ids_list_ = obj_ids_to_track[start_obj_idx:end_obj_idx]
            per_obj_dict_ = {}
            per_obj_obj_ptr_dict_ = {}
            for id_ in obj_ids_list_:
                per_obj_dict_[id_] = output_dict_['per_obj_dict'][id_]
                per_obj_obj_ptr_dict_[id_] = output_dict_[
                    'per_obj_obj_ptr_dict'][id_]
            output_dict_tmp = {
                'per_obj_dict': per_obj_dict_,
                'per_obj_obj_ptr_dict': per_obj_obj_ptr_dict_,
                'maskmem_pos_enc': output_dict_['maskmem_pos_enc'],
                'obj_ids_list': obj_ids_list_
            }

            current_out_tmp = self.sam.track_step(
                frame_idx=self.frame_index,
                is_init_cond_frame=False,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=feat_sizes,
                point_inputs=None,
                mask_inputs=None,
                output_dict=output_dict_tmp,
                num_frames=self.n_frames,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
            current_out_tmp['maskmem_pos_enc'] = None

            if current_out is None:
                current_out = current_out_tmp
            else:
                current_out['pred_masks'] = torch.cat(
                    [current_out['pred_masks'], current_out_tmp['pred_masks']], 0)
                current_out['obj_ptr'] = torch.cat(
                    [current_out['obj_ptr'], current_out_tmp['obj_ptr']], 0)
                current_out['object_score_logits'] = torch.cat(
                    [current_out['object_score_logits'], current_out_tmp['object_score_logits']], 0)
                current_out['maskmem_features'] = torch.cat(
                    [current_out['maskmem_features'], current_out_tmp['maskmem_features']], 0)

        # ========================================
        # STEP 5: 후처리
        # ========================================
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area)

        sz_ = (self.img_height, self.img_width)
        masks_out = torch.nn.functional.interpolate(
            pred_masks_gpu, size=sz_, mode="bilinear", align_corners=False)
        m = [(m_[0] > 0).float().cpu().numpy().astype(np.uint8)
             for m_ in masks_out]
        n_pixels_pos = [m_single.sum() for m_single in m]

        maskmem_features = current_out["maskmem_features"].to(torch.bfloat16)
        maskmem_features = maskmem_features.to(
            self.storage_device, non_blocking=True)

        alternative_masks_all = torch.nn.functional.interpolate(
            current_out["multimasks_logits"], size=sz_, mode="bilinear", align_corners=False
        )
        alternative_masks_all = (
            alternative_masks_all > 0).detach().cpu().numpy().astype(np.uint8)
        all_ious = current_out["ious"].detach().cpu().numpy()

        pred_masks_cpu = pred_masks_gpu.detach().cpu()

        # ========================================
        # STEP 6: 메모리 업데이트 (필터링된 객체만)
        # ========================================
        for obj_idx, obj_id in enumerate(obj_ids_to_track):  # ← 필터링된 리스트

            # DRM 체크
            if self.add_to_drm_next[obj_id]:
                obj_mem = self.per_object_outputs_all[obj_id]
                obj_mem[-1] = self.add_to_drm_next[obj_id]
                self.add_to_drm_next[obj_id] = None
                drm_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem)
                            if (not mem_el['is_init'] and mem_el['is_drm'])]
                if len(drm_idxs) > self.max_drm:
                    obj_mem.pop(drm_idxs[0])

            # 객체가 보이는 경우만 업데이트
            if n_pixels_pos[obj_idx] > 0:
                obj_mem = self.per_object_outputs_all[obj_id]

                # Object pointer 업데이트
                per_obj_obj_ptr_dict = {
                    "obj_ptr": current_out["obj_ptr"][obj_idx].unsqueeze(0),
                    "frame_idx": self.frame_index,
                    "is_init": False
                }
                obj_mem_obj_ptr = self.per_object_obj_ptr[obj_id]
                obj_mem_obj_ptr.append(per_obj_obj_ptr_dict)
                if len(obj_mem_obj_ptr) > self.sam.max_obj_ptrs_in_encoder:
                    rem_idx = None
                    for i, ptr_el in enumerate(obj_mem_obj_ptr):
                        if not ptr_el["is_init"]:
                            rem_idx = i
                            break
                    if rem_idx:
                        obj_mem_obj_ptr.pop(rem_idx)

                # Memory 업데이트
                per_obj_dict = {
                    "maskmem_features": maskmem_features[obj_idx].unsqueeze(0),
                    "pred_masks": pred_masks_cpu[obj_idx].unsqueeze(0).numpy(),
                    "is_init": False,
                    "frame_idx": self.frame_index,
                    "is_drm": False
                }

                # RAM 관리 (기존 로직 유지)
                if self.use_last:
                    ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem)
                                if (not mem_el['is_init'] and not mem_el['is_drm'])]

                    if len(ram_idxs) == 0:
                        obj_mem.append(per_obj_dict)
                    elif (self.frame_index % self.update_delta) == 0:
                        if (obj_mem[ram_idxs[-1]]['frame_idx'] % self.update_delta) == 0:
                            obj_mem.append(per_obj_dict)
                        else:
                            obj_mem[ram_idxs[-1]] = per_obj_dict
                    else:
                        if (obj_mem[ram_idxs[-1]]['frame_idx'] % self.update_delta) == 0:
                            obj_mem.append(per_obj_dict)
                        else:
                            obj_mem[ram_idxs[-1]] = per_obj_dict
                else:
                    if (self.frame_index % self.update_delta) == 0:
                        obj_mem.append(per_obj_dict)

                ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem)
                            if (not mem_el['is_init'] and not mem_el['is_drm'])]
                if len(ram_idxs) > self.max_ram and len(obj_mem) > self.sam.num_maskmem:
                    obj_mem.pop(ram_idxs[0])

                # DRM 관리 (기존 로직 유지)
                if self.max_drm > 0:
                    # object_sizes와 last_added 인덱스 찾기
                    # 주의: all_obj_ids와 object_sizes의 인덱스가 맞지 않을 수 있음
                    try:
                        actual_idx = self.all_obj_ids.index(obj_id)
                        if actual_idx < len(self.object_sizes):
                            m_idx = np.argmax(all_ious[obj_idx])
                            m_iou = all_ious[obj_idx][m_idx]
                            alternative_masks = [mask for i, mask in enumerate(
                                alternative_masks_all[obj_idx]) if i != m_idx]

                            self.object_sizes[actual_idx].append(
                                n_pixels_pos[obj_idx])
                            if len(self.object_sizes[actual_idx]) > 1:
                                obj_sizes_ratio = n_pixels_pos[obj_idx] / np.median([
                                    size for size in self.object_sizes[actual_idx][-300:] if size >= 1
                                ][-10:])
                            else:
                                obj_sizes_ratio = -1

                            if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and \
                                    (self.frame_index - self.last_added[actual_idx] > self.update_delta or self.last_added[actual_idx] == -1):

                                chosen_mask_np = m[obj_idx]
                                chosen_bbox = self._npmask2box(m[obj_idx])

                                alternative_masks = [np.logical_and(m_, np.logical_not(
                                    chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks]
                                alternative_masks = [keep_largest_component(
                                    m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                                if len(alternative_masks) > 0:
                                    alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(
                                        np.uint8) for m_ in alternative_masks]
                                    alternative_bboxes = [self._npmask2box(
                                        m_) for m_ in alternative_masks]
                                    ious = [calculate_overlaps(
                                        [Rectangle(*chosen_bbox)], [Rectangle(*bbox)])[0] for bbox in alternative_bboxes]
                                    if np.min(np.array(ious)) <= 0.7:
                                        self.last_added[actual_idx] = self.frame_index

                                        per_obj_dict_drm = {
                                            "maskmem_features": maskmem_features[obj_idx].unsqueeze(0),
                                            "pred_masks": pred_masks_cpu[obj_idx].unsqueeze(0).numpy(),
                                            "is_init": False,
                                            "frame_idx": self.frame_index,
                                            "is_drm": True
                                        }

                                        if self.frame_index == obj_mem[-1]['frame_idx']:
                                            self.add_to_drm_next[obj_id] = per_obj_dict_drm
                                        else:
                                            obj_mem.append(per_obj_dict_drm)

                                            if len(obj_mem) > self.sam.num_maskmem:
                                                drm_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem)
                                                            if (not mem_el['is_init'] and mem_el['is_drm'])]
                                                if len(drm_idxs) > self.max_drm:
                                                    obj_mem.pop(drm_idxs[0])
                                                else:
                                                    ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem)
                                                                if (not mem_el['is_init'] and not mem_el['is_drm'])]
                                                    obj_mem.pop(ram_idxs[0])
                    except (ValueError, IndexError) as e:
                        # obj_id가 all_obj_ids에 없거나 인덱스 오류
                        pass

        # GPU 메모리 정리
        del pred_masks_gpu, masks_out, feats, pos
        torch.cuda.empty_cache()

        # ========================================
        # STEP 7: 결과 반환
        # ========================================
        outputs = {
            'masks': m,
            'active_track_ids': obj_ids_to_track.copy(),
            'removed_track_ids': removed_track_ids
        }
        return outputs

    @torch.no_grad()
    def add_new_objects(self, frame_idx, image, regions):
        """
        Add multiple new objects to track starting from the specified frame.
        Uses DAM4SAM's per-object memory structure.
        """
        if not regions or len(regions) == 0:
            return []

        print(f"\n{'='*60}")
        print(f"[Frame {frame_idx}] Adding {len(regions)} new object(s)...")
        print(f"{'='*60}")

        # prepare image
        img = self._prepare_image(image)
        img = img.unsqueeze(0)

        # compute features
        feats, pos, feat_sizes = self._get_features(img, num_obj=1)

        new_obj_ids = []

        # Process each new object
        for reg_idx, reg in enumerate(regions):
            print(f"\nProcessing new object {reg_idx + 1}/{len(regions)}...")

            # Process bbox or mask input (동일)
            if 'bbox' in reg:
                bbox = reg['bbox']
                box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

                points = torch.zeros(0, 2, dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int32)
                if points.dim() == 2:
                    points = points.unsqueeze(0)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)

                box = torch.tensor(box, dtype=torch.float32,
                                   device=points.device)
                box_coords = box.reshape(1, 2, 2)
                box_labels = torch.tensor(
                    [2, 3], dtype=torch.int32, device=labels.device)
                box_labels = box_labels.reshape(1, 2)
                points = torch.cat([box_coords, points], dim=1)
                labels = torch.cat([box_labels, labels], dim=1)
                points = points / \
                    torch.tensor([image.width, image.height]).to(points.device)

                points = points * self.sam.image_size
                points = points.to(feats[0].device)
                labels = labels.to(feats[0].device)

                point_inputs_ = {"point_coords": points,
                                 "point_labels": labels}
                mask_inputs_ = None

            elif 'mask' in reg:
                # mask 처리 (동일)
                mask = reg['mask']
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, dtype=torch.bool)
                mask_H, mask_W = mask.shape
                mask_inputs_orig = mask[None, None]
                mask_inputs_orig = mask_inputs_orig.float().to(feats[0].device)

                if mask_H != self.sam.image_size or mask_W != self.sam.image_size:
                    mask_inputs = torch.nn.functional.interpolate(
                        mask_inputs_orig,
                        size=(self.sam.image_size, self.sam.image_size),
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,
                    )
                    mask_inputs_ = (mask_inputs >= 0.5).float()
                else:
                    mask_inputs_ = mask_inputs_orig

                point_inputs_ = None
            else:
                raise ValueError(
                    'Error: region must contain "bbox" or "mask".')

            # ✅ 수정: DAM4SAM의 완전한 메모리 구조 사용
            # 새 객체를 위한 임시 ID 생성 (실제 추가 전)
            temp_obj_id = self.next_obj_id

            # 기존 객체들의 메모리를 포함한 output_dict 구성
            output_dict_ = {
                'per_obj_dict': self.per_object_outputs_all.copy(),  # ✅ 기존 객체들
                'per_obj_obj_ptr_dict': self.per_object_obj_ptr.copy(),  # ✅ 기존 포인터들
                # ✅ 위치 인코딩
                'maskmem_pos_enc': self.output_dict['maskmem_pos_enc'],
                'obj_ids_list': self.all_obj_ids.copy()  # ✅ 기존 객체 ID들
            }

            # 새 객체를 위한 빈 메모리 구조 추가
            output_dict_['per_obj_dict'][temp_obj_id] = []
            output_dict_['per_obj_obj_ptr_dict'][temp_obj_id] = []
            output_dict_['obj_ids_list'].append(temp_obj_id)

            # Run SAM with full DAM4SAM structure
            current_out = self.sam.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=True,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs_,
                mask_inputs=mask_inputs_,
                output_dict=output_dict_,  # ✅ 완전한 구조
                num_frames=self.n_frames,
                track_in_reverse=False,
                run_mem_encoder=False,
                prev_sam_mask_logits=None,
            )

            pred_masks_gpu = current_out["pred_masks"]

            if self.fill_hole_area > 0:
                pred_masks_gpu = fill_holes_in_mask_scores(
                    pred_masks_gpu, self.fill_hole_area
                )

            pred_masks = pred_masks_gpu.to(img.device, non_blocking=True)

            high_res_masks = torch.nn.functional.interpolate(
                pred_masks,
                size=(self.sam.image_size, self.sam.image_size),
                mode="bilinear",
                align_corners=False,
            )

            # Encode memory
            maskmem_features, maskmem_pos_enc = self.sam._encode_new_memory(
                current_vision_feats=feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=current_out['object_score_logits'],
                is_mask_from_pts=True
            )

            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(
                self.storage_device, non_blocking=True)

            if self.maskmem_pos_enc is None:
                self.maskmem_pos_enc = [x[0:1].clone()
                                        for x in maskmem_pos_enc]
                maskmem_pos_enc_ = self.maskmem_pos_enc[0].to(img.device)
                self.output_dict['maskmem_pos_enc'] = maskmem_pos_enc_

            # Create memory dict
            per_obj_dict = {
                "maskmem_features": maskmem_features,
                "pred_masks": pred_masks,
                "is_init": True,
                "frame_idx": frame_idx,
                "is_drm": False
            }

            per_obj_obj_ptr_dict = {
                "obj_ptr": current_out["obj_ptr"],
                "frame_idx": frame_idx,
                "is_init": True
            }

            # Assign new internal ID (temp_obj_id를 실제로 사용)
            new_obj_id = temp_obj_id

            # Add to tracking structures
            self.per_object_outputs_all[new_obj_id] = [per_obj_dict]
            self.per_object_obj_ptr[new_obj_id] = [per_obj_obj_ptr_dict]
            self.add_to_drm_next[new_obj_id] = None
            self.all_obj_ids.append(new_obj_id)

            # Initialize tracking metadata
            self.object_sizes.append([])
            self.last_added.append(-1)

            self.next_obj_id += 1
            new_obj_ids.append(new_obj_id)

            print(
                f"  ✓ Object ID {new_obj_id} initialized with bbox {reg.get('bbox', 'mask')}")

        print(
            f"\n[Frame {frame_idx}] Successfully added {len(new_obj_ids)} object(s)")
        print(
            f"Total tracking objects: {len(self.all_obj_ids)} (IDs: {self.all_obj_ids})")
        print(f"{'='*60}\n")

        return new_obj_ids

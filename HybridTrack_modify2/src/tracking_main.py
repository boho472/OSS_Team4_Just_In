# from model.video_frame_exchange import video_to_frame, frame_to_video
# from model.model_YOLO import use_YOLO_segmentation
# from model.model_ZoeDepth import zoeDepth
# from model.model_3D_convert import convert_to_3D
# from model.model_DAM4SAM import DAM4SAM
from dataset.tracking_dataset import KittiTrackingDataset
from tracker.hybridtrack import HYBRIDTRACK
from configs.config_utils import cfg, cfg_from_yaml_file
from model.model_DAM4SAM import DAM4SAMIntegration
import numpy as np
import os
import multiprocessing
import argparse
import torch
from PIL import Image


def track_one_seq(seq_id, config, video_path, save_frame, used_frame, result_file_name):
    saved_frame = [f for f in os.listdir(
        save_frame) if f.endswith('.jpg') or f.endswith('.png')]
    saved_frame.sort()

    dataset_path = config.dataset_path
    detections_path = config.detections_path
    tracking_type = config.tracking_type
    detections_path += "/" + str(seq_id).zfill(4)

    # ================================================
    # JSON 저장 경로 생성
    # ================================================
    json_output_dir = os.path.join(
        config.save_frame_path, f"{seq_id:04d}_jsons")
    os.makedirs(json_output_dir, exist_ok=True)

    # ================================================
    # DAM4SAM 초기화
    # ================================================
    dam4sam = DAM4SAMIntegration(
        model_size=config.d4sm_model_size if hasattr(
            config, 'd4sm_model_size') else 'tiny',
        checkpoint_dir=config.checkpoint_dir if hasattr(
            config, 'checkpoint_dir') else './checkpoints'
    )

    tracker = HYBRIDTRACK(
        box_type="Kitti", tracking_features=False, config=config)
    dataset = KittiTrackingDataset(
        dataset_path, save_frame, seq_id=seq_id, ob_path=detections_path, type=[tracking_type])

    new_info = []

    for i in range(len(dataset)):
        image_path = os.path.join(save_frame, saved_frame[i])

        # with multiprocessing.Pool() as pool:
        #    results = []
        #    results.append(pool.apply_async(use_YOLO_segmentation(image_path)))
        #    results.append(pool.apply_async(zoeDepth))
        #    final_results = [res.get() for res in results]
        # convert_to_3D()

        _, _, _, _, objects, det_scores, _ = dataset[i]
        mask = det_scores > config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        tracker.tracking(objects[:, :7],
                         features=None,
                         scores=torch.tensor(det_scores),
                         timestamp=i)

        # PIL Image 로드
        image = Image.open(image_path)

        new_obj = tracker.new_obj
        not_new = []
        disappear = []

        if new_info:
            new_info = [j for j in new_info if j["gap_btw_18_nseen"] != 18]

        with open(dataset.ob_txt_path) as f:
            obj_info = f.readlines()
            whole_txt_file = np.array(
                [item.strip().split(' ') for item in obj_info])

        for (obj_num, obj_id) in new_obj:
            tracker.current_frame_ids.remove(obj_id)
            if obj_id in tracker.previous_frame_ids:
                tracker.previous_frame_ids.remove(obj_id)
            dict_key = []
            for dict in new_info:
                dict_key.append(dict["object_id"])
                if dict["object_id"] == obj_id:
                    dict["bbox"]["x"] = float(whole_txt_file[obj_num, 4])
                    dict["bbox"]["y"] = float(whole_txt_file[obj_num, 5])
                    dict["bbox"]["w"] = float(whole_txt_file[obj_num, 6])
                    dict["bbox"]["h"] = float(whole_txt_file[obj_num, 7])
                    dict["gap_btw_18_nseen"] = 0
            if obj_id not in dict_key:
                new_dict = {}
                new_dict["object_id"] = obj_id
                new_dict["bbox"] = {}
                new_dict["bbox"]["x"] = float(whole_txt_file[obj_num, 4])
                new_dict["bbox"]["y"] = float(whole_txt_file[obj_num, 5])
                new_dict["bbox"]["w"] = float(whole_txt_file[obj_num, 6])
                new_dict["bbox"]["h"] = float(whole_txt_file[obj_num, 7])
                new_dict["gap_btw_18_nseen"] = 0
                new_info.append(new_dict)

        for j in tracker.previous_frame_ids:
            new_dict = {}
            for info in new_info:
                if info["object_id"] == j:
                    info["bbox"]["x"] = 0.0
                    info["bbox"]["y"] = 0.0
                    info["bbox"]["w"] = 0.0
                    info["bbox"]["h"] = 0.0
                    info["gap_btw_18_nseen"] += 1

        print(f"Frame {i}: HybridTrack new_info = {new_info}\n")

        # ==============================
        # DAM4SAM 값 처리
        # ==============================
        dam_outputs = dam4sam.process_frame(
            i, new_info, json_output_dir, image)

        # DAM4SAM()


def tracking_val_seq(arg):

    yaml_file = arg.cfg_file
    config = cfg_from_yaml_file(yaml_file, cfg)
    videos_path = config.dataset_path
    video_files = [f for f in os.listdir(
        videos_path) if f.endswith('.mp4') or f. endswith('.avi')]
    # the results saving path
    save_path = config.save_frame_path
    used_frame_path = config.used_frame_path
    result_path = config.save_video_path
    os.makedirs(save_path, exist_ok=True)
    # seq_list = config.tracking_seqs    # the tracking sequences

    for id in range(len(video_files)):
        file_name = video_files[id][:-4]
        video_path = os.path.join(videos_path, video_files[id])
        save_frame = os.path.join(save_path, file_name)
        used_frame = os.path.join(used_frame_path, file_name)
        result_file_name = os.path.join(result_path, video_files[id])
        file_name = int(file_name)

        # video_to_frame(video_path, save_frame)

        track_one_seq(file_name, config, video_path,
                      save_frame, used_frame, result_file_name)

        # frame_to_video(used_frame, result_file_name)

        # print(f"\n{id + 1}번째 영상 처리 완료. [{id + 1}/{len(seq_list)}]\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="",
                        help='specify the config for tracking')
    args = parser.parse_args()
    tracking_val_seq(args)

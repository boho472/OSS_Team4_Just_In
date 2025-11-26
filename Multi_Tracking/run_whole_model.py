from model.video_frame_exchange import video_to_frame, frame_to_video
from model.model_YOLO import use_YOLO_segmentation
from model.model_ZoeDepth import zoeDepth
from model.model_3D_convert import convert_to_3D
from model.model_HybridTrack import hybridTrack
from model.model_DAM4SAM import DAM4SAM
import multiprocessing
import os

videos_path = "data/input_video"
video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4') or f. endswith('.avi')]
save_path = "data/save_frames"
result_path = "data/result_video"

for video_name in video_files:
    file_name = video_name[:-4]
    video_path = os.path.join(videos_path, video_name)
    save_frame = os.path.join(save_path, file_name)
    result_file_name = os.path.join(result_path, video_name)
    
    video_to_frame(video_path, save_frame)
    saved_frame = [f for f in os.listdir(save_frame) if f.endswith('.jpg') or f.endswith('.png')]
    saved_frame.sort()
    for image in saved_frame:
        image_path = os.path.join(save_frame, image)
        with multiprocessing.Pool() as pool:
            results = []
            results.append(pool.apply_async(use_YOLO_segmentation(image_path)))
            results.append(pool.apply_async(zoeDepth))
            final_results = [res.get() for res in results]
        convert_to_3D()
        hybridTrack()
        DAM4SAM()
    frame_to_video(save_frame, result_file_name)
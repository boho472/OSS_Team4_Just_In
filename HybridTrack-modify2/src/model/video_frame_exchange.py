import cv2
import os
import shutil

def video_to_frame(video_path, save_frames, skip_frames = 0):

    if os.path.exists(save_frames):
        shutil.rmtree(save_frames)
    
    os.makedirs(save_frames, exist_ok=True)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"오류: 영상을 열 수 없습니다. 경로를 확인하세요: {video_path}")
        return
    
    count = 0
    saved_count = 0
    
    print("프레임 추출 시작...") 
    
    success, image = video_capture.read()
    
    while success:
        if count % (skip_frames + 1) == 0:
            filename = os.path.join(save_frames, str(count).zfill(6)+".jpg")
            cv2.imwrite(filename, image)  
            saved_count += 1
        success, image = video_capture.read()
        count += 1

    video_capture.release()
    print("-----------------------------------")
    print(f"프레임 추출 완료.")
    print(f"총 읽은 프레임 수: {count}")
    print(f"총 저장된 프레임 수: {saved_count}")


def frame_to_video(save_frames, result_path, fps = 15):
    files = [f for f in os.listdir(save_frames) if f.endswith('.jpg') or f.endswith('.png')]
    if not files:
        print(f"오류: '{save_frames}'에 이미지 파일이 없습니다.")
        return
    
    files.sort()
    
    first_frame_path = os.path.join(save_frames, files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape
    size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(result_path, fourcc, fps, size)
    
    print("영상 파일 생성 시작")

    for i, filename in enumerate(files):
        img_path = os.path.join(save_frames, filename)
        img = cv2.imread(img_path)
        video.write(img)

    video.release()
    print("-----------------------------------")
    print(f"영상 변환 완료")
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# DAM4SAM 임포트
from dam4sam_tracker import DAM4SAMTracker


def select_bboxes_from_frame(video_path, frame_idx=0, num_objects=2):
    """
    특정 프레임에서 마우스로 bbox 선택

    Args:
    - video_path: 비디오 경로
    - frame_idx: bbox를 선택할 프레임 번호
    - num_objects: 선택할 객체 개수

    Returns:
    - bboxes: [(x, y, w, h), ...] 리스트
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read frame {frame_idx}")
        return []

    bboxes = []
    for i in range(num_objects):
        print(f"\nSelecting bbox for object {i+1}/{num_objects}")
        print("- 마우스로 드래그해서 영역 선택")
        print("- SPACE 또는 ENTER: 선택 완료")
        print("- ESC: 다시 선택")

        bbox = cv2.selectROI(
            f"Select Object {i+1}", frame, fromCenter=False, showCrosshair=True)

        if bbox[2] > 0 and bbox[3] > 0:  # 유효한 bbox인지 확인
            bboxes.append(bbox)
            print(f"Object {i+1} bbox: {bbox}")
        else:
            print(f"Invalid bbox for object {i+1}, skipping...")

    cv2.destroyAllWindows()
    return bboxes


def test_multi_object_tracking():
    """
    다중 객체 추적 및 중간 객체 추가 테스트
    """
    # 1. 비디오 경로 설정
    video_path = "multi_test_video.avi"  # 실제 비디오 경로로 변경

    # 2. 첫 프레임(frame 0)에서 초기 객체 bbox 선택
    print("=" * 50)
    print("STEP 1: 첫 프레임에서 초기 객체 선택")
    print("=" * 50)
    bboxes_init = select_bboxes_from_frame(
        video_path, frame_idx=0, num_objects=2)

    if len(bboxes_init) == 0:
        print("Error: No objects selected")
        return

    print(f"\n선택된 초기 bbox: {bboxes_init}")

    # 3. 43프레임에서 새 객체 bbox 선택
    print("\n" + "=" * 50)
    print("STEP 2: 43프레임에서 새로 등장하는 객체 선택")
    print("=" * 50)
    new_bbox_list = select_bboxes_from_frame(
        video_path, frame_idx=43, num_objects=1)

    if len(new_bbox_list) > 0:
        new_bbox = new_bbox_list[0]
        print(f"\n선택된 새 객체 bbox: {new_bbox}")
    else:
        print("Warning: No new object selected, skipping...")
        new_bbox = None

    # 4. 트래커 초기화
    cap = cv2.VideoCapture(video_path)
    tracker = DAM4SAMTracker("sam21pp-L")

    # 5. 첫 프레임 읽기 및 초기화
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    obj_ids_init = [0, 1]

    print("\n" + "=" * 50)
    print("STEP 3: 추적 시작")
    print("=" * 50)
    print("Initializing tracker with 2 objects...")
    result = tracker.initialize(
        frame_pil, bboxes=bboxes_init, obj_ids=obj_ids_init)
    print(f"Initialized objects: {result['obj_ids']}")

    # 6. 프레임 1~42: 두 객체 추적
    all_results = {0: [result['pred_masks'][0]], 1: [result['pred_masks'][1]]}

    print("\nTracking frames 1-42 with 2 objects...")
    for frame_idx in range(1, 43):
        ret, frame = cap.read()
        if not ret:
            print(f"End of video at frame {frame_idx}")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        result = tracker.track(frame_pil)

        # 결과 저장
        for obj_id in result['obj_ids']:
            all_results[obj_id].append(result['pred_masks'][obj_id])

        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: Tracked objects {result['obj_ids']}")

    # 7. 프레임 43: 새 객체 추가
    print("\nFrame 43: Adding new object (obj_id=2)...")
    ret, frame = cap.read()
    if ret and new_bbox is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # 먼저 기존 객체 추적
        result = tracker.track(frame_pil)
        print(f"Tracked existing objects: {result['obj_ids']}")

        # 새 객체 추가
        new_result = tracker.add_new_object(frame_pil, bbox=new_bbox, obj_id=2)
        print(f"Added new object: {new_result['obj_id']}")

        # 결과 저장
        for obj_id in result['obj_ids']:
            all_results[obj_id].append(result['pred_masks'][obj_id])
        all_results[2] = [new_result['pred_mask']]
    else:
        print("Skipping new object addition...")

    # 8. 프레임 44~끝: 세 객체 모두 추적
    print("\nTracking remaining frames with 3 objects...")
    frame_idx = 44
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video at frame {frame_idx}")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        result = tracker.track(frame_pil)

        # 결과 저장
        for obj_id in result['obj_ids']:
            all_results[obj_id].append(result['pred_masks'][obj_id])

        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: Tracked objects {result['obj_ids']}")

        frame_idx += 1

    cap.release()

    # 9. 결과 시각화
    print("\nVisualizing results...")
    visualize_tracking_results(video_path, all_results)

    return all_results


def visualize_tracking_results(video_path, all_results):
    """
    추적 결과 시각화
    """
    cap = cv2.VideoCapture(video_path)

    # 몇 개 프레임만 샘플링해서 시각화
    sample_frames = [0, 20, 42, 43, 60, 80]
    colors = {
        0: (255, 0, 0),    # 빨강
        1: (0, 255, 0),    # 초록
        2: (0, 0, 255),    # 파랑
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, frame_idx in enumerate(sample_frames):
        if frame_idx >= len(all_results[0]):
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 각 객체의 마스크를 오버레이
        overlay = frame_rgb.copy()
        for obj_id, masks in all_results.items():
            if frame_idx < len(masks):
                mask = masks[frame_idx]
                color = colors.get(obj_id, (255, 255, 255))
                overlay[mask > 0] = overlay[mask > 0] * \
                    0.5 + np.array(color) * 0.5

        axes[idx].imshow(overlay)
        axes[idx].set_title(f"Frame {frame_idx+1}")
        axes[idx].axis('off')

    cap.release()
    plt.tight_layout()
    plt.savefig("tracking_results.png")
    print("Results saved to tracking_results.png")
    plt.show()


if __name__ == "__main__":
    test_multi_object_tracking()

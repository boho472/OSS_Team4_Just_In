import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch

# DAM4SAM import
from tracking_wrapper_mot import DAM4SAMMOT


def create_dummy_image(output_dir, n_frames=30, img_size=(1024, 1024)):
    """
    테스트용 더미 이미지 생성

    Args:
        output_dir: 이미지 저장 경로
        n_frames: 생성할 프레임 수
        img_size: 이미지 크기(width, height)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Creating {n_frames} dummy images in '{output_dir}/'...")

    for frame_idx in range(n_frames):
        # 간단한 그라디언트 이미지 생성
        img = np.zeros((*img_size[::-1], 3), dtype=np.uint8)

        # 배경 그라디언트
        for y in range(img_size[1]):
            img[y::, :] = int(255 * y / img_size[1])

        # 객체들을 시뮬레이션 하기 위해 박스 그리기
        # 객체1: 계속 보이는 상황
        x1, y1 = 100 + frame_idx, 100
        cv2_rectangle(img, x1, y1, 50, 50, (255, 0, 0))

        # 객체2: 0~3, 22~29
        if 0 <= frame_idx <= 3 or frame_idx >= 22:
            x2, y2 = 200, 150
            cv2_rectangle(img, x2, y2, 60, 60, (0, 255, 0))

        # 객체 3: 0~7, 26~29
        if 0 <= frame_idx <= 7 or frame_idx >= 26:
            x3, y3 = 300, 300
            cv2_rectangle(img, x3, y3, 55, 55, (0, 255, 255))

        # 객체 4: 0~10, 21~29
        if 0 <= frame_idx <= 10 or frame_idx >= 21:
            x4, y4 = 400, 100
            cv2_rectangle(img, x4, y4, 45, 45, (255, 255, 0))

        # 객체 5: 25~29
        if frame_idx >= 25:
            x5, y5 = 500, 300
            cv2_rectangle(img, x5, y5, 70, 70, (255, 0, 255))

        # PIL Image로 변환하여 저장
        pil_img = Image.fromarray(img)
        img_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        pil_img.save(img_path)

    print(f"✅ Created {n_frames} images")


def cv2_rectangle(img, x, y, w, h, color):
    """OpenCV 없이 사각형 그리기"""
    img[y:y+h, x:x+h] = color


def run_test():
    """테스트 시나리오 실행"""
    print("="*80)
    print("DAM4SAM with HybridTrack JSON Integeration Test")
    print("="*80)

    # 경로 설정
    json_dir = Path("test_jsons")
    image_dir = Path("test_images")

    # 더미 이미지 생성
    create_dummy_image(image_dir, n_frames=30)

    # DAM4SAM 초기화
    print("\n[1] Initializing DAM4SAM...")
    checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
    tracker = DAM4SAMMOT(
        model_size='tiny',
        checkpoint_dir=checkpoint_path
    )
    print("✅ DAM4SAM Initialized")

    # 30개 프레임 처리
    print("\n[2] Processing 30 frames...")
    print("-"*80)

    for frame_idx in range(30):
        # JSON 파일 경로
        json_path = json_dir / f"frame_{frame_idx:06d}.json"

        # 이미지 로드
        img_path = image_dir / f"frame_{frame_idx:06d}.jpg"
        image = Image.open(img_path)

        print(f"\n🚩 Frame {frame_idx:03d}")
        print(f"     JSON: {json_path.name}")
        print(f"     Image: {img_path.name}")

        # JSON 읽어서 HT 결과 확인
        with open(json_path, 'r') as f:
            frame_data = json.load(f)

        ht_results = frame_data['dam4sam_tracking']['HybridTrack_results']
        print(f"   HT detected {len(ht_results)} objects:")
        for ht_obj in ht_results:
            print(
                f"       - obj_id={ht_obj['object_id']}, bbox={ht_obj['bbox']}")

        # DAM4SAM 처리
        outputs = tracker.process_frame_with_ht_json(
            frame_idx, json_path, image)

        # 결과 출력
        print(f"   DAM4SAM tracking {len(outputs['masks'])} objects")

        # JSON에 저장된 결과 확인
        with open(json_path, 'r') as f:
            updated_data = json.load(f)

        dam_results = updated_data['dam4sam_tracking']['DAM4SAM_results']
        print(f"   DAM4SAM results saved: {len(dam_results)} objects")
        for dam_obj in dam_results:
            print(f"   - internal_id={dam_obj['internal_id']}, "
                  f"bbox={dam_obj['bbox']}, pixels={dam_obj['mask_pixels']}")

    print("\n" + "="*80)
    print("🎉 Test 완료!")
    print("="*80)

    # 주요 프레임 결과 확인
    print("\n주요 프레임 결과 확인:")
    print("="*80)

    key_frames = [0, 3, 4, 21, 22, 25, 26]

    for frame_idx in key_frames:
        json_path = json_dir / f"frame_{frame_idx:06d}.json"
        with open(json_path, 'r') as f:
            data = json.load(f)

        ht_count = len(data['dam4sam_tracking']['HybridTrack_results'])
        dam_count = len(data['dam4sam_tracking']['DAM4SAM_results'])

        print(f"\nFrame {frame_idx:03d}")
        print(f"   HT objects: {ht_count}")
        print(f"   DAM objects: {dam_count}")

        if frame_idx == 22:
            print("   ⚡Expected: obj_id=6 should be FILTERED (ID switching)")
            ht_obj_6 = [obj for obj in data['dam4sam_tracking']['HybridTrack_results']
                        if obj['object_id'] == 6]
            if ht_obj_6:
                print(f"     HT detected obj_id=6: {ht_obj_6[0]['bbox']}")
            print(
                f"     DAM tracking {dam_count} objects (should NOT increase)")

        if frame_idx == 26:
            print("   ⚡Expected: obj_id=7 should be FILTERED (ID switching)")
            ht_obj_7 = [obj for obj in data['dam4sam_tracking']['HybridTrack_results']
                        if obj['object_id'] == 7]
            if ht_obj_7:
                print(f"   HT detected obj_id=7: {ht_obj_7[0]['bbox']}")
            print(f"   DAM tracking {dam_count} objects (should NOT increase)")

    print("\n" + "=*80")


if __name__ == "__main__":
    # CUDA 사용 가능한지 확인
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Using CPU (will be slow)")

    run_test()

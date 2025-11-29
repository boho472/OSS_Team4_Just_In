"""
test_integration.py
HybridTrack + DAM4SAM 통합 테스트
"""

import os
import json
import numpy as np
from PIL import Image
from HybridTrack_modify2.src.model.model_DAM4SAM import DAM4SAMIntegration


def create_dummy_image(width=1024, height=1024):
    """더미 이미지 생성"""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # 몇 개 객체 시뮬레이션 (사각형 그리기)
    # 객체 1
    img[100:150, 100:150] = [255, 0, 0]  # 빨강
    # 객체 2
    img[200:260, 200:260] = [0, 255, 0]  # 초록

    return Image.fromarray(img)


def create_dummy_ht_new_info(frame_idx):
    """HybridTrack new_info 시뮬레이션"""

    if frame_idx == 0:
        # 첫 프레임: 2개 객체 등장
        return [
            {
                "object_id": 1,
                "bbox": {"x": 100, "y": 100, "w": 50, "h": 50},
                "gap_btw_18_nseen": 0
            },
            {
                "object_id": 2,
                "bbox": {"x": 200, "y": 200, "w": 60, "h": 60},
                "gap_btw_18_nseen": 0
            }
        ]

    elif frame_idx < 5:
        # 프레임 1~4: 기존 2개 유지
        return [
            {
                "object_id": 1,
                "bbox": {"x": 100 + frame_idx*5, "y": 100, "w": 50, "h": 50},
                "gap_btw_18_nseen": 0
            },
            {
                "object_id": 2,
                "bbox": {"x": 200, "y": 200 + frame_idx*3, "w": 60, "h": 60},
                "gap_btw_18_nseen": 0
            }
        ]

    elif frame_idx == 5:
        # 프레임 5: 새로운 객체 3 등장! (동적 추가 테스트)
        return [
            {
                "object_id": 1,
                "bbox": {"x": 100 + frame_idx*5, "y": 100, "w": 50, "h": 50},
                "gap_btw_18_nseen": 0
            },
            {
                "object_id": 2,
                "bbox": {"x": 200, "y": 200 + frame_idx*3, "w": 60, "h": 60},
                "gap_btw_18_nseen": 0
            },
            {
                "object_id": 3,  # ← 새 객체!
                "bbox": {"x": 300, "y": 300, "w": 55, "h": 55},
                "gap_btw_18_nseen": 0
            }
        ]

    else:
        # 프레임 6~9: 3개 모두 유지
        return [
            {
                "object_id": 1,
                "bbox": {"x": 100 + frame_idx*5, "y": 100, "w": 50, "h": 50},
                "gap_btw_18_nseen": 0
            },
            {
                "object_id": 2,
                "bbox": {"x": 200, "y": 200 + frame_idx*3, "w": 60, "h": 60},
                "gap_btw_18_nseen": 0
            },
            {
                "object_id": 3,
                "bbox": {"x": 300, "y": 300 + (frame_idx-5)*2, "w": 55, "h": 55},
                "gap_btw_18_nseen": 0
            }
        ]


def test_integration():
    """통합 테스트 실행"""

    print("="*80)
    print("HybridTrack + DAM4SAM Integration Test")
    print("="*80)

    # 테스트 디렉토리 생성
    test_dir = "test_integration"
    json_dir = os.path.join(test_dir, "jsons")
    os.makedirs(json_dir, exist_ok=True)

    # DAM4SAM 초기화
    print("\n[1] Initializing DAM4SAM...")
    dam4sam = DAM4SAMIntegration(
        model_size='tiny',
        checkpoint_dir='./checkpoints'
    )

    # 10 프레임 테스트
    n_frames = 10
    print(f"\n[2] Processing {n_frames} frames...")
    print("-"*80)

    for frame_idx in range(n_frames):
        print(f"\n🚩 Frame {frame_idx}")

        # 더미 이미지 생성
        image = create_dummy_image()

        # HybridTrack new_info 시뮬레이션
        new_info = create_dummy_ht_new_info(frame_idx)

        print(f"   HybridTrack detected {len(new_info)} objects:")
        for obj in new_info:
            print(f"     - obj_id={obj['object_id']}, bbox={obj['bbox']}")

        # DAM4SAM 처리
        try:
            dam_outputs = dam4sam.process_frame(
                frame_idx, new_info, json_dir, image
            )

            print(f"   ✅ DAM4SAM tracked {len(dam_outputs['masks'])} objects")

            # JSON 결과 확인
            json_path = os.path.join(json_dir, f"frame_{frame_idx:06d}.json")
            with open(json_path, 'r') as f:
                result = json.load(f)

            dam_results = result['dam4sam_tracking']['DAM4SAM_results']
            print(
                f"   DAM4SAM results: {len(dam_results)} objects saved to JSON")
            for dam_obj in dam_results:
                print(f"     - internal_id={dam_obj['internal_id']}, "
                      f"bbox={dam_obj['bbox']}, pixels={dam_obj['mask_pixels']}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break

    print("\n" + "="*80)
    print("🎉 Test Complete!")
    print("="*80)

    # 주요 프레임 검증
    print("\n[3] Key Frame Verification:")
    print("-"*80)

    key_frames = [0, 4, 5, 9]
    for frame_idx in key_frames:
        json_path = os.path.join(json_dir, f"frame_{frame_idx:06d}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

            ht_count = len(data['dam4sam_tracking']['HybridTrack_results'])
            dam_count = len(data['dam4sam_tracking']['DAM4SAM_results'])

            print(f"\nFrame {frame_idx}:")
            print(f"  HT objects: {ht_count}")
            print(f"  DAM objects: {dam_count}")

            if frame_idx == 5:
                print(f"  ⚡ Expected: New object (id=3) should be added dynamically")


if __name__ == "__main__":
    import torch

    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available. Using CPU (will be slow)")

    test_integration()

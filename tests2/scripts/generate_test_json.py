import json
from pathlib import Path


def generate_test_scenario():
    """
    테스트 시나리오 (총 30프레임):

    HT는 매 프레임마다 현재 화면에 보이는 객체만 기록
    DAM4SAM은 매 프레임마다 mask 존재 여부로 판단

    객체 1: Baseline (계속 보임, 계속 추적)

    객체 2: ID Switching 케이스 1
    - DAM4SAM: 계속 추적 (mask 유지)
    - HT: obj_id=2 → (사라짐) → obj_id=6 (새 ID!)
    - Frame 22: DAM4SAM이 obj_id=6 요청을 mask 존재로 필터링

    객체 3: ID Switching 케이스 2
    - DAM4SAM: 계속 추적 (mask 유지)
    - HT: obj_id=3 → (사라짐) → obj_id=7 (새 ID!)
    - Frame 26: 필터링

    객체 4: 정상 케이스 (ID 유지)
    - DAM4SAM: 계속 추적
    - HT: obj_id=4 → (사라짐) → obj_id=4 (같은 ID)
    - Frame 21: mask 있지만 같은 ID라 상관없음

    객체 5: 진짜 새 객체
    - Frame 25: 처음 등장
    - DAM4SAM: mask 없음 → 초기화
    """

    output_dir = Path("test_jsons")
    output_dir.mkdir(exist_ok=True)

    for frame_idx in range(30):
        frame_data = {
            "frame_number": frame_idx,
            "dam4sam_tracking": {
                "HybridTrack_results": [],
                "DAM4SAM_results": []
            }
        }

        # 객체 1: 계속 등장
        frame_data["dam4sam_tracking"]["HybridTrack_results"].append({
            "object_id": 1,
            "bbox": {"x": 100 + frame_idx, "y": 100, "w": 50, "h": 50}
        })

        # 객체 2: 0~3, 22~29
        if 0 <= frame_idx <= 3:
            frame_data["dam4sam_tracking"]["HybridTrack_results"].append({
                "object_id": 2,
                "bbox": {"x": 200, "y": 150, "w": 60, "h": 60}
            })
        elif frame_idx >= 22:
            frame_data["dam4sam_tracking"]["HybridTrack_results"].append({
                "object_id": 6,  # HT가 새 ID 부여!
                "bbox": {"x": 202, "y": 151, "w": 60, "h": 60}
            })

        # 객체 3: 0~7, 26~29
        if 0 <= frame_idx <= 7:
            frame_data["dam4sam_tracking"]["HybridTrack_results"].append({
                "object_id": 3,
                "bbox": {"x": 300, "y": 200, "w": 55, "h": 55}
            })
        elif frame_idx >= 26:
            frame_data["dam4sam_tracking"]["HybridTrack_results"].append({
                "object_id": 7,  # 새 ID!
                "bbox": {"x": 301, "y": 201, "w": 55, "h": 55}
            })

        # 객체 4: 0~10, 21~29
        if 0 <= frame_idx <= 10:
            frame_data["dam4sam_tracking"]["HybridTrack_results"].append({
                "object_id": 4,
                "bbox": {"x": 400, "y": 100, "w": 45, "h": 45}
            })
        elif frame_idx >= 21:
            frame_data["dam4sam_tracking"]["HybridTrack_results"].append({
                "object_id": 4,  # ID 유지
                "bbox": {"x": 401, "y": 101, "w": 45, "h": 45}
            })

        # 객체 5: 25~29
        if frame_idx >= 25:
            frame_data["dam4sam_tracking"]["HybridTrack_results"].append({
                "object_id": 5,
                "bbox": {"x": 500, "y": 300, "w": 70, "h": 70}
            })

        # JSON 저장
        json_path = output_dir / f"frame_{frame_idx:06d}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(frame_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Generated 30 test JSON files in '{output_dir}/'")
    print("\n" + "="*80)
    print("📋 TEST SCENARIO - Simplified")
    print("="*80)
    print("\nHT records only visible objects each frame")
    print("DAM4SAM checks mask existence for each HT request\n")

    print("[Object 1] Frame 0~29: obj_id=1 (continuous)")
    print("[Object 2] Frame 0~3: obj_id=2 → Frame 22~29: obj_id=6 (ID switch!)")
    print("[Object 3] Frame 0~7: obj_id=3 → Frame 26~29: obj_id=7 (ID switch!)")
    print("[Object 4] Frame 0~10: obj_id=4 → Frame 21~29: obj_id=4 (ID kept)")
    print("[Object 5] Frame 25~29: obj_id=5 (new object)")

    print("\n💡 Expected DAM4SAM behavior:")
    print("  Frame 22: obj_id=6 → mask exists (from obj_id=2) → FILTER ⭐")
    print("  Frame 26: obj_id=7 → mask exists (from obj_id=3) → FILTER ⭐")
    print("  Frame 21: obj_id=4 → mask exists but same ID → Continue tracking")
    print("  Frame 25: obj_id=5 → no mask → INITIALIZE")
    print("="*80)


if __name__ == "__main__":
    generate_test_scenario()

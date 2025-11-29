import json
from pathlib import Path

def create_edge_case_jsons():
    """
    엣지 케이스 테스트 JSON 생성
    
    시나리오:
    - Frame 0: 2개 객체 동시 초기화
    - Frame 10: 3개 객체 동시 새로 등장
    - Frame 15: 2개 객체 동시 ID switching
    """
    
    output_dir = Path("test_jsons_edge")
    output_dir.mkdir(exist_ok=True)
    
    for frame_idx in range(20):
        frame_data = {
            "frame_number": frame_idx,
            "dam4sam_tracking": {
                "HybridTrack_results": [],
                "DAM4SAM_results": []
            }
        }
        
        # Frame 0: 2개 객체 동시 초기화
        if frame_idx == 0:
            frame_data["dam4sam_tracking"]["HybridTrack_results"] = [
                {"object_id": 1, "bbox": {"x": 100, "y": 100, "w": 50, "h": 50}},
                {"object_id": 2, "bbox": {"x": 200, "y": 200, "w": 50, "h": 50}}
            ]
        
        # Frame 1~9: 계속 추적
        elif 1 <= frame_idx <= 9:
            frame_data["dam4sam_tracking"]["HybridTrack_results"] = [
                {"object_id": 1, "bbox": {"x": 100 + frame_idx, "y": 100, "w": 50, "h": 50}},
                {"object_id": 2, "bbox": {"x": 200 + frame_idx, "y": 200, "w": 50, "h": 50}}
            ]
        
        # Frame 10: 3개 객체 동시 새로 등장!
        elif frame_idx == 10:
            frame_data["dam4sam_tracking"]["HybridTrack_results"] = [
                {"object_id": 1, "bbox": {"x": 110, "y": 100, "w": 50, "h": 50}},
                {"object_id": 2, "bbox": {"x": 210, "y": 200, "w": 50, "h": 50}},
                {"object_id": 3, "bbox": {"x": 300, "y": 300, "w": 60, "h": 60}},  # 새 객체 1
                {"object_id": 4, "bbox": {"x": 400, "y": 400, "w": 60, "h": 60}},  # 새 객체 2
                {"object_id": 5, "bbox": {"x": 500, "y": 500, "w": 60, "h": 60}}   # 새 객체 3
            ]
        
        # Frame 11~14: 5개 모두 추적
        elif 11 <= frame_idx <= 14:
            frame_data["dam4sam_tracking"]["HybridTrack_results"] = [
                {"object_id": 1, "bbox": {"x": 100 + frame_idx, "y": 100, "w": 50, "h": 50}},
                {"object_id": 2, "bbox": {"x": 200 + frame_idx, "y": 200, "w": 50, "h": 50}},
                {"object_id": 3, "bbox": {"x": 300 + frame_idx - 10, "y": 300, "w": 60, "h": 60}},
                {"object_id": 4, "bbox": {"x": 400 + frame_idx - 10, "y": 400, "w": 60, "h": 60}},
                {"object_id": 5, "bbox": {"x": 500 + frame_idx - 10, "y": 500, "w": 60, "h": 60}}
            ]
        
        # Frame 15: 2개 객체 동시 ID switching!
        elif frame_idx == 15:
            frame_data["dam4sam_tracking"]["HybridTrack_results"] = [
                {"object_id": 1, "bbox": {"x": 115, "y": 100, "w": 50, "h": 50}},
                {"object_id": 6, "bbox": {"x": 216, "y": 201, "w": 50, "h": 50}},  # 2 → 6 (ID switch!)
                {"object_id": 7, "bbox": {"x": 306, "y": 301, "w": 60, "h": 60}},  # 3 → 7 (ID switch!)
                {"object_id": 4, "bbox": {"x": 406, "y": 400, "w": 60, "h": 60}},
                {"object_id": 5, "bbox": {"x": 506, "y": 500, "w": 60, "h": 60}}
            ]
        
        # Frame 16~19: 계속 추적
        elif frame_idx >= 16:
            frame_data["dam4sam_tracking"]["HybridTrack_results"] = [
                {"object_id": 1, "bbox": {"x": 100 + frame_idx, "y": 100, "w": 50, "h": 50}},
                {"object_id": 6, "bbox": {"x": 200 + frame_idx, "y": 200, "w": 50, "h": 50}},
                {"object_id": 7, "bbox": {"x": 300 + frame_idx - 10, "y": 300, "w": 60, "h": 60}},
                {"object_id": 4, "bbox": {"x": 400 + frame_idx - 10, "y": 400, "w": 60, "h": 60}},
                {"object_id": 5, "bbox": {"x": 500 + frame_idx - 10, "y": 500, "w": 60, "h": 60}}
            ]
        
        # JSON 저장
        json_path = output_dir / f"frame_{frame_idx:06d}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(frame_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated 20 edge case test JSONs in '{output_dir}/'")
    print("\n📋 Edge Case Scenario:")
    print("  Frame 0:  2 objects initialize simultaneously")
    print("  Frame 10: 3 NEW objects appear simultaneously")
    print("  Frame 15: 2 objects ID switch simultaneously (2→6, 3→7)")
    print("\n💡 Expected behavior:")
    print("  Frame 10: DAM should add 3 new objects (2 → 5 total)")
    print("  Frame 15: DAM should FILTER obj_id 6 and 7 (keep 5 total)")


if __name__ == "__main__":
    create_edge_case_jsons()
"""
순수 DAM4SAM 성능 테스트
- HT bbox 재입력 없이 DAM4SAM만 단독 실행
- 빠른 움직임에서의 한계 측정
"""

from pathlib import Path
import sys
from PIL import Image
import numpy as np
import json
import os
import cv2

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from model.tracking_wrapper_mot import DAM4SAMMOT


def create_dummy_image_with_objects(frame_idx, width=1024, height=1024):
    """
    실제 객체가 그려진 더미 이미지 생성 (25프레임 시나리오)
    
    시나리오:
    - 객체 1 (빨강): 속도 변화 테스트
    - 객체 2 (초록): 일정 속도 제어군
    - 객체 3 (파랑): Frame 5부터 등장
    - 객체 4 (노랑): Frame 10부터 등장
    """
    # 그라디언트 배경
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        intensity = int(255 * y / height)
        img[y, :] = [intensity // 3, intensity // 2, intensity]
    
    # 객체 1: 속도 변화 테스트
    if frame_idx < 5:
        x1 = 100 + frame_idx * 5
    elif frame_idx < 15:
        base_x = 100 + 4 * 5
        x1 = base_x + (frame_idx - 4) * 20
    elif frame_idx < 20:
        base_x = 100 + 4 * 5 + 10 * 20
        x1 = base_x + (frame_idx - 14) * 30
    else:
        base_x = 100 + 4 * 5 + 10 * 20 + 5 * 30
        x1 = base_x + (frame_idx - 19) * 10
    
    y1 = 100
    cv2.rectangle(img, (x1, y1), (x1 + 50, y1 + 50), (0, 0, 255), -1)
    
    # 객체 2: 일정 속도
    x2 = 200
    y2 = 200 + frame_idx * 3
    cv2.rectangle(img, (x2, y2), (x2 + 60, y2 + 60), (0, 255, 0), -1)
    
    # 객체 3: Frame 5부터
    if frame_idx >= 5:
        x3 = 300
        y3 = 300 + (frame_idx - 5) * 2
        cv2.rectangle(img, (x3, y3), (x3 + 55, y3 + 55), (255, 0, 0), -1)
    
    # 객체 4: Frame 10부터
    if frame_idx >= 10:
        x4 = 400 + (frame_idx - 10) * 4
        y4 = 100 + (frame_idx - 10) * 3
        cv2.rectangle(img, (x4, y4), (x4 + 45, y4 + 45), (0, 255, 255), -1)
    
    return img


def get_ground_truth_bbox(frame_idx):
    """
    Ground Truth bbox 반환 (HT 시뮬레이션)
    
    Returns:
        dict: {obj_id: {'bbox': [x, y, w, h], 'created_frame': int}}
    """
    ground_truth = {}
    
    # 객체 1
    if frame_idx < 5:
        x1 = 100 + frame_idx * 5
    elif frame_idx < 15:
        base_x = 100 + 4 * 5
        x1 = base_x + (frame_idx - 4) * 20
    elif frame_idx < 20:
        base_x = 100 + 4 * 5 + 10 * 20
        x1 = base_x + (frame_idx - 14) * 30
    else:
        base_x = 100 + 4 * 5 + 10 * 20 + 5 * 30
        x1 = base_x + (frame_idx - 19) * 10
    
    ground_truth[1] = {
        'bbox': [x1, 100, 50, 50],
        'created_frame': 0
    }
    
    # 객체 2
    ground_truth[2] = {
        'bbox': [200, 200 + frame_idx * 3, 60, 60],
        'created_frame': 0
    }
    
    # 객체 3
    if frame_idx >= 5:
        ground_truth[3] = {
            'bbox': [300, 300 + (frame_idx - 5) * 2, 55, 55],
            'created_frame': 5
        }
    
    # 객체 4
    if frame_idx >= 10:
        ground_truth[4] = {
            'bbox': [400 + (frame_idx - 10) * 4, 
                    100 + (frame_idx - 10) * 3, 45, 45],
            'created_frame': 10
        }
    
    return ground_truth


def compute_bbox_from_mask(mask):
    """
    Mask로부터 bbox 계산
    
    Args:
        mask: binary mask (H, W)
    
    Returns:
        [x, y, w, h] or None if empty
    """
    coords = np.argwhere(mask > 0)
    
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return [int(x_min), int(y_min), 
            int(x_max - x_min + 1), int(y_max - y_min + 1)]


def compute_bbox_iou(bbox1, bbox2):
    """
    두 bbox의 IoU 계산
    
    Args:
        bbox1, bbox2: [x, y, w, h]
    
    Returns:
        float: IoU
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_dam_to_gt(dam_masks, gt_bboxes, iou_threshold=0.3):
    """
    DAM4SAM masks를 Ground Truth bbox와 매칭
    
    Args:
        dam_masks: list of masks from DAM4SAM
        gt_bboxes: dict {gt_id: {'bbox': [x,y,w,h]}}
        iou_threshold: matching threshold
    
    Returns:
        dict: {dam_internal_id: (gt_id, iou) or None}
    """
    matching = {}
    
    for dam_idx, mask in enumerate(dam_masks):
        # DAM mask → bbox
        dam_bbox = compute_bbox_from_mask(mask)
        
        if dam_bbox is None:
            matching[dam_idx] = None
            continue
        
        # GT bbox들과 비교
        best_iou = 0
        best_gt_id = None
        
        for gt_id, gt_info in gt_bboxes.items():
            gt_bbox = gt_info['bbox']
            iou = compute_bbox_iou(dam_bbox, gt_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_iou >= iou_threshold:
            matching[dam_idx] = (best_gt_id, best_iou)
        else:
            matching[dam_idx] = None
    
    return matching


def test_pure_dam4sam():
    """순수 DAM4SAM 성능 테스트"""
    
    print("=" * 80)
    print("Pure DAM4SAM Performance Test")
    print("Scenario: NO HybridTrack bbox re-input")
    print("Goal: Measure DAM4SAM's standalone tracking capability")
    print("=" * 80)
    
    # 테스트 디렉토리 생성
    test_dir = "test_pure_dam4sam"
    image_dir = os.path.join(test_dir, "images")
    result_dir = os.path.join(test_dir, "results")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # DAM4SAM 초기화
    print("\n[1] Initializing Pure DAM4SAM...")
    tracker = DAM4SAMMOT(
        model_size='tiny',
        checkpoint_dir='./checkpoints'
    )
    print("✅ DAM4SAM initialized")
    
    # 테스트 프레임 수
    n_frames = 25
    print(f"\n[2] Processing {n_frames} frames (Pure DAM4SAM only)...")
    print("-" * 80)
    
    # 추적 기록
    tracking_history = {
        'dam_internal_ids': [],  # 각 프레임의 internal IDs
        'gt_matching': [],       # GT와의 매칭 결과
        'id_switches': []        # ID switching 발생 지점
    }
    
    # GT ID → DAM internal ID 매핑 (시간에 따라 변할 수 있음)
    gt_to_dam_history = {1: [], 2: [], 3: [], 4: []}
    
    for frame_idx in range(n_frames):
        print(f"\n{'='*60}")
        print(f"🚩 Frame {frame_idx}")
        print(f"{'='*60}")
        
        # 이미지 생성
        img_array = create_dummy_image_with_objects(frame_idx)
        image = Image.fromarray(img_array)
        
        # 이미지 저장
        img_path = os.path.join(image_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # Ground Truth
        gt_bboxes = get_ground_truth_bbox(frame_idx)
        
        print(f"\nGround Truth: {len(gt_bboxes)} objects")
        for gt_id, gt_info in gt_bboxes.items():
            bbox = gt_info['bbox']
            print(f"  GT-{gt_id}: bbox={bbox}")
        
        # ========================================
        # DAM4SAM 실행 (순수)
        # ========================================
        
        if frame_idx == 0:
            # Frame 0: 초기화만 GT bbox 사용
            print("\n🔧 Initializing DAM4SAM with GT bboxes...")
            
            init_regions = []
            for gt_id in sorted(gt_bboxes.keys()):
                bbox = gt_bboxes[gt_id]['bbox']
                init_regions.append({'bbox': bbox})
            
            tracker.initialize(image, init_regions)
            outputs = tracker.track(image)
            
            print(f"✅ Initialized {len(init_regions)} objects")
        
        else:
            # Frame 1+: DAM4SAM 혼자 추적!
            print("\n🤖 Pure DAM4SAM tracking (no HT input)...")
            
            outputs = tracker.track(image)
        
        # ========================================
        # 결과 분석
        # ========================================
        
        masks = outputs['masks']
        print(f"\n📊 DAM4SAM Output: {len(masks)} masks")
        
        for idx, mask in enumerate(masks):
            pixels = np.sum(mask > 0)
            print(f"  DAM-{idx}: {pixels} pixels")
        
        # GT와 매칭
        matching = match_dam_to_gt(masks, gt_bboxes)
        
        print(f"\n🔗 Matching with Ground Truth:")
        for dam_idx, match_result in matching.items():
            if match_result:
                gt_id, iou = match_result
                print(f"  DAM-{dam_idx} ← GT-{gt_id} (IoU={iou:.3f})")
                
                # 매핑 기록
                gt_to_dam_history[gt_id].append({
                    'frame': frame_idx,
                    'dam_id': dam_idx,
                    'iou': iou
                })
            else:
                print(f"  DAM-{dam_idx} ← ⚠️ No match (lost or spurious)")
        
        # 매칭 안 된 GT 확인
        matched_gt_ids = set()
        for match_result in matching.values():
            if match_result:
                matched_gt_ids.add(match_result[0])
        
        unmatched_gts = set(gt_bboxes.keys()) - matched_gt_ids
        if unmatched_gts:
            print(f"\n⚠️  Unmatched GTs: {unmatched_gts}")
            for gt_id in unmatched_gts:
                print(f"  GT-{gt_id}: DAM4SAM failed to track!")
        
        # 기록 저장
        tracking_history['dam_internal_ids'].append(list(range(len(masks))))
        tracking_history['gt_matching'].append(matching)
    
    print("\n" + "=" * 80)
    print("🎉 Test Complete!")
    print("=" * 80)
    
    # ========================================
    # 최종 분석
    # ========================================
    print("\n[3] Final Analysis:")
    print("-" * 80)
    
    analyze_pure_dam4sam_results(gt_to_dam_history, n_frames)
    
    # 결과 저장
    result_path = os.path.join(result_dir, "tracking_analysis.json")
    with open(result_path, 'w') as f:
        # numpy 타입 변환
        history_serializable = {}
        for gt_id, records in gt_to_dam_history.items():
            history_serializable[str(gt_id)] = [
                {
                    'frame': int(r['frame']),
                    'dam_id': int(r['dam_id']),
                    'iou': float(r['iou'])
                }
                for r in records
            ]
        
        json.dump(history_serializable, f, indent=2)
    
    print(f"\n✅ Results saved: {result_path}")


def analyze_pure_dam4sam_results(gt_to_dam_history, n_frames):
    """
    순수 DAM4SAM 결과 분석
    """
    print("\n📊 Pure DAM4SAM Tracking Analysis:")
    print("=" * 60)
    
    for gt_id in [1, 2, 3, 4]:
        history = gt_to_dam_history[gt_id]
        
        if not history:
            continue
        
        print(f"\n{'🔴' if gt_id == 1 else '🟢' if gt_id == 2 else '🔵' if gt_id == 3 else '🟡'} GT Object {gt_id}:")
        
        if gt_id == 1:
            print("  (Speed Change: 5 → 20 → 30 → 10 px/frame)")
        elif gt_id == 2:
            print("  (Constant Speed: 3 px/frame)")
        elif gt_id == 3:
            print("  (Appears at Frame 5)")
        elif gt_id == 4:
            print("  (Appears at Frame 10, diagonal)")
        
        # DAM ID 변화 추적
        dam_ids = [r['dam_id'] for r in history]
        unique_dam_ids = set(dam_ids)
        
        print(f"\n  Total frames tracked: {len(history)}")
        print(f"  DAM internal IDs used: {sorted(unique_dam_ids)}")
        
        if len(unique_dam_ids) > 1:
            print(f"  ⚠️  ID SWITCHING DETECTED!")
            print(f"  Number of different IDs: {len(unique_dam_ids)}")
            
            # ID 전환 지점 찾기
            switches = []
            for i in range(1, len(dam_ids)):
                if dam_ids[i] != dam_ids[i-1]:
                    switches.append({
                        'frame': history[i]['frame'],
                        'from': dam_ids[i-1],
                        'to': dam_ids[i]
                    })
            
            print(f"\n  ID Switches:")
            for sw in switches:
                print(f"    Frame {sw['frame']}: DAM-{sw['from']} → DAM-{sw['to']}")
        
        else:
            print(f"  ✅ Stable tracking - single ID maintained")
        
        # IoU 통계
        ious = [r['iou'] for r in history]
        avg_iou = np.mean(ious)
        min_iou = np.min(ious)
        
        print(f"\n  IoU Statistics:")
        print(f"    Average: {avg_iou:.3f}")
        print(f"    Minimum: {min_iou:.3f}")
        
        # 샘플 출력
        print(f"\n  Sample tracking (first 3, last 3):")
        for r in history[:3]:
            print(f"    Frame {r['frame']:2d}: DAM-{r['dam_id']} (IoU={r['iou']:.3f})")
        
        if len(history) > 6:
            print("    ...")
            for r in history[-3:]:
                print(f"    Frame {r['frame']:2d}: DAM-{r['dam_id']} (IoU={r['iou']:.3f})")
    
    # 전체 요약
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    total_switches = 0
    for gt_id in [1, 2, 3, 4]:
        history = gt_to_dam_history[gt_id]
        if history:
            dam_ids = [r['dam_id'] for r in history]
            if len(set(dam_ids)) > 1:
                total_switches += len(set(dam_ids)) - 1
    
    if total_switches > 0:
        print(f"\n❌ DAM4SAM alone had {total_switches} ID switch(es)")
        print(f"✅ This demonstrates the need for HybridTrack integration!")
        print(f"\n💡 Key Insight:")
        print(f"   - DAM4SAM: Excellent segmentation, but struggles with fast motion")
        print(f"   - HybridTrack: Robust ID management with motion prediction")
        print(f"   - Integration: Combines both strengths!")
    else:
        print(f"\n✅ DAM4SAM maintained stable IDs throughout")
        print(f"⚠️  Consider more challenging scenarios:")
        print(f"   - Higher speeds (50-100 px/frame)")
        print(f"   - Smaller objects (15x15 pixels)")
        print(f"   - Occlusions")


if __name__ == "__main__":
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available. Using CPU (will be slow)")
    
    test_pure_dam4sam()
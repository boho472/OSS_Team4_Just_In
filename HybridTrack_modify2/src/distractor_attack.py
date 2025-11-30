"""
실제 실패 사례 재현: Distractor로 인한 추적 이동
- Target과 유사한 객체가 교차하며 가림
- DAM4SAM: 추적 옮겨감
- HybridTrack: 속도 예측으로 구분
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


def create_distractor_image(frame_idx, width=1024, height=1024):
    """
    Distractor 시나리오 이미지 생성
    
    시나리오:
    - Target (빨강, 30x30): 왼쪽 → 오른쪽 (10px/frame)
    - Distractor (빨강, 35x35): 오른쪽 → 왼쪽 (10px/frame)
    - Frame 12: 교차하며 Target 가림!
    
    - Control (초록, 30x30): 일정 속도 이동
    
    - Target2 (파랑, 25x25): 위쪽 → 아래쪽 (8px/frame)
    - Distractor2 (파랑, 28x28): 아래쪽 → 위쪽 (8px/frame)
    - Frame 10: 교차
    """
    # 배경
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        intensity = int(255 * y / height)
        img[y, :] = [intensity // 3, intensity // 2, intensity]
    
    # ==========================================
    # 시나리오 1: 수평 교차
    # ==========================================
    
    # Target (빨강, 30x30): 왼쪽 → 오른쪽
    x_target = 100 + frame_idx * 10
    y_target = 150
    
    # Distractor (빨강, 35x35): 오른쪽 → 왼쪽
    x_distractor = 600 - frame_idx * 10
    y_distractor = 150
    
    # 교차 판정 (Frame 12 근처)
    # x_target=220, x_distractor=480 → 충돌 X
    # x_target=320, x_distractor=380 → Frame 12에서 교차!
    
    # Z-depth 시뮬레이션: Distractor가 앞에 있음 (나중에 그림)
    
    # Target 먼저 그리기 (뒤)
    cv2.rectangle(img, (x_target, y_target), 
                 (x_target + 30, y_target + 30), (0, 0, 255), -1)
    
    # Distractor 나중에 그리기 (앞) - Target을 가림
    cv2.rectangle(img, (x_distractor, y_distractor), 
                 (x_distractor + 35, y_distractor + 35), (0, 0, 220), -1)  # 약간 어두운 빨강
    
    # ==========================================
    # 시나리오 2: 수직 교차
    # ==========================================
    
    # Target2 (파랑, 25x25): 위 → 아래
    x_target2 = 400
    y_target2 = 100 + frame_idx * 8
    
    # Distractor2 (파랑, 28x28): 아래 → 위
    x_distractor2 = 400
    y_distractor2 = 500 - frame_idx * 8
    
    # Target2 먼저 (뒤)
    cv2.rectangle(img, (x_target2, y_target2),
                 (x_target2 + 25, y_target2 + 25), (255, 0, 0), -1)
    
    # Distractor2 나중에 (앞)
    cv2.rectangle(img, (x_distractor2, y_distractor2),
                 (x_distractor2 + 28, y_distractor2 + 28), (220, 0, 0), -1)  # 약간 어두운 파랑
    
    # ==========================================
    # Control (제어군)
    # ==========================================
    x_control = 700
    y_control = 300 + frame_idx * 5
    cv2.rectangle(img, (x_control, y_control),
                 (x_control + 30, y_control + 30), (0, 255, 0), -1)
    
    return img


def get_distractor_ground_truth(frame_idx):
    """
    Ground Truth (실제 Target 위치)
    
    주의: Distractor는 GT에 포함 안 됨 (추적 대상 아님)
    """
    gt = {}
    
    # Target 1 (빨강)
    x_target = 100 + frame_idx * 10
    gt[1] = {'bbox': [x_target, 150, 30, 30], 'created_frame': 0}
    
    # Target 2 (파랑)
    y_target2 = 100 + frame_idx * 8
    gt[2] = {'bbox': [400, y_target2, 25, 25], 'created_frame': 0}
    
    # Control (초록)
    y_control = 300 + frame_idx * 5
    gt[3] = {'bbox': [700, y_control, 30, 30], 'created_frame': 0}
    
    return gt


def get_distractor_positions(frame_idx):
    """
    Distractor 위치 정보 (분석용)
    """
    distractors = {}
    
    # Distractor 1
    x_d1 = 600 - frame_idx * 10
    distractors['D1'] = {'bbox': [x_d1, 150, 35, 35]}
    
    # Distractor 2
    y_d2 = 500 - frame_idx * 8
    distractors['D2'] = {'bbox': [400, y_d2, 28, 28]}
    
    return distractors


def compute_bbox_from_mask(mask):
    """Mask → bbox"""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return [int(x_min), int(y_min), 
            int(x_max - x_min + 1), int(y_max - y_min + 1)]


def compute_bbox_iou(bbox1, bbox2):
    """IoU"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_dam_to_targets(dam_masks, gt_bboxes, distractors):
    """
    DAM masks를 Target/Distractor와 매칭
    
    Returns:
        {dam_idx: ('target', gt_id, iou) or ('distractor', d_id, iou) or None}
    """
    matching = {}
    
    for dam_idx, mask in enumerate(dam_masks):
        dam_bbox = compute_bbox_from_mask(mask)
        
        if dam_bbox is None:
            matching[dam_idx] = None
            continue
        
        best_iou = 0
        best_match = None
        
        # Target과 비교
        for gt_id, gt_info in gt_bboxes.items():
            gt_bbox = gt_info['bbox']
            iou = compute_bbox_iou(dam_bbox, gt_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_match = ('target', gt_id, iou)
        
        # Distractor와 비교
        for d_id, d_info in distractors.items():
            d_bbox = d_info['bbox']
            iou = compute_bbox_iou(dam_bbox, d_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_match = ('distractor', d_id, iou)
        
        if best_iou >= 0.3:
            matching[dam_idx] = best_match
        else:
            matching[dam_idx] = None
    
    return matching


def test_distractor_attack():
    """Distractor 공격 시나리오 테스트"""
    
    print("=" * 80)
    print("🎯 DISTRACTOR ATTACK TEST")
    print("Scenario: Similar objects crossing paths")
    print("=" * 80)
    print("\n📋 Test Setup:")
    print("  Target 1 (Red, 30x30): Left → Right (10px/frame)")
    print("  Distractor 1 (Dark Red, 35x35): Right → Left (10px/frame)")
    print("  → Cross at Frame 12!")
    print()
    print("  Target 2 (Blue, 25x25): Top → Bottom (8px/frame)")
    print("  Distractor 2 (Dark Blue, 28x28): Bottom → Top (8px/frame)")
    print("  → Cross at Frame 10!")
    print()
    print("  Control (Green, 30x30): Simple downward motion")
    print("=" * 80)
    
    # 디렉토리
    test_dir = "test_distractor_attack"
    image_dir = os.path.join(test_dir, "images")
    result_dir = os.path.join(test_dir, "results")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # DAM4SAM 초기화
    print("\n[1] Initializing DAM4SAM (tracking Targets only)...")
    tracker = DAM4SAMMOT(
        model_size='tiny',
        checkpoint_dir='./checkpoints'
    )
    
    n_frames = 25
    
    # 추적 기록
    target_tracking = {1: [], 2: [], 3: []}  # GT ID → tracking history
    distractor_contamination = {1: [], 2: []}  # 언제 Distractor 추적했나
    
    for frame_idx in range(n_frames):
        print(f"\n{'='*60}")
        print(f"🚩 Frame {frame_idx}")
        print(f"{'='*60}")
        
        # 이미지 생성
        img_array = create_distractor_image(frame_idx)
        image = Image.fromarray(img_array)
        
        img_path = os.path.join(image_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # Ground Truth & Distractors
        gt_bboxes = get_distractor_ground_truth(frame_idx)
        distractors = get_distractor_positions(frame_idx)
        
        print(f"\nTargets (GT): {len(gt_bboxes)}")
        for gt_id, gt_info in gt_bboxes.items():
            print(f"  Target-{gt_id}: {gt_info['bbox']}")
        
        print(f"\nDistractors (not tracked): {len(distractors)}")
        for d_id, d_info in distractors.items():
            print(f"  {d_id}: {d_info['bbox']}")
        
        # DAM4SAM 실행
        if frame_idx == 0:
            print("\n🔧 Initializing with Targets only...")
            init_regions = []
            for gt_id in sorted(gt_bboxes.keys()):
                bbox = gt_bboxes[gt_id]['bbox']
                init_regions.append({'bbox': bbox})
            
            tracker.initialize(image, init_regions)
            outputs = tracker.track(image)
        else:
            print("\n🤖 Pure DAM4SAM tracking...")
            outputs = tracker.track(image)
        
        # 결과 분석
        masks = outputs['masks']
        print(f"\n📊 DAM4SAM Output: {len(masks)} masks")
        
        # 매칭
        matching = match_dam_to_targets(masks, gt_bboxes, distractors)
        
        print(f"\n🔗 Matching:")
        for dam_idx, match in matching.items():
            if match:
                match_type, obj_id, iou = match
                
                if match_type == 'target':
                    print(f"  DAM-{dam_idx} ← ✅ Target-{obj_id} (IoU={iou:.3f})")
                    
                    target_tracking[obj_id].append({
                        'frame': frame_idx,
                        'dam_id': dam_idx,
                        'iou': iou,
                        'status': 'correct'
                    })
                
                elif match_type == 'distractor':
                    print(f"  DAM-{dam_idx} ← ❌ DISTRACTOR {obj_id}! (IoU={iou:.3f})")
                    
                    # 어느 Target의 추적이 옮겨갔나?
                    # 이전 프레임에서 이 dam_idx가 어느 Target을 추적했나 확인
                    prev_target = None
                    for t_id, history in target_tracking.items():
                        if history and history[-1]['dam_id'] == dam_idx:
                            prev_target = t_id
                            break
                    
                    if prev_target:
                        print(f"      → Target-{prev_target} tracking LOST!")
                        distractor_contamination[prev_target].append({
                            'frame': frame_idx,
                            'distractor': obj_id
                        })
            else:
                print(f"  DAM-{dam_idx} ← ? Unknown")
    
    print("\n" + "=" * 80)
    print("🎉 Test Complete!")
    print("=" * 80)
    
    # 분석
    print("\n[3] Distractor Attack Analysis:")
    print("-" * 80)
    
    analyze_distractor_results(target_tracking, distractor_contamination, n_frames)


def analyze_distractor_results(target_tracking, distractor_contamination, n_frames):
    """Distractor 공격 결과 분석"""
    
    print("\n📊 DISTRACTOR ATTACK RESULTS:")
    print("=" * 60)
    
    # Target 1 분석
    print("\n🔴 Target 1 - Horizontal Crossing Attack")
    print("  Expected: Distractor crosses at Frame 12")
    
    history = target_tracking[1]
    contamination = distractor_contamination.get(1, [])
    
    if contamination:
        print(f"\n  ❌ TRACKING CONTAMINATED!")
        for event in contamination:
            print(f"     Frame {event['frame']}: Switched to {event['distractor']}")
        
        # 복구 확인
        last_contam_frame = contamination[-1]['frame']
        recovered = any(r['frame'] > last_contam_frame for r in history if r.get('status') == 'correct')
        
        if recovered:
            print(f"  ⚠️  Tracking recovered after contamination")
        else:
            print(f"  ❌ Tracking permanently lost to distractor")
    else:
        print(f"  ✅ Survived distractor attack!")
    
    # Target 2 분석
    print("\n🔵 Target 2 - Vertical Crossing Attack")
    print("  Expected: Distractor crosses at Frame 10")
    
    contamination2 = distractor_contamination.get(2, [])
    
    if contamination2:
        print(f"\n  ❌ TRACKING CONTAMINATED!")
        for event in contamination2:
            print(f"     Frame {event['frame']}: Switched to {event['distractor']}")
    else:
        print(f"  ✅ Survived distractor attack!")
    
    # Control
    print("\n🟢 Control - No Distractors")
    control_history = target_tracking[3]
    dam_ids = [r['dam_id'] for r in control_history]
    
    if len(set(dam_ids)) == 1:
        print(f"  ✅ Perfect tracking - baseline maintained")
    else:
        print(f"  ⚠️  Unexpected ID switch in control!")
    
    # 최종 판정
    print("\n" + "=" * 60)
    print("🎯 FINAL VERDICT:")
    print("=" * 60)
    
    total_attacks = len(distractor_contamination[1]) + len(distractor_contamination.get(2, []))
    
    if total_attacks > 0:
        print(f"\n❌ DAM4SAM VULNERABLE TO DISTRACTORS!")
        print(f"   Total contamination events: {total_attacks}")
        print(f"\n💡 ROOT CAUSE:")
        print(f"   - No depth (Z) awareness → Can't distinguish overlapping objects")
        print(f"   - No velocity prediction → Can't distinguish motion patterns")
        print(f"   - Memory-based matching → Chooses visually similar object")
        print(f"\n✅ HYBRIDTRACK SOLUTION:")
        print(f"   - Motion prediction: Knows where Target SHOULD be")
        print(f"   - Bbox provides spatial prior: Disambiguates overlap")
        print(f"   - Kalman filter: Maintains velocity estimates")
        print(f"\n🎓 RESEARCH CONTRIBUTION:")
        print(f"   This demonstrates the CRITICAL need for:")
        print(f"   1. Motion-based tracking (not just appearance)")
        print(f"   2. Spatial priors from detection")
        print(f"   3. Integration of complementary methods")
    else:
        print(f"\n⚠️  DAM4SAM survived all distractor attacks!")
        print(f"   Consider:")
        print(f"   - More similar distractors (identical size/color)")
        print(f"   - Longer occlusion periods")
        print(f"   - Multiple simultaneous distractors")


if __name__ == "__main__":
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available")
    
    test_distractor_attack()
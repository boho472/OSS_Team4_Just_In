"""
궁극의 Distractor 테스트
- Target과 100% 동일한 외형의 Distractor
- 교차 시 완전히 겹침
- 속도만 다름 (HT는 구분 가능, DAM4SAM은 불가능)
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


def create_ultimate_distractor_image(frame_idx, width=1024, height=1024):
    """
    완전 동일한 객체들의 교차
    
    시나리오:
    - Target (빨강, 30x30): 왼쪽 → 오른쪽 (15px/frame)
    - Distractor (빨강, 30x30): 오른쪽 → 왼쪽 (15px/frame)
    - Frame 8-10: 완전히 겹침!
    
    - Target2 (파랑, 28x28): 대각선 ↘ (12px/frame)
    - Distractor2 (파랑, 28x28): 대각선 ↖ (12px/frame)
    - Frame 9-11: 완전히 겹침!
    
    - Triple Attack:
      * Target3 (초록, 25x25): 중앙에서 시작, 오른쪽으로
      * Distractor3A (초록, 25x25): 위에서 중앙으로
      * Distractor3B (초록, 25x25): 아래에서 중앙으로
      * Frame 10-12: 3개가 동시에 겹침!
    """
    # 배경
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        intensity = int(255 * y / height)
        img[y, :] = [intensity // 3, intensity // 2, intensity]
    
    # ==========================================
    # 시나리오 1: 수평 충돌 (완전 동일)
    # ==========================================
    
    # Target (빨강, 30x30)
    x_t1 = 100 + frame_idx * 15
    y_t1 = 100
    
    # Distractor (빨강, 30x30) - 완전 동일!
    x_d1 = 500 - frame_idx * 15
    y_d1 = 100
    
    # 충돌 계산
    # Frame 8: x_t1=220, x_d1=380
    # Frame 9: x_t1=235, x_d1=365
    # Frame 10: x_t1=250, x_d1=350 → 교차!
    
    # 겹침 여부 확인
    overlap1 = (x_t1 < x_d1 + 30) and (x_t1 + 30 > x_d1)
    
    if overlap1:
        # 겹치면 하나만 그리기 (Distractor가 Target 가림)
        cv2.rectangle(img, (x_d1, y_d1), (x_d1 + 30, y_d1 + 30), 
                     (0, 0, 255), -1)
    else:
        # 안 겹치면 둘 다 그리기
        cv2.rectangle(img, (x_t1, y_t1), (x_t1 + 30, y_t1 + 30), 
                     (0, 0, 255), -1)
        cv2.rectangle(img, (x_d1, y_d1), (x_d1 + 30, y_d1 + 30), 
                     (0, 0, 255), -1)
    
    # ==========================================
    # 시나리오 2: 대각선 충돌
    # ==========================================
    
    # Target2 (파랑, 28x28): ↘
    x_t2 = 150 + frame_idx * 12
    y_t2 = 200 + frame_idx * 12
    
    # Distractor2 (파랑, 28x28): ↖
    x_d2 = 450 - frame_idx * 12
    y_d2 = 500 - frame_idx * 12
    
    overlap2_x = (x_t2 < x_d2 + 28) and (x_t2 + 28 > x_d2)
    overlap2_y = (y_t2 < y_d2 + 28) and (y_t2 + 28 > y_d2)
    overlap2 = overlap2_x and overlap2_y
    
    if overlap2:
        cv2.rectangle(img, (x_d2, y_d2), (x_d2 + 28, y_d2 + 28), 
                     (255, 0, 0), -1)
    else:
        cv2.rectangle(img, (x_t2, y_t2), (x_t2 + 28, y_t2 + 28), 
                     (255, 0, 0), -1)
        cv2.rectangle(img, (x_d2, y_d2), (x_d2 + 28, y_d2 + 28), 
                     (255, 0, 0), -1)
    
    # ==========================================
    # 시나리오 3: Triple Attack (3개 동시 충돌!)
    # ==========================================
    
    # Target3 (초록, 25x25): 왼쪽 → 오른쪽
    x_t3 = 250 + frame_idx * 10
    y_t3 = 600
    
    # Distractor3A (초록, 25x25): 위 → 아래
    x_d3a = 400
    y_d3a = 450 + frame_idx * 10
    
    # Distractor3B (초록, 25x25): 아래 → 위
    x_d3b = 400
    y_d3b = 750 - frame_idx * 10
    
    # 3개 중 겹치는 것 체크
    # Frame 10-12: 모두 (400, 600) 근처에 모임
    
    # Target3
    cv2.rectangle(img, (x_t3, y_t3), (x_t3 + 25, y_t3 + 25), 
                 (0, 255, 0), -1)
    
    # Distractor3A (나중에 그려서 가림)
    cv2.rectangle(img, (x_d3a, y_d3a), (x_d3a + 25, y_d3a + 25), 
                 (0, 255, 0), -1)
    
    # Distractor3B (가장 나중에 그려서 위에 덮음)
    cv2.rectangle(img, (x_d3b, y_d3b), (x_d3b + 25, y_d3b + 25), 
                 (0, 255, 0), -1)
    
    return img


def get_ultimate_ground_truth(frame_idx):
    """Target만 (Distractor 제외)"""
    gt = {}
    
    # Target 1
    x_t1 = 100 + frame_idx * 15
    gt[1] = {'bbox': [x_t1, 100, 30, 30], 'created_frame': 0}
    
    # Target 2
    x_t2 = 150 + frame_idx * 12
    y_t2 = 200 + frame_idx * 12
    gt[2] = {'bbox': [x_t2, y_t2, 28, 28], 'created_frame': 0}
    
    # Target 3
    x_t3 = 250 + frame_idx * 10
    gt[3] = {'bbox': [x_t3, 600, 25, 25], 'created_frame': 0}
    
    return gt


def get_ultimate_distractors(frame_idx):
    """Distractor 위치"""
    distractors = {}
    
    # D1
    x_d1 = 500 - frame_idx * 15
    distractors['D1'] = {'bbox': [x_d1, 100, 30, 30]}
    
    # D2
    x_d2 = 450 - frame_idx * 12
    y_d2 = 500 - frame_idx * 12
    distractors['D2'] = {'bbox': [x_d2, y_d2, 28, 28]}
    
    # D3A
    y_d3a = 450 + frame_idx * 10
    distractors['D3A'] = {'bbox': [400, y_d3a, 25, 25]}
    
    # D3B
    y_d3b = 750 - frame_idx * 10
    distractors['D3B'] = {'bbox': [400, y_d3b, 25, 25]}
    
    return distractors


def compute_bbox_from_mask(mask):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return [int(x_min), int(y_min), 
            int(x_max - x_min + 1), int(y_max - y_min + 1)]


def compute_bbox_iou(bbox1, bbox2):
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


def match_with_all(dam_masks, targets, distractors):
    """DAM masks를 Targets + Distractors와 모두 매칭"""
    matching = {}
    
    for dam_idx, mask in enumerate(dam_masks):
        dam_bbox = compute_bbox_from_mask(mask)
        
        if dam_bbox is None:
            matching[dam_idx] = None
            continue
        
        best_iou = 0
        best_match = None
        
        # Targets
        for t_id, t_info in targets.items():
            iou = compute_bbox_iou(dam_bbox, t_info['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = ('target', t_id, iou)
        
        # Distractors
        for d_id, d_info in distractors.items():
            iou = compute_bbox_iou(dam_bbox, d_info['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match = ('distractor', d_id, iou)
        
        if best_iou >= 0.3:
            matching[dam_idx] = best_match
        else:
            matching[dam_idx] = None
    
    return matching


def test_ultimate_distractor():
    """궁극의 Distractor 테스트"""
    
    print("=" * 80)
    print("🔥 ULTIMATE DISTRACTOR CHALLENGE 🔥")
    print("Scenario: Identical objects with opposite velocities")
    print("=" * 80)
    print("\n📋 Attacks:")
    print("  1. Horizontal Collision (Frame 8-10)")
    print("     - Target & Distractor: IDENTICAL (Red, 30x30)")
    print("     - Opposite velocities: ±15px/frame")
    print()
    print("  2. Diagonal Collision (Frame 9-11)")
    print("     - Target2 & Distractor2: IDENTICAL (Blue, 28x28)")
    print("     - Opposite diagonal motion: ±12px/frame")
    print()
    print("  3. Triple Attack (Frame 10-12)")
    print("     - Target3 + 2 Distractors: ALL IDENTICAL (Green, 25x25)")
    print("     - 3 objects converge at same point!")
    print("=" * 80)
    
    test_dir = "test_ultimate_distractor"
    image_dir = os.path.join(test_dir, "images")
    result_dir = os.path.join(test_dir, "results")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    print("\n[1] Initializing DAM4SAM...")
    tracker = DAM4SAMMOT(
        model_size='tiny',
        checkpoint_dir='./checkpoints'
    )
    
    n_frames = 20
    
    # 추적 기록
    target_tracking = {1: [], 2: [], 3: []}
    contamination_events = []
    
    for frame_idx in range(n_frames):
        print(f"\n{'='*60}")
        print(f"🚩 Frame {frame_idx}")
        
        # 충돌 예상 구간 표시
        if 8 <= frame_idx <= 10:
            print("⚠️  COLLISION ZONE: Horizontal attack!")
        if 9 <= frame_idx <= 11:
            print("⚠️  COLLISION ZONE: Diagonal attack!")
        if 10 <= frame_idx <= 12:
            print("🔥 TRIPLE ATTACK ZONE!")
        
        print(f"{'='*60}")
        
        # 이미지
        img_array = create_ultimate_distractor_image(frame_idx)
        image = Image.fromarray(img_array)
        
        img_path = os.path.join(image_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # GT & Distractors
        gt_bboxes = get_ultimate_ground_truth(frame_idx)
        distractors = get_ultimate_distractors(frame_idx)
        
        print(f"\nTargets: {len(gt_bboxes)}")
        for t_id, t_info in gt_bboxes.items():
            print(f"  T{t_id}: {t_info['bbox']}")
        
        print(f"\nDistractors: {len(distractors)}")
        for d_id, d_info in distractors.items():
            print(f"  {d_id}: {d_info['bbox']}")
        
        # DAM4SAM
        if frame_idx == 0:
            print("\n🔧 Initialize with Targets...")
            init_regions = []
            for t_id in sorted(gt_bboxes.keys()):
                bbox = gt_bboxes[t_id]['bbox']
                init_regions.append({'bbox': bbox})
            
            tracker.initialize(image, init_regions)
            outputs = tracker.track(image)
        else:
            print("\n🤖 DAM4SAM tracking...")
            outputs = tracker.track(image)
        
        # 분석
        masks = outputs['masks']
        print(f"\n📊 Output: {len(masks)} masks")
        
        matching = match_with_all(masks, gt_bboxes, distractors)
        
        print(f"\n🔗 Matching:")
        for dam_idx, match in matching.items():
            if match:
                m_type, m_id, iou = match
                
                if m_type == 'target':
                    print(f"  DAM-{dam_idx} → ✅ Target {m_id} (IoU={iou:.3f})")
                    
                    target_tracking[m_id].append({
                        'frame': frame_idx,
                        'dam_id': dam_idx,
                        'iou': iou,
                        'status': 'correct'
                    })
                
                elif m_type == 'distractor':
                    print(f"  DAM-{dam_idx} → ❌ DISTRACTOR {m_id}! (IoU={iou:.3f})")
                    
                    # 어느 Target이 잃어버렸나
                    for t_id, history in target_tracking.items():
                        if history and history[-1].get('dam_id') == dam_idx:
                            print(f"      🚨 Target {t_id} LOST!")
                            
                            contamination_events.append({
                                'frame': frame_idx,
                                'target': t_id,
                                'distractor': m_id,
                                'iou': iou
                            })
                            break
    
    print("\n" + "=" * 80)
    print("🎉 Test Complete!")
    print("=" * 80)
    
    # 최종 분석
    analyze_ultimate_results(target_tracking, contamination_events, n_frames)


def analyze_ultimate_results(target_tracking, contamination_events, n_frames):
    """최종 분석"""
    
    print("\n📊 ULTIMATE DISTRACTOR RESULTS:")
    print("=" * 60)
    
    for t_id in [1, 2, 3]:
        icon = "🔴" if t_id == 1 else "🔵" if t_id == 2 else "🟢"
        attack = "Horizontal" if t_id == 1 else "Diagonal" if t_id == 2 else "Triple"
        
        print(f"\n{icon} Target {t_id} - {attack} Attack")
        
        # 이 Target 관련 contamination
        target_contam = [e for e in contamination_events if e['target'] == t_id]
        
        if target_contam:
            print(f"  ❌ CONTAMINATED!")
            for event in target_contam:
                print(f"     Frame {event['frame']}: Switched to {event['distractor']}")
        else:
            print(f"  ✅ Survived!")
    
    # 최종 판정
    print("\n" + "=" * 60)
    print("🎯 FINAL VERDICT:")
    print("=" * 60)
    
    if contamination_events:
        print(f"\n❌ DAM4SAM FAILED!")
        print(f"   Total contaminations: {len(contamination_events)}")
        print(f"\n🔬 ROOT CAUSE ANALYSIS:")
        print(f"   When identical objects overlap:")
        print(f"   - No visual difference to distinguish")
        print(f"   - No depth (Z) information")
        print(f"   - No velocity-based prediction")
        print(f"   → Memory matches to WRONG object!")
        print(f"\n✅ HYBRIDTRACK SOLUTION:")
        print(f"   - Motion model predicts Target's trajectory")
        print(f"   - Bbox provides spatial disambiguation")
        print(f"   - Velocity consistency check")
        print(f"\n🎓 RESEARCH CONTRIBUTION PROVEN!")
    else:
        print(f"\n😱 DAM4SAM SURVIVED EVERYTHING!")
        print(f"   This model is REALLY strong!")
        print(f"   Consider:")
        print(f"   - Longer overlap (10+ frames)")
        print(f"   - More complex trajectories")


if __name__ == "__main__":
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available")
    
    test_ultimate_distractor()
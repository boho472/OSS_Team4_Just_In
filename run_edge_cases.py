import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from tracking_wrapper_mot import DAM4SAMMOT


def create_edge_case_images(output_dir, n_frames=20):
    """엣지 케이스용 더미 이미지 생성"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for frame_idx in range(n_frames):
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        
        # 배경
        for y in range(1024):
            img[y, :, :] = int(255 * y / 1024)
        
        # 객체들 그리기 (간단한 사각형)
        if frame_idx >= 0:
            # 객체 1, 2
            x1 = 100 + frame_idx
            x2 = 200 + frame_idx
            img[100:150, x1:x1+50] = [255, 0, 0]
            img[200:250, x2:x2+50] = [0, 255, 0]
        
        if frame_idx >= 10:
            # 객체 3, 4, 5
            x3 = 300 + frame_idx - 10
            x4 = 400 + frame_idx - 10
            x5 = 500 + frame_idx - 10
            img[300:360, x3:x3+60] = [0, 0, 255]
            img[400:460, x4:x4+60] = [255, 255, 0]
            img[500:560, x5:x5+60] = [255, 0, 255]
        
        pil_img = Image.fromarray(img)
        pil_img.save(output_dir / f"frame_{frame_idx:06d}.jpg")
    
    print(f"✅ Created {n_frames} edge case images")


def run_edge_case_test():
    print("="*80)
    print("DAM4SAM Edge Case Test: Multiple Objects Per Frame")
    print("="*80)
    
    json_dir = Path("test_jsons_edge")
    image_dir = Path("test_images_edge")
    
    # 이미지 생성
    create_edge_case_images(image_dir, n_frames=20)
    
    # DAM4SAM 초기화
    print("\n[1] Initializing DAM4SAM...")
    tracker = DAM4SAMMOT(model_size='tiny', checkpoint_dir='./checkpoints')
    print("✅ Initialized")
    
    # 주요 프레임만 상세 출력
    key_frames = [0, 10, 15]
    
    for frame_idx in range(20):
        json_path = json_dir / f"frame_{frame_idx:06d}.json"
        img_path = image_dir / f"frame_{frame_idx:06d}.jpg"
        image = Image.open(img_path)
        
        if frame_idx in key_frames:
            print(f"\n{'='*80}")
            print(f"🚩 Frame {frame_idx:03d} (Key Frame)")
            print(f"{'='*80}")
        else:
            print(f"\n📍 Frame {frame_idx:03d}")
        
        # JSON 읽기
        with open(json_path, 'r') as f:
            frame_data = json.load(f)
        
        ht_results = frame_data['dam4sam_tracking']['HybridTrack_results']
        print(f"   HT detected {len(ht_results)} objects:")
        for ht_obj in ht_results:
            print(f"      - obj_id={ht_obj['object_id']}, bbox={ht_obj['bbox']}")
        
        # DAM4SAM 처리
        outputs = tracker.process_frame_with_ht_json(frame_idx, json_path, image)
        
        # 결과
        print(f"   ✅ DAM4SAM tracking {len(outputs['masks'])} objects")
        
        # 상세 결과
        if frame_idx in key_frames:
            with open(json_path, 'r') as f:
                updated = json.load(f)
            
            dam_results = updated['dam4sam_tracking']['DAM4SAM_results']
            print(f"\n   📊 DAM4SAM Results:")
            for dam_obj in dam_results:
                print(f"      - internal_id={dam_obj['internal_id']}, "
                      f"pixels={dam_obj['mask_pixels']}, bbox={dam_obj['bbox']}")
    
    # 최종 분석
    print("\n" + "="*80)
    print("📊 Final Analysis")
    print("="*80)
    
    for frame_idx in [0, 10, 15]:
        json_path = json_dir / f"frame_{frame_idx:06d}.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        ht_count = len(data['dam4sam_tracking']['HybridTrack_results'])
        dam_count = len(data['dam4sam_tracking']['DAM4SAM_results'])
        
        print(f"\nFrame {frame_idx}:")
        print(f"  HT:  {ht_count} objects")
        print(f"  DAM: {dam_count} objects")
        
        if frame_idx == 0:
            print(f"  Expected: 2 objects initialized ✓" if dam_count == 2 else f"  ❌ Expected 2, got {dam_count}")
        elif frame_idx == 10:
            print(f"  Expected: 5 objects (2 old + 3 new) ✓" if dam_count == 5 else f"  ❌ Expected 5, got {dam_count}")
        elif frame_idx == 15:
            print(f"  Expected: 5 objects (ID switch filtered) ✓" if dam_count == 5 else f"  ❌ Expected 5, got {dam_count}")


if __name__ == "__main__":
    run_edge_case_test()
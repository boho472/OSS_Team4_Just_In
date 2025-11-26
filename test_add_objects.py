import os
import torch
from PIL import Image
import numpy as np
import gc

from tracking_wrapper_mot import DAM4SAMMOT


def create_dummy_image(width=640, height=480):
    """Create a dummy RGB image"""
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_add_objects_no_dataset():
    """
    Test adding objects without VOT dataset.
    Uses dummy images with hardcoded bboxes.
    
    Scenario:
    - Frame 0: Initialize with 2 objects
    - Frames 1-9: Track 2 objects
    - Frame 10: Add 1 new object
    - Frames 11-20: Track all 3 objects
    """
    
    # 환경 변수 설정
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    checkpoint_dir = './checkpoints'
    
    print(f"\n{'='*80}")
    print(f"Testing DAM4SAM: Add New Objects During Tracking")
    print(f"{'='*80}\n")
    
    # 초기 메모리 정리
    clear_memory()
    
    # Initialize tracker with memory optimization
    print("Initializing tracker with memory optimization...")
    tracker = DAM4SAMMOT(
        model_size='base',  # large → base로 변경 (메모리 절약)
        checkpoint_dir=checkpoint_dir,
        offload_state_to_cpu=True  # CPU 오프로드 활성화
    )
    print("✓ Tracker initialized\n")
    
    clear_memory()
    
    # ===== Frame 0: Initialize with 2 objects =====
    print("=" * 80)
    print("FRAME 0: Initializing with 2 objects")
    print("=" * 80)
    
    image_0 = create_dummy_image()
    
    # 초기 객체 2개 (하드코딩된 bbox)
    init_regions = [
        {'bbox': [100, 100, 80, 80]},   # Object 1
        {'bbox': [300, 200, 100, 100]}  # Object 2
    ]
    
    print(f"Object 1 bbox: {init_regions[0]['bbox']}")
    print(f"Object 2 bbox: {init_regions[1]['bbox']}")
    
    outputs = tracker.initialize(image_0, init_regions)
    print(f"\n✓ Initialized with {len(init_regions)} objects")
    print(f"  Tracking IDs: {tracker.all_obj_ids}")
    print(f"  Total objects: {len(tracker.all_obj_ids)}\n")
    
    # 메모리 정리
    del image_0, outputs
    clear_memory()
    
    # ===== Frames 1-9: Track initial objects =====
    print("=" * 80)
    print("FRAMES 1-9: Tracking initial objects")
    print("=" * 80)
    
    for frame_idx in range(1, 10):
        # 매 프레임 전에 메모리 정리
        clear_memory()
        
        image = create_dummy_image()
        outputs = tracker.track(image)
        pred_masks = outputs['masks']
        
        print(f"Frame {frame_idx}: Tracked {len(pred_masks)} objects | IDs: {tracker.all_obj_ids}")
        
        # 즉시 메모리 해제
        del image, outputs, pred_masks
    
    print()
    clear_memory()
    
    # ===== Frame 10: Track + Add new object =====
    print("=" * 80)
    print("FRAME 10: Adding new object")
    print("=" * 80)
    
    clear_memory()
    image_10 = create_dummy_image()
    
    # 먼저 기존 객체 추적
    print("Step 1: Tracking existing objects...")
    outputs = tracker.track(image_10)
    print(f"  Tracked {len(outputs['masks'])} objects | IDs: {tracker.all_obj_ids}")
    
    # 메모리 정리
    del outputs
    clear_memory()
    
    # 새 객체 추가
    print("\nStep 2: Adding new object...")
    new_regions = [
        {'bbox': [450, 300, 90, 90]}  # New Object 3
    ]
    print(f"  New object bbox: {new_regions[0]['bbox']}")
    
    new_ids = tracker.add_new_objects(
        frame_idx=10,
        image=image_10,
        regions=new_regions
    )
    
    print(f"\n✓ Added {len(new_ids)} new object(s)")
    print(f"  New IDs: {new_ids}")
    print(f"  Total objects now: {len(tracker.all_obj_ids)} | IDs: {tracker.all_obj_ids}\n")
    
    # 메모리 정리
    del image_10
    clear_memory()
    
    # ===== Frames 11-20: Track all objects =====
    print("=" * 80)
    print("FRAMES 11-20: Tracking all objects (including new one)")
    print("=" * 80)
    
    for frame_idx in range(11, 21):
        # 매 프레임 전에 메모리 정리
        clear_memory()
        
        image = create_dummy_image()
        outputs = tracker.track(image)
        pred_masks = outputs['masks']
        
        print(f"Frame {frame_idx}: Tracked {len(pred_masks)} objects | IDs: {tracker.all_obj_ids}", end="")
        
        # 각 객체의 마스크 픽셀 수 표시
        mask_sizes = [mask.sum() for mask in pred_masks]
        print(f" | Mask sizes: {mask_sizes}")
        
        # 즉시 메모리 해제
        del image, outputs, pred_masks
    
    # ===== Summary =====
    print("\n" + "=" * 80)
    print("TEST COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"✓ Initial objects: 2 (IDs: [1, 2])")
    print(f"✓ Added at frame 10: 1 (ID: 3)")
    print(f"✓ Final tracking objects: {len(tracker.all_obj_ids)} (IDs: {tracker.all_obj_ids})")
    print(f"✓ Tracked until frame: 20")
    print("=" * 80)
    
    # 검증
    assert len(tracker.all_obj_ids) == 3, f"Expected 3 objects, got {len(tracker.all_obj_ids)}"
    assert tracker.all_obj_ids == [1, 2, 3], f"Expected IDs [1, 2, 3], got {tracker.all_obj_ids}"
    
    print("\n✅ ALL TESTS PASSED!\n")
    
    # 최종 메모리 정리
    del tracker
    clear_memory()


if __name__ == "__main__":
    try:
        test_add_objects_no_dataset()
    except Exception as e:
        print(f"\n❌ TEST FAILED with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
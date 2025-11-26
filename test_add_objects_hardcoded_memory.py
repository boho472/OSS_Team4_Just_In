# test_add_objects_hardcoded_with_memory_log.py
import os
import torch
from PIL import Image
import numpy as np
import gc

from tracking_wrapper_mot import DAM4SAMMOT


def get_gpu_memory():
    """현재 GPU 메모리 사용량 반환 (MB)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(0) / 1024**2    # MB
        return allocated, reserved
    return 0, 0


def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_add_objects_hardcoded(video_dir, checkpoint_dir='./checkpoints'):
    """
    Test with hardcoded bboxes + memory logging
    """
    
    # ===== 🔧 CONFIGURE BBOXES HERE =====
    INIT_BBOXES = [
        [520, 85, 220, 640],  # Object 1
        [725, 140, 175, 580],  # Object 2
    ]
    
    NEW_BBOXES = [
        [0, 150, 50, 300],  # Object 3
    ]
    # ===================================
    
    # 메모리 로그 저장
    memory_log = []
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print(f"\n{'='*80}")
    print(f"Testing DAM4SAM: Add New Objects (With Memory Logging)")
    print(f"{'='*80}\n")
    
    # Initial memory
    alloc, res = get_gpu_memory()
    memory_log.append(('Start', 0, alloc, res))
    print(f"Initial GPU Memory: Allocated={alloc:.2f}MB, Reserved={res:.2f}MB\n")
    
    # Load frames
    print("Loading frames...")
    frames = []
    frame_paths = []
    
    for i in range(20):
        patterns = [
            f"{i:06d}.jpg",
            f"{i:06d}.png",
            f"frame_{i:04d}.jpg",
            f"frame_{i:04d}.png",
        ]
        
        frame_path = None
        for pattern in patterns:
            test_path = os.path.join(video_dir, pattern)
            if os.path.exists(test_path):
                frame_path = test_path
                break
        
        if frame_path is None:
            raise FileNotFoundError(f"Frame {i} not found in {video_dir}")
        
        frames.append(Image.open(frame_path).convert("RGB"))
        frame_paths.append(frame_path)
    
    print(f"✓ Loaded {len(frames)} frames (resolution: {frames[0].size})\n")
    
    # After loading frames
    alloc, res = get_gpu_memory()
    memory_log.append(('After Loading Frames', 0, alloc, res))
    
    # Initialize tracker
    print("Initializing tracker...")
    tracker = DAM4SAMMOT(
        model_size='small',
        checkpoint_dir=checkpoint_dir,
        offload_state_to_cpu=True
    )
    print("✓ Tracker initialized\n")
    
    # After tracker init
    alloc, res = get_gpu_memory()
    memory_log.append(('After Tracker Init', 0, alloc, res))
    print(f"After Tracker Init: Allocated={alloc:.2f}MB, Reserved={res:.2f}MB\n")
    
    # ===== Frame 0: Initialize =====
    print("=" * 80)
    print(f"FRAME 0: Initializing with {len(INIT_BBOXES)} objects")
    print(f"  BBoxes: {INIT_BBOXES}")
    print("=" * 80)
    
    init_regions = [{'bbox': bbox} for bbox in INIT_BBOXES]
    tracker.initialize(frames[0], init_regions)
    
    alloc, res = get_gpu_memory()
    memory_log.append(('Initialize', 0, alloc, res))
    print(f"✓ Initialized with {len(init_regions)} objects")
    print(f"  Tracking IDs: {tracker.all_obj_ids}")
    print(f"  Memory: Allocated={alloc:.2f}MB, Reserved={res:.2f}MB\n")
    
    # ===== Frames 1-7: Track =====
    print("=" * 80)
    print("FRAMES 1-7: Tracking initial objects")
    print("=" * 80)
    
    for frame_idx in range(1, 8):
        outputs = tracker.track(frames[frame_idx])
        mask_sizes = [int(mask.sum()) for mask in outputs['masks']]
        
        alloc, res = get_gpu_memory()
        memory_log.append(('Track', frame_idx, alloc, res))
        
        print(f"Frame {frame_idx}: {len(outputs['masks'])} objects | "
              f"Mask sizes: {mask_sizes} | "
              f"Memory: {alloc:.2f}MB (Δ{alloc - memory_log[-2][2]:.2f}MB)")
    
    print()
    
    # ===== Frame 8: Add new object =====
    print("=" * 80)
    print(f"FRAME 8: Adding {len(NEW_BBOXES)} new object(s)")
    print(f"  BBoxes: {NEW_BBOXES}")
    print("=" * 80)
    
    # Track first
    print("\nStep 1: Tracking existing objects...")
    outputs = tracker.track(frames[8])
    mask_sizes = [int(mask.sum()) for mask in outputs['masks']]
    
    alloc, res = get_gpu_memory()
    memory_log.append(('Track', 8, alloc, res))
    print(f"  Tracked {len(outputs['masks'])} objects | Mask sizes: {mask_sizes}")
    print(f"  Memory: {alloc:.2f}MB\n")
    
    # Add new object
    print("Step 2: Adding new object...")
    new_regions = [{'bbox': bbox} for bbox in NEW_BBOXES]
    
    new_ids = tracker.add_new_objects(
        frame_idx=8,
        image=frames[8],
        regions=new_regions
    )
    
    alloc, res = get_gpu_memory()
    memory_log.append(('Add Objects', 8, alloc, res))
    
    print(f"\n✓ Added {len(new_ids)} new object(s)")
    print(f"  New IDs: {new_ids}")
    print(f"  Total objects: {len(tracker.all_obj_ids)} | IDs: {tracker.all_obj_ids}")
    print(f"  Memory: {alloc:.2f}MB (Δ{alloc - memory_log[-2][2]:.2f}MB)\n")
    
    # ===== Frames 9-19: Track all =====
    print("=" * 80)
    print("FRAMES 9-19: Tracking all objects")
    print("=" * 80)
    
    for frame_idx in range(9, 20):
        outputs = tracker.track(frames[frame_idx])
        mask_sizes = [int(mask.sum()) for mask in outputs['masks']]
        
        alloc, res = get_gpu_memory()
        memory_log.append(('Track', frame_idx, alloc, res))
        
        print(f"Frame {frame_idx}: {len(outputs['masks'])} objects | "
              f"Mask sizes: {mask_sizes} | "
              f"Memory: {alloc:.2f}MB (Δ{alloc - memory_log[-2][2]:.2f}MB)")
    
    # ===== Summary =====
    print("\n" + "=" * 80)
    print("TEST COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"✓ Initial objects at frame 0: {len(INIT_BBOXES)}")
    print(f"✓ New objects added at frame 8: {len(NEW_BBOXES)}")
    print(f"✓ Final tracking objects: {len(tracker.all_obj_ids)}")
    print(f"✓ Object IDs: {tracker.all_obj_ids}")
    print("=" * 80)
    
    # Validation
    expected_count = len(INIT_BBOXES) + len(NEW_BBOXES)
    if len(tracker.all_obj_ids) == expected_count:
        print(f"\n✅ ALL TESTS PASSED! ({expected_count} objects tracked)\n")
    else:
        print(f"\n⚠️  Warning: Expected {expected_count} objects, got {len(tracker.all_obj_ids)}\n")
    
    # ===== Memory Analysis =====
    print("\n" + "=" * 80)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 80)
    print(f"{'Stage':<20} {'Frame':<8} {'Allocated (MB)':<18} {'Reserved (MB)':<15} {'Delta (MB)'}")
    print("-" * 80)
    
    for i, (stage, frame, alloc, res) in enumerate(memory_log):
        if i == 0:
            delta = 0
        else:
            delta = alloc - memory_log[i-1][2]
        
        frame_str = str(frame) if frame != 0 or stage == 'Initialize' else '-'
        print(f"{stage:<20} {frame_str:<8} {alloc:<18.2f} {res:<15.2f} {delta:+.2f}")
    
    print("-" * 80)
    print(f"Total Allocated: {memory_log[-1][2]:.2f} MB")
    print(f"Total Reserved:  {memory_log[-1][3]:.2f} MB")
    print(f"Peak Delta:      {max(memory_log[i][2] - memory_log[i-1][2] for i in range(1, len(memory_log))):.2f} MB")
    print("=" * 80 + "\n")
    
    return tracker, memory_log


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_add_objects_hardcoded_with_memory_log.py <video_dir>")
        print("\nExample:")
        print("  python test_add_objects_hardcoded_with_memory_log.py ./video_frames")
    else:
        video_dir = sys.argv[1]
        tracker, memory_log = test_add_objects_hardcoded(video_dir)
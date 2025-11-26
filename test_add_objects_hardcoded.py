# test_add_objects_hardcoded.py
import os
import torch
from PIL import Image
import numpy as np
import gc

from tracking_wrapper_mot import DAM4SAMMOT


def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_add_objects_hardcoded(video_dir, checkpoint_dir='./checkpoints'):
    """
    Test with hardcoded bboxes for quick testing.
    
    ⚠️ MODIFY THESE BBOXES BASED ON YOUR VIDEO:
    """
    
    # ===== 🔧 CONFIGURE BBOXES HERE =====
    # Format: [x, y, width, height]
    
    # Initial objects at frame 0
    INIT_BBOXES = [
        [520, 85, 220, 640],  # Object 1
        [725, 140, 175, 580],  # Object 2
    ]
    
    # New object at frame 8
    NEW_BBOXES = [
        [0, 150, 50, 300],  # Object 3
    ]
    # ===================================
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print(f"\n{'='*80}")
    print(f"Testing DAM4SAM: Add New Objects (Hardcoded BBoxes)")
    print(f"{'='*80}\n")
    
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
            f"{i:08d}.jpg",
            f"{i:08d}.png",
            f"{i:04d}.jpg",
            f"{i:04d}.png",
            f"{i}.jpg",
            f"{i}.png",
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
    
    # Initialize tracker
    print("Initializing tracker...")
    tracker = DAM4SAMMOT(
        model_size='small',
        checkpoint_dir=checkpoint_dir,
        offload_state_to_cpu=True
    )
    print("✓ Tracker initialized\n")
    
    # ===== Frame 0: Initialize with hardcoded bboxes =====
    print("=" * 80)
    print(f"FRAME 0: Initializing with {len(INIT_BBOXES)} objects")
    print(f"  BBoxes: {INIT_BBOXES}")
    print("=" * 80)
    
    init_regions = [{'bbox': bbox} for bbox in INIT_BBOXES]
    tracker.initialize(frames[0], init_regions)
    print(f"✓ Initialized with {len(init_regions)} objects")
    print(f"  Tracking IDs: {tracker.all_obj_ids}\n")
    
    # ===== Frames 1-7: Track =====
    print("=" * 80)
    print("FRAMES 1-7: Tracking initial objects")
    print("=" * 80)
    
    for frame_idx in range(1, 8):
        outputs = tracker.track(frames[frame_idx])
        mask_sizes = [int(mask.sum()) for mask in outputs['masks']]
        print(f"Frame {frame_idx}: {len(outputs['masks'])} objects | "
              f"Mask sizes: {mask_sizes}")
    
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
    print(f"  Tracked {len(outputs['masks'])} objects | Mask sizes: {mask_sizes}\n")
    
    # Add new object
    print("Step 2: Adding new object...")
    new_regions = [{'bbox': bbox} for bbox in NEW_BBOXES]
    
    new_ids = tracker.add_new_objects(
        frame_idx=8,
        image=frames[8],
        regions=new_regions
    )
    
    print(f"\n✓ Added {len(new_ids)} new object(s)")
    print(f"  New IDs: {new_ids}")
    print(f"  Total objects: {len(tracker.all_obj_ids)} | IDs: {tracker.all_obj_ids}\n")
    
    # ===== Frames 9-19: Track all =====
    print("=" * 80)
    print("FRAMES 9-19: Tracking all objects")
    print("=" * 80)
    
    for frame_idx in range(9, 20):
        outputs = tracker.track(frames[frame_idx])
        mask_sizes = [int(mask.sum()) for mask in outputs['masks']]
        print(f"Frame {frame_idx}: {len(outputs['masks'])} objects | "
              f"Mask sizes: {mask_sizes}")
    
    # Summary
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
    
    return tracker


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python test_add_objects_hardcoded.py <video_dir>")
        print("\nExample:")
        print("  python test_add_objects_hardcoded.py ./video_frames")
        print("\n⚠️  Remember to modify INIT_BBOXES and NEW_BBOXES in the code!")
    else:
        video_dir = sys.argv[1]
        tracker = test_add_objects_hardcoded(video_dir)
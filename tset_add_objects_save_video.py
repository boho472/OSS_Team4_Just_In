import os
import torch
from PIL import Image
import numpy as np
import cv2
import gc
from datetime import datetime

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


def get_color_map(num_objects):
    """Generate distinct colors for each object"""
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring Green
        (255, 0, 128),    # Rose
    ]
    while len(colors) < num_objects:
        colors.append((
            np.random.randint(50, 256),
            np.random.randint(50, 256),
            np.random.randint(50, 256)
        ))
    return colors[:num_objects]


def draw_tracking_results(frame, masks, obj_ids, colors, frame_idx, alpha=0.5):
    """
    Draw masks, bboxes, and labels on frame

    Args:
        frame: PIL Image or numpy array (H, W, 3) in RGB
        masks: list of binary masks (H, W)
        obj_ids: list of object IDs
        colors: list of RGB tuples
        frame_idx: current frame index
        alpha: transparency for masks

    Returns:
        numpy array (H, W, 3) in BGR for OpenCV
    """
    # Convert PIL to numpy if needed
    if isinstance(frame, Image.Image):
        frame = np.array(frame)

    frame = frame.copy()
    overlay = frame.copy()

    # Draw masks
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = color

    # Blend with original frame
    frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    # Draw bboxes and labels
    for mask, obj_id, color in zip(masks, obj_ids, colors):
        if mask.sum() == 0:
            continue

        # Get bbox from mask
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            continue

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # Draw bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)

        # Draw label background
        label = f"ID: {obj_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(
            label, font, font_scale, thickness)

        cv2.rectangle(frame,
                      (x_min, y_min - text_h - 10),
                      (x_min + text_w + 8, y_min),
                      color, -1)

        # Draw label text
        cv2.putText(frame, label,
                    (x_min + 4, y_min - 6),
                    font, font_scale, (255, 255, 255), thickness)

    # Add frame info overlay
    info_text = f"Frame: {frame_idx} | Objects: {len(obj_ids)}"
    cv2.rectangle(frame, (10, 10), (450, 55), (0, 0, 0), -1)
    cv2.putText(frame, info_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def test_add_objects_with_video(video_dir, checkpoint_dir='./checkpoints',
                                output_dir='./output_videos', fps=10):
    """
    Test with hardcoded bboxes + save tracking video

    Args:
        video_dir: Directory containing input frames
        checkpoint_dir: SAM2 checkpoint directory
        output_dir: Directory to save output video
        fps: Output video FPS
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

    # Memory logging
    memory_log = []

    # Video setup
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"tracking_{timestamp}.mp4")

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print(f"\n{'='*80}")
    print(f"Testing DAM4SAM: Add New Objects (With Video Saving)")
    print(f"{'='*80}")
    print(f"📹 Output video: {output_path}")
    print(f"🎬 FPS: {fps}\n")

    # Initial memory
    alloc, res = get_gpu_memory()
    memory_log.append(('Start', 0, alloc, res))
    print(
        f"Initial GPU Memory: Allocated={alloc:.2f}MB, Reserved={res:.2f}MB\n")

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

    # Setup video writer
    # PIL: (width, height)
    height, width = frames[0].size[1], frames[0].size[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Generate colors for max objects
    max_objects = len(INIT_BBOXES) + len(NEW_BBOXES)
    colors = get_color_map(max_objects)

    # Track results for video
    all_results = []

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
    print(
        f"After Tracker Init: Allocated={alloc:.2f}MB, Reserved={res:.2f}MB\n")

    # ===== Frame 0: Initialize =====
    print("=" * 80)
    print(f"FRAME 0: Initializing with {len(INIT_BBOXES)} objects")
    print(f"  BBoxes: {INIT_BBOXES}")
    print("=" * 80)

    init_regions = [{'bbox': bbox} for bbox in INIT_BBOXES]
    tracker.initialize(frames[0], init_regions)

    # Get initial masks (create from bboxes for visualization)
    init_masks = []
    for bbox in INIT_BBOXES:
        mask = np.zeros((height, width), dtype=np.uint8)
        x, y, w, h = bbox
        mask[y:y+h, x:x+w] = 1
        init_masks.append(mask)

    all_results.append({
        'frame_idx': 0,
        'masks': init_masks,
        'obj_ids': tracker.all_obj_ids.copy()
    })

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

        all_results.append({
            'frame_idx': frame_idx,
            'masks': outputs['masks'],
            'obj_ids': tracker.all_obj_ids.copy()
        })

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
    print(
        f"  Tracked {len(outputs['masks'])} objects | Mask sizes: {mask_sizes}")
    print(f"  Memory: {alloc:.2f}MB\n")

    # Add new object
    print("Step 2: Adding new object...")
    new_regions = [{'bbox': bbox} for bbox in NEW_BBOXES]

    new_ids = tracker.add_new_objects(
        frame_idx=8,
        image=frames[8],
        regions=new_regions
    )

    # Re-track to get all masks including new object
    outputs = tracker.track(frames[8])

    all_results.append({
        'frame_idx': 8,
        'masks': outputs['masks'],
        'obj_ids': tracker.all_obj_ids.copy()
    })

    alloc, res = get_gpu_memory()
    memory_log.append(('Add Objects', 8, alloc, res))

    print(f"\n✓ Added {len(new_ids)} new object(s)")
    print(f"  New IDs: {new_ids}")
    print(
        f"  Total objects: {len(tracker.all_obj_ids)} | IDs: {tracker.all_obj_ids}")
    print(f"  Memory: {alloc:.2f}MB (Δ{alloc - memory_log[-2][2]:.2f}MB)\n")

    # ===== Frames 9-19: Track all =====
    print("=" * 80)
    print("FRAMES 9-19: Tracking all objects")
    print("=" * 80)

    for frame_idx in range(9, 20):
        outputs = tracker.track(frames[frame_idx])
        mask_sizes = [int(mask.sum()) for mask in outputs['masks']]

        all_results.append({
            'frame_idx': frame_idx,
            'masks': outputs['masks'],
            'obj_ids': tracker.all_obj_ids.copy()
        })

        alloc, res = get_gpu_memory()
        memory_log.append(('Track', frame_idx, alloc, res))

        print(f"Frame {frame_idx}: {len(outputs['masks'])} objects | "
              f"Mask sizes: {mask_sizes} | "
              f"Memory: {alloc:.2f}MB (Δ{alloc - memory_log[-2][2]:.2f}MB)")

    # ===== Generate Video =====
    print("\n" + "=" * 80)
    print("GENERATING VIDEO...")
    print("=" * 80)

    for result in all_results:
        frame_idx = result['frame_idx']
        masks = result['masks']
        obj_ids = result['obj_ids']

        # Get colors for current objects
        current_colors = [colors[i] for i in range(len(obj_ids))]

        # Draw tracking results
        vis_frame = draw_tracking_results(
            frames[frame_idx],
            masks,
            obj_ids,
            current_colors,
            frame_idx
        )

        # Write frame
        video_writer.write(vis_frame)
        print(f"  Frame {frame_idx}: Written")

    video_writer.release()
    print(f"\n✅ Video saved: {output_path}\n")

    # ===== Summary =====
    print("=" * 80)
    print("TEST COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"✓ Initial objects at frame 0: {len(INIT_BBOXES)}")
    print(f"✓ New objects added at frame 8: {len(NEW_BBOXES)}")
    print(f"✓ Final tracking objects: {len(tracker.all_obj_ids)}")
    print(f"✓ Object IDs: {tracker.all_obj_ids}")
    print(f"✓ Output video: {output_path}")
    print("=" * 80)

    # Validation
    expected_count = len(INIT_BBOXES) + len(NEW_BBOXES)
    if len(tracker.all_obj_ids) == expected_count:
        print(f"\n✅ ALL TESTS PASSED! ({expected_count} objects tracked)\n")
    else:
        print(
            f"\n⚠️  Warning: Expected {expected_count} objects, got {len(tracker.all_obj_ids)}\n")

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
    print(
        f"Peak Delta:      {max(memory_log[i][2] - memory_log[i-1][2] for i in range(1, len(memory_log))):.2f} MB")
    print("=" * 80 + "\n")

    return tracker, memory_log, output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\nUsage:")
        print(
            "  python test_add_objects_save_video.py <video_dir> [output_dir] [fps]")
        print("\nExample:")
        print("  python test_add_objects_save_video.py ./video_frames")
        print("  python test_add_objects_save_video.py ./video_frames ./outputs 15")
        print("\n⚠️  Remember to modify INIT_BBOXES and NEW_BBOXES in the code!")
    else:
        video_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else './output_videos'
        fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10

        tracker, memory_log, video_path = test_add_objects_with_video(
            video_dir,
            output_dir=output_dir,
            fps=fps
        )

        print(f"\n🎬 Video saved at: {video_path}")
        print("You can download it from Colab or view it directly!\n")

import os
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import gc
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from tracking_wrapper_mot import DAM4SAMMOT


def show_image_with_grid(image_path, grid_size=50):
    """
    Display image with grid overlay for bbox coordinate reference.
    
    Args:
        image_path: Path to the image
        grid_size: Size of grid cells in pixels
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Create drawing context
    draw = ImageDraw.Draw(img)
    
    # Draw vertical grid lines
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill='cyan', width=1)
        # Add x coordinate labels
        if x % (grid_size * 2) == 0:
            draw.text((x + 2, 5), str(x), fill='cyan')
    
    # Draw horizontal grid lines
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill='cyan', width=1)
        # Add y coordinate labels
        if y % (grid_size * 2) == 0:
            draw.text((5, y + 2), str(y), fill='cyan')
    
    # Display
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.title(f'Frame with Grid (grid size: {grid_size}px)\nUse coordinates to define bbox: [x, y, width, height]')
    plt.axis('on')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    print(f"\nImage size: {width} x {height}")
    print(f"Grid size: {grid_size} pixels")
    print("\n📍 How to read coordinates:")
    print("  - Find the top-left corner of your object")
    print("  - Read the x (horizontal) and y (vertical) values from the grid")
    print("  - Estimate width and height of the bounding box")
    print("  - Format: [x, y, width, height]")
    
    return width, height


def show_image_with_bboxes(image_path, bboxes, labels=None):
    """
    Display image with drawn bounding boxes for verification.
    
    Args:
        image_path: Path to the image
        bboxes: List of bboxes in [x, y, width, height] format
        labels: Optional list of labels for each bbox
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'orange']
    
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        # Draw label
        label = labels[i] if labels and i < len(labels) else f"Object {i+1}"
        draw.text((x, y - 15), label, fill=color)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.title('Bounding Boxes Verification')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_bboxes_manual_input(frame_path, num_objects, grid_size=50):
    """
    Get bounding boxes through manual input with grid reference.
    
    Args:
        frame_path: Path to the frame image
        num_objects: Number of objects to annotate
        grid_size: Grid size for reference
    
    Returns:
        List of bboxes in [x, y, width, height] format
    """
    print(f"\n{'='*80}")
    print(f"Manual BBox Input: {os.path.basename(frame_path)}")
    print(f"{'='*80}\n")
    
    # Show grid
    width, height = show_image_with_grid(frame_path, grid_size)
    
    bboxes = []
    
    for i in range(num_objects):
        print(f"\n--- Object {i+1}/{num_objects} ---")
        
        while True:
            try:
                bbox_str = input(f"Enter bbox (format: x,y,w,h): ").strip()
                
                # Parse input
                parts = [int(p.strip()) for p in bbox_str.split(',')]
                
                if len(parts) != 4:
                    print("❌ Error: Please enter exactly 4 values (x,y,w,h)")
                    continue
                
                x, y, w, h = parts
                
                # Validate
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    print("❌ Error: Invalid bbox values (must be positive)")
                    continue
                
                if x + w > width or y + h > height:
                    print(f"⚠️  Warning: BBox exceeds image bounds ({width}x{height})")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                bboxes.append([x, y, w, h])
                print(f"✓ BBox {i+1} added: {[x, y, w, h]}")
                break
                
            except ValueError:
                print("❌ Error: Invalid format. Use: x,y,w,h (e.g., 100,150,200,300)")
            except KeyboardInterrupt:
                print("\n\n⚠️  Input cancelled")
                return None
    
    # Verify
    print(f"\n{'='*80}")
    print("Verification: Showing bboxes on image...")
    print(f"{'='*80}\n")
    
    show_image_with_bboxes(frame_path, bboxes, 
                          labels=[f"Obj {i+1}" for i in range(len(bboxes))])
    
    confirm = input("\n✓ Are these bboxes correct? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Restarting bbox input...")
        return get_bboxes_manual_input(frame_path, num_objects, grid_size)
    
    return bboxes


def test_add_objects_colab(video_dir, checkpoint_dir='./checkpoints', grid_size=50):
    """
    Test adding objects in Google Colab with grid-based bbox selection.
    
    Scenario:
    - Frame 0: Manually input 2 initial object bboxes
    - Frames 1-7: Track 2 objects
    - Frame 8: Manually input 1 new object bbox
    - Frames 9-19: Track all 3 objects
    
    Args:
        video_dir: Directory containing video frames
        checkpoint_dir: SAM2 checkpoint directory
        grid_size: Grid size for coordinate reference (default: 50px)
    """
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print(f"\n{'='*80}")
    print(f"Testing DAM4SAM: Add New Objects (Google Colab)")
    print(f"{'='*80}\n")
    
    # Load frames
    print("Loading frames...")
    frames = []
    frame_paths = []
    
    for i in range(20):
        frame_path = os.path.join(video_dir, f"frame_{i:04d}.jpg")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(video_dir, f"{i:08d}.jpg")
        if not os.path.exists(frame_path):
            # Try .png
            frame_path = os.path.join(video_dir, f"frame_{i:04d}.png")
        if not os.path.exists(frame_path):
            frame_path = os.path.join(video_dir, f"{i:08d}.png")
        
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame {i} not found in {video_dir}")
        
        frames.append(Image.open(frame_path).convert("RGB"))
        frame_paths.append(frame_path)
    
    print(f"✓ Loaded {len(frames)} frames (resolution: {frames[0].size})\n")
    
    # Initialize tracker
    print("Initializing tracker...")
    tracker = DAM4SAMMOT(
        model_size='base',
        checkpoint_dir=checkpoint_dir,
        offload_state_to_cpu=True
    )
    print("✓ Tracker initialized\n")
    
    # ===== Frame 0: Manual bbox input =====
    print("=" * 80)
    print("FRAME 0: Define 2 initial objects")
    print("=" * 80)
    
    init_bboxes = get_bboxes_manual_input(frame_paths[0], num_objects=2, grid_size=grid_size)
    
    if init_bboxes is None or len(init_bboxes) != 2:
        print("❌ Error: Need exactly 2 objects for initialization")
        return
    
    init_regions = [{'bbox': bbox} for bbox in init_bboxes]
    
    print("\nInitializing tracker with objects...")
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
    print("FRAME 8: Adding new object")
    print("=" * 80)
    
    # Track first
    print("\nStep 1: Tracking existing objects...")
    outputs = tracker.track(frames[8])
    mask_sizes = [int(mask.sum()) for mask in outputs['masks']]
    print(f"  Tracked {len(outputs['masks'])} objects | Mask sizes: {mask_sizes}\n")
    
    # Manual bbox input for new object
    print("Step 2: Define new object...")
    new_bboxes = get_bboxes_manual_input(frame_paths[8], num_objects=1, grid_size=grid_size)
    
    if new_bboxes is None or len(new_bboxes) != 1:
        print("❌ Error: Need exactly 1 new object")
        return
    
    new_regions = [{'bbox': bbox} for bbox in new_bboxes]
    
    print("\nAdding new object to tracker...")
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
    print(f"✓ Initial objects at frame 0: 2")
    print(f"✓ New object added at frame 8: 1")
    print(f"✓ Final tracking objects: {len(tracker.all_obj_ids)}")
    print(f"✓ Object IDs: {tracker.all_obj_ids}")
    print("=" * 80)
    
    # Validation
    if len(tracker.all_obj_ids) == 3 and tracker.all_obj_ids == [1, 2, 3]:
        print("\n✅ ALL TESTS PASSED!\n")
    else:
        print(f"\n⚠️  Warning: Expected 3 objects [1,2,3], got {tracker.all_obj_ids}\n")
    
    return tracker


# ===== Quick Helper Functions =====

def quick_preview(video_dir, frame_idx=0, grid_size=50):
    """Quickly preview a frame with grid"""
    frame_path = os.path.join(video_dir, f"frame_{frame_idx:04d}.jpg")
    if not os.path.exists(frame_path):
        frame_path = os.path.join(video_dir, f"{frame_idx:08d}.jpg")
    
    show_image_with_grid(frame_path, grid_size)


def quick_verify_bbox(video_dir, frame_idx, bboxes):
    """Quickly verify bboxes on a frame"""
    frame_path = os.path.join(video_dir, f"frame_{frame_idx:04d}.jpg")
    if not os.path.exists(frame_path):
        frame_path = os.path.join(video_dir, f"{frame_idx:08d}.jpg")
    
    show_image_with_bboxes(frame_path, bboxes)


# ===== Usage Example =====

if __name__ == "__main__":
    """
    Example usage in Google Colab:
    
    # 1. Preview frames to understand the scene
    quick_preview('./video_frames', frame_idx=0, grid_size=50)
    quick_preview('./video_frames', frame_idx=8, grid_size=50)
    
    # 2. Run the test (will prompt for bbox input)
    tracker = test_add_objects_colab('./video_frames', grid_size=50)
    
    # 3. Optionally verify bboxes before running
    test_bboxes = [[100, 150, 200, 300], [400, 200, 180, 280]]
    quick_verify_bbox('./video_frames', 0, test_bboxes)
    """
    
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage in Colab:")
        print("="*60)
        print("# Step 1: Preview frames with grid")
        print("quick_preview('./video_frames', frame_idx=0)")
        print("quick_preview('./video_frames', frame_idx=8)")
        print()
        print("# Step 2: Run test (will prompt for bbox coordinates)")
        print("tracker = test_add_objects_colab('./video_frames')")
        print("="*60)
    else:
        video_dir = sys.argv[1]
        test_add_objects_colab(video_dir)

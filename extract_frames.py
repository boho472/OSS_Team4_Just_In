import cv2
import os

def extract_frames(video_path, output_dir):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print()
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # SAM2 expects 8-digit frame numbers starting from 1
        frame_path = os.path.join(output_dir, f"{frame_idx+1:08d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"Extracted {frame_idx + 1}/{total_frames} frames")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\n✓ Extracted {frame_idx} frames to {output_dir}")

if __name__ == "__main__":
    video_path = "n1_video.mp4"
    output_dir = "n1_frames"
    
    extract_frames(video_path, output_dir)

import cv2
import os
import glob
import numpy as np

def images_to_video(image_folder, output_video, fps=5):
    """
    Convert a sequence of images to a video file.
    
    Args:
        image_folder: Path to folder containing images
        output_video: Path to output video file
        fps: Frames per second for the output video
    """
    # Get all jpg files and sort them
    images = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    
    if not images:
        print(f"No images found in {image_folder}")
        return
    
    print(f"Found {len(images)} images")
    print(f"First image: {images[0]}")
    print(f"Last image: {images[-1]}")
    
    # Read the first image to get dimensions
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"Failed to read first image: {images[0]}")
        return
    
    height, width, layers = frame.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write each image to video
    for i, image_path in enumerate(images):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Failed to read {image_path}")
            continue
        out.write(frame)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} frames")
    
    out.release()
    print(f"\nVideo saved to: {output_video}")
    print(f"Total frames: {len(images)}")
    print(f"FPS: {fps}")
    print(f"Duration: {len(images)/fps:.2f} seconds")

if __name__ == "__main__":
    image_folder = "n1"
    output_video = "n1_video.mp4"
    
    images_to_video(image_folder, output_video, fps=5)

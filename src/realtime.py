"""
Real-time Webcam Tracking Pipeline
Integrates YOLO → ZoeDepth → HybridTrack → DAM4SAM
"""

import cv2
import torch
import numpy as np
from collections import deque
import time
from pathlib import Path
import json

from model.model_YOLO import use_YOLO
from model.model_ZoeDepth import use_ZoeDepth
from model.model_DAM4SAM import DAM4SAMIntegration, extract_ht_for_dam4sam, visualize_segmentation
from tracker.hybridtrack import HYBRIDTRACK
from configs.config_utils import cfg, cfg_from_yaml_file
from ultralytics import YOLO
from PIL import Image


class RealtimeTracker:
    """Real-time tracking pipeline for webcam input"""
    
    def __init__(self, config_path, model_size='tiny', save_output=False):
        """
        Args:
            config_path: Path to config YAML file
            model_size: DAM4SAM model size ('tiny', 'small', 'base', 'large')
            save_output: Whether to save processed frames
        """
        # Load configuration
        self.cfg = cfg_from_yaml_file(config_path, cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize models
        print("Loading models...")
        self.yolo_det = YOLO("yolo11n.pt")
        self.yolo_seg = YOLO("yolo11n-seg.pt")
        self.depth_model = torch.hub.load(
            "isl-org/ZoeDepth", "ZoeD_NK", pretrained=True
        ).to(self.device).eval()
        
        # Initialize tracker
        self.tracker = HYBRIDTRACK(
            box_type="Kitti",
            tracking_features=False,
            config=self.cfg
        )
        
        # Initialize DAM4SAM
        checkpoint_dir = getattr(self.cfg, 'checkpoint_dir', 'src/checkpoints')
        self.dam4sam = DAM4SAMIntegration(
            model_size=model_size,
            checkpoint_dir=checkpoint_dir
        )
        
        # Frame management
        self.frame_idx = 0
        self.new_info_dict = {}
        self.dict_key = []
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # Output settings
        self.save_output = save_output
        if save_output:
            self.output_dir = Path("realtime_output")
            self.output_dir.mkdir(exist_ok=True)
            (self.output_dir / "frames").mkdir(exist_ok=True)
            (self.output_dir / "json").mkdir(exist_ok=True)
    
    def process_frame(self, frame):
        """
        Process a single frame through the entire pipeline
        
        Args:
            frame: BGR image from webcam (numpy array)
            
        Returns:
            visualized_frame: Frame with tracking visualization
            fps: Current FPS
        """
        start_time = time.time()
        
        # Convert BGR to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Save temporary frame for model processing
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        try:
            # 1. YOLO Detection + Segmentation
            boxes, masks, scores = use_YOLO(temp_path, self.yolo_det, self.yolo_seg)
            
            # 2. Depth Estimation
            depth_map = use_ZoeDepth(temp_path, self.depth_model, self.device)
            
            # 3. Convert to 3D (simplified - we need bounding box info)
            objects_3d = self._convert_to_tracking_format(boxes, depth_map, scores)
            
            if len(objects_3d) > 0:
                # 4. HybridTrack
                self.tracker.tracking(
                    objects_3d[:, :7],
                    features=None,
                    scores=torch.tensor(scores),
                    timestamp=self.frame_idx
                )
                
                # 5. Extract HybridTrack results
                hybridtrack_data = self._extract_ht_results()
                
                # 6. DAM4SAM processing
                pil_image = Image.fromarray(frame_rgb)
                dam_outputs = self.dam4sam.tracker.process_frame_with_ht_data(
                    frame_idx=self.frame_idx,
                    ht_data=hybridtrack_data,
                    image=pil_image
                )
                
                # 7. Visualize results
                if dam_outputs.get('mask_arrays'):
                    masks_arr = dam_outputs['mask_arrays']
                    meta_list = dam_outputs.get('masks', [])
                    obj_ids = [m.get('internal_id', i) for i, m in enumerate(meta_list)]
                    
                    visualized_frame = visualize_segmentation(
                        image=frame_rgb,
                        masks=masks_arr,
                        obj_ids=obj_ids,
                        scores=None,
                        alpha=0.5
                    )
                    visualized_frame = cv2.cvtColor(visualized_frame, cv2.COLOR_RGB2BGR)
                else:
                    visualized_frame = frame.copy()
            else:
                visualized_frame = frame.copy()
            
            # Save output if enabled
            if self.save_output:
                frame_name = f"frame_{self.frame_idx:06d}.jpg"
                cv2.imwrite(str(self.output_dir / "frames" / frame_name), visualized_frame)
            
            self.frame_idx += 1
            
        except Exception as e:
            print(f"Error processing frame {self.frame_idx}: {e}")
            visualized_frame = frame.copy()
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_queue.append(fps)
        avg_fps = np.mean(self.fps_queue)
        
        # Draw FPS on frame
        cv2.putText(
            visualized_frame,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        return visualized_frame, avg_fps
    
    def _convert_to_tracking_format(self, boxes, depth_map, scores):
        """Convert 2D boxes + depth to 3D tracking format"""
        if len(boxes) == 0:
            return np.array([])
        
        objects_3d = []
        for bbox, score in zip(boxes, scores):
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            
            # Get depth at center
            if depth_map is not None:
                cy_int, cx_int = int(cy), int(cx)
                cy_int = np.clip(cy_int, 0, depth_map.shape[0] - 1)
                cx_int = np.clip(cx_int, 0, depth_map.shape[1] - 1)
                z = depth_map[cy_int, cx_int]
            else:
                z = 10.0  # default depth
            
            # Simplified 3D format: [x, y, z, l, w, h, ry]
            # Using image coordinates as proxy for world coordinates
            obj = [cx / 100.0, cy / 100.0, z, h / 100.0, w / 100.0, 1.8, 0.0]
            objects_3d.append(obj)
        
        return np.array(objects_3d)
    
    def _extract_ht_results(self):
        """Extract HybridTrack results for DAM4SAM"""
        ht_data = []
        
        if not hasattr(self.tracker, 'tracks') or len(self.tracker.tracks) == 0:
            return ht_data
        
        for track in self.tracker.tracks:
            if not hasattr(track, 'det_bbox') or track.det_bbox is None:
                continue
            
            # Convert bbox format (assuming det_bbox is in some format)
            bbox = track.det_bbox
            
            # Handle different bbox formats
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                ht_data.append({
                    'object_id': track.track_id,
                    'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                })
            elif hasattr(bbox, 'x') and hasattr(bbox, 'y'):
                ht_data.append({
                    'object_id': track.track_id,
                    'bbox': [int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)]
                })
        
        return ht_data
    
    def run(self, camera_id=0, display=True, max_frames=None):
        """
        Run real-time tracking on webcam feed
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            display: Whether to display video window
            max_frames: Maximum number of frames to process (None for infinite)
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting real-time tracking...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame, fps = self.process_frame(frame)
                
                # Display
                if display:
                    cv2.imshow('Real-time Tracking', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    save_path = f"snapshot_{self.frame_idx}.jpg"
                    cv2.imwrite(save_path, processed_frame)
                    print(f"Saved snapshot: {save_path}")
                
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    print(f"Reached max frames ({max_frames})")
                    break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print(f"Processed {frame_count} frames")
            print(f"Average FPS: {np.mean(self.fps_queue):.2f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time tracking with webcam')
    parser.add_argument('--cfg', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--model-size', type=str, default='tiny',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='DAM4SAM model size')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display window')
    parser.add_argument('--save', action='store_true',
                       help='Save processed frames to disk')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    # Create tracker
    tracker = RealtimeTracker(
        config_path=args.cfg,
        model_size=args.model_size,
        save_output=args.save
    )
    
    # Run
    tracker.run(
        camera_id=args.camera,
        display=not args.no_display,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
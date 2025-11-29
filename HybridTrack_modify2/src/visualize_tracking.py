import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def draw_bbox(img, bbox, color, label, thickness=2):
    """
    이미지에 bounding box와 라벨 그리기
    
    Args:
        img: numpy array (H, W, 3)
        bbox: dict with 'x', 'y', 'w', 'h'
        color: (B, G, R) tuple
        label: str
        thickness: int
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    
    # 박스 그리기
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    # 라벨 배경
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - label_size[1] - 5), 
                  (x + label_size[0], y), color, -1)
    
    # 라벨 텍스트
    cv2.putText(img, label, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def draw_mask_overlay(img, mask, color, alpha=0.5):
    """
    이미지에 mask overlay 그리기
    
    Args:
        img: numpy array (H, W, 3)
        mask: binary mask (H, W)
        color: (B, G, R) tuple
        alpha: float, transparency
    """
    overlay = img.copy()
    overlay[mask > 0] = color
    
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return img


def generate_color(obj_id):
    """객체 ID에 따라 고유 색상 생성"""
    np.random.seed(obj_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


def create_tracking_video(json_dir, output_path, fps=10):
    """
    JSON 결과를 읽어서 추적 비디오 생성
    
    Args:
        json_dir: JSON 파일들이 있는 디렉토리
        output_path: 출력 비디오 경로
        fps: 프레임 레이트
    """
    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("frame_*.json"))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    print(f"Found {len(json_files)} frames")
    print(f"Creating video: {output_path}")
    
    # 첫 프레임으로 이미지 크기 확인
    with open(json_files[0], 'r') as f:
        first_frame = json.load(f)
    
    # 더미 이미지 생성 (실제로는 이미지 파일에서 읽어야 함)
    img_width = 1024
    img_height = 1024
    
    # 비디오 writer 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                   (img_width * 2, img_height))  # 2배 폭 (HT | DAM)
    
    # 각 프레임 처리
    for frame_idx, json_path in enumerate(json_files):
        print(f"Processing frame {frame_idx}...")
        
        with open(json_path, 'r') as f:
            frame_data = json.load(f)
        
        # 베이스 이미지 생성 (실제로는 실제 이미지 로드)
        img = create_frame_image(frame_idx, img_width, img_height)
        
        # 왼쪽: HybridTrack 결과
        img_ht = img.copy()
        ht_results = frame_data['dam4sam_tracking']['HybridTrack_results']
        
        for ht_obj in ht_results:
            obj_id = ht_obj['object_id']
            bbox = ht_obj['bbox']
            color = generate_color(obj_id)
            
            label = f"HT-{obj_id}"
            img_ht = draw_bbox(img_ht, bbox, color, label, thickness=3)
        
        # 오른쪽: DAM4SAM 결과
        img_dam = img.copy()
        dam_results = frame_data['dam4sam_tracking']['DAM4SAM_results']
        
        for dam_obj in dam_results:
            internal_id = dam_obj['internal_id']
            bbox = dam_obj['bbox']
            pixels = dam_obj['mask_pixels']
            color = generate_color(internal_id)
            
            # mask가 0이면 빨간색으로 경고
            if pixels == 0:
                color = (0, 0, 255)  # 빨강
                label = f"DAM-{internal_id} ⚠️ LOST"
            else:
                label = f"DAM-{internal_id} ({pixels}px)"
            
            img_dam = draw_bbox(img_dam, bbox, color, label, thickness=3)
        
        # 액션 로그 표시 (있으면)
        if 'actions' in frame_data['dam4sam_tracking']:
            actions = frame_data['dam4sam_tracking']['actions']
            y_pos = 30
            
            for action in actions:
                ht_id = action['ht_obj_id']
                act = action['action']
                
                if act == 'INIT':
                    text = f"HT-{ht_id}: INIT"
                    color = (0, 255, 0)  # 초록
                elif act == 'NEW':
                    dam_id = action['internal_id']
                    text = f"HT-{ht_id} → DAM-{dam_id}: NEW"
                    color = (255, 0, 0)  # 파랑
                elif act == 'FILTER':
                    dam_id = action['internal_id']
                    overlap = action['overlap_ratio']
                    text = f"HT-{ht_id} → DAM-{dam_id}: FILTER ({overlap:.2f})"
                    color = (0, 255, 255)  # 노랑
                
                cv2.putText(img_dam, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 30
        
        # 프레임 번호 표시
        cv2.putText(img_ht, f"Frame {frame_idx}", (10, img_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_dam, f"Frame {frame_idx}", (10, img_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 타이틀
        cv2.putText(img_ht, "HybridTrack", (img_width // 2 - 100, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img_dam, "DAM4SAM", (img_width // 2 - 80, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # 좌우 결합
        combined = np.hstack([img_ht, img_dam])
        
        # 중앙선 그리기
        cv2.line(combined, (img_width, 0), (img_width, img_height),
                (255, 255, 255), 2)
        
        # 비디오에 쓰기
        video_writer.write(combined)
    
    video_writer.release()
    print(f"✅ Video saved: {output_path}")


def create_frame_image(frame_idx, width, height):
    """
    프레임 이미지 생성 (더미 또는 실제 이미지 로드)
    
    Args:
        frame_idx: 프레임 번호
        width: 이미지 너비
        height: 이미지 높이
    
    Returns:
        img: numpy array (H, W, 3) BGR
    """
    # 그라디언트 배경
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        intensity = int(255 * y / height)
        img[y, :] = [intensity // 3, intensity // 2, intensity]
    
    # 객체들 시뮬레이션 (실제 테스트 시나리오에 맞춤)
    # 객체 1: 계속 보임, 오른쪽으로 이동
    x1 = 100 + frame_idx * 5
    y1 = 100
    cv2.rectangle(img, (x1, y1), (x1 + 50, y1 + 50), (0, 0, 255), -1)
    
    # 객체 2: 계속 보임, 아래로 이동
    x2 = 200
    y2 = 200 + frame_idx * 3
    cv2.rectangle(img, (x2, y2), (x2 + 60, y2 + 60), (0, 255, 0), -1)
    
    # 객체 3: Frame 5부터 등장
    if frame_idx >= 5:
        x3 = 300
        y3 = 300 + (frame_idx - 5) * 2
        cv2.rectangle(img, (x3, y3), (x3 + 55, y3 + 55), (255, 0, 0), -1)
    
    return img


def create_tracking_video_with_masks(tracker, json_dir, image_dir, output_path, fps=10):
    """
    실제 mask를 포함한 추적 비디오 생성
    
    Args:
        tracker: DAM4SAMMOT instance
        json_dir: JSON 결과 디렉토리
        image_dir: 이미지 파일 디렉토리
        output_path: 출력 비디오 경로
        fps: 프레임 레이트
    """
    json_dir = Path(json_dir)
    image_dir = Path(image_dir)
    
    json_files = sorted(json_dir.glob("frame_*.json"))
    image_files = sorted(image_dir.glob("frame_*.jpg"))
    
    if not json_files or not image_files:
        print("No files found")
        return
    
    # 첫 이미지로 크기 확인
    first_img = cv2.imread(str(image_files[0]))
    height, width = first_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx, (json_path, img_path) in enumerate(zip(json_files, image_files)):
        print(f"Frame {frame_idx}...")
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        
        # JSON 로드
        with open(json_path, 'r') as f:
            frame_data = json.load(f)
        
        dam_results = frame_data['dam4sam_tracking']['DAM4SAM_results']
        
        # DAM4SAM의 실제 mask 그리기
        for dam_obj in dam_results:
            internal_id = dam_obj['internal_id']
            
            # tracker에서 실제 mask 가져오기
            if internal_id < len(tracker.all_obj_ids):
                obj_id = tracker.all_obj_ids[internal_id]
                obj_mem = tracker.per_object_outputs_all[obj_id]
                
                if obj_mem:
                    latest = obj_mem[-1]
                    pred_mask = latest['pred_masks']
                    
                    if isinstance(pred_mask, np.ndarray):
                        mask = pred_mask[0, 0]
                    else:
                        mask = pred_mask[0, 0].cpu().numpy()
                    
                    # 리사이즈
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask, (width, height))
                    
                    mask_binary = (mask > 0).astype(np.uint8)
                    
                    # Mask overlay
                    color = generate_color(internal_id)
                    img = draw_mask_overlay(img, mask_binary, color, alpha=0.4)
            
            # Bbox 그리기
            bbox = dam_obj['bbox']
            pixels = dam_obj['mask_pixels']
            color = generate_color(internal_id)
            
            if pixels == 0:
                color = (0, 0, 255)
                label = f"⚠️ {internal_id} LOST"
            else:
                label = f"#{internal_id} ({pixels}px)"
            
            img = draw_bbox(img, bbox, color, label)
        
        # 프레임 정보
        cv2.putText(img, f"Frame {frame_idx}", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        video_writer.write(img)
    
    video_writer.release()
    print(f"✅ Video with masks saved: {output_path}")
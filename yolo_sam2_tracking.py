"""
YOLOv11 + SAM2 통합 자동 추적 시스템

YOLOv11로 첫 프레임에서 객체를 자동 감지하고,
SAM2로 비디오 전체에서 추적합니다.
"""

import sys
print("Starting imports...", flush=True)
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

print("Imports done.", flush=True)


class YOLOSAMTracker:
    def __init__(self, video_dir, sam_checkpoint, sam_config, output_dir, yolo_model="yolo11n.pt"):
        """
        Args:
            video_dir: 비디오 프레임 디렉토리
            sam_checkpoint: SAM2 체크포인트 경로
            sam_config: SAM2 설정 파일 경로
            output_dir: 출력 디렉토리
            yolo_model: YOLO 모델 (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # YOLO 모델 로드
        print("Loading YOLOv11 model...")
        self.yolo = YOLO(yolo_model)
        print(f"✓ YOLOv11 loaded: {yolo_model}\n")
        
        # SAM2 모델 로드
        print("Loading SAM2 model...")
        self.predictor = build_sam2_video_predictor(sam_config, sam_checkpoint)
        self.inference_state = self.predictor.init_state(video_path=video_dir)
        print("✓ SAM2 loaded successfully\n")
        
        self.current_obj_id = 1
    
    def detect_objects_yolo(self, frame_idx=0, conf_threshold=0.5, target_classes=None):
        """
        YOLO로 객체 감지
        
        Args:
            frame_idx: 감지할 프레임 인덱스
            conf_threshold: 신뢰도 임계값
            target_classes: 감지할 클래스 리스트 (None이면 모든 클래스)
                          예: [0] = person만, [0, 2, 3] = person, car, motorcycle
        
        Returns:
            detections: [(class_id, class_name, confidence, bbox), ...]
        """
        img_path = os.path.join(self.video_dir, f"{frame_idx+1:08d}.jpg")
        
        print(f"{'='*80}")
        print(f"YOLOv11 객체 감지 - Frame {frame_idx}")
        print(f"{'='*80}")
        
        # YOLO 추론
        results = self.yolo(img_path, conf=conf_threshold, verbose=False)
        
        # (이전 디버그 코드는 track_video 내부의 상세 디버깅으로 대체하므로 제거하거나 유지해도 됨. 
        #  여기서는 중복되므로 제거하고 track_video에서 통합 관리하는 것이 깔끔함.
        #  하지만 사용자가 요청했던 기능이므로 유지하되, 파일명만 다르게 함)
        # annotated_frame = results[0].plot()
        # debug_dir = os.path.join(self.output_dir, "yolo_raw_debug")
        # os.makedirs(debug_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(debug_dir, f"yolo_raw_{frame_idx:08d}.jpg"), annotated_frame)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                
                # 클래스 필터링
                if target_classes is None or class_id in target_classes:
                    detections.append((class_id, class_name, confidence, bbox))
                    print(f"  ✓ {class_name} (conf: {confidence:.2f}) - BBox: {bbox.astype(int)}")
        
        print(f"\n총 {len(detections)}개 객체 감지됨\n")
        return detections
    
    def add_detections_to_sam(self, frame_idx, detections):
        """
        YOLO 감지 결과를 SAM2에 추가
        
        Args:
            frame_idx: 프레임 인덱스
            detections: YOLO 감지 결과
        """
        print(f"{'='*80}")
        print(f"SAM2에 객체 등록 - Frame {frame_idx}")
        print(f"{'='*80}")
        
        for i, (class_id, class_name, confidence, bbox) in enumerate(detections):
            # BBox를 SAM2 형식으로 변환 [x1, y1, x2, y2]
            box = np.array(bbox, dtype=np.float32)
            
            try:
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=self.current_obj_id,
                    box=box,
                )
                print(f"  ✓ Object {self.current_obj_id}: {class_name} (conf: {confidence:.2f})")
                self.current_obj_id += 1
            except Exception as e:
                print(f"  ✗ Failed to add {class_name}: {e}")
        
        print(f"\n총 {self.current_obj_id - 1}개 객체 등록 완료\n")
        return self.current_obj_id - 1

    def compute_iou(self, box1, box2):
        """
        두 박스 간의 IoU 계산
        box: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        if union <= 0:
            return 0
        return intersection / union

    def compute_containment(self, box1, box2):
        """
        box1이 box2 안에 포함되는 비율 계산 (Intersection / Area_of_box1)
        box: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        if area1 <= 0:
            return 0
        return intersection / area1



    def mask_to_box(self, mask):
        """
        이진 마스크를 바운딩 박스로 변환
        mask: 2D numpy array (boolean or 0/1)
        Returns: [x1, y1, x2, y2] or None if empty
        """
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return None
        
        x1 = np.min(x_indices)
        x2 = np.max(x_indices)
        y1 = np.min(y_indices)
        y2 = np.max(y_indices)
        
        return [x1, y1, x2, y2]

    
    def track_video(self, start_frame=0, end_frame=None, yolo_interval=1):
        """
        SAM2로 비디오 추적 및 YOLO를 이용한 자동 객체 추가
        
        Args:
            start_frame: 시작 프레임
            end_frame: 끝 프레임 (None이면 끝까지)
            yolo_interval: YOLO 실행 주기 (프레임 단위)
        """
        print(f"{'='*80}")
        print(f"SAM2 비디오 추적 및 자동 객체 감지 시작")
        print(f"{'='*80}")
        
        if end_frame is None:
            # 전체 프레임 수 계산
            frame_files = sorted([f for f in os.listdir(self.video_dir) if f.endswith('.jpg')])
            end_frame = len(frame_files) - 1
        
        num_frames = end_frame - start_frame + 1
        print(f"프레임 범위: {start_frame} ~ {end_frame} (총 {num_frames} 프레임)")
        print(f"YOLO 감지 주기: 매 {yolo_interval} 프레임")
        
        # 프레임별 루프
        for frame_idx in range(start_frame, end_frame + 1):
            print(f"\n--- Frame {frame_idx} ---")
            
            current_masks = []
            current_obj_ids = []
            
            # 1. 현재 추적 중인 객체가 있으면 SAM2 전파 (1 프레임)
            if self.inference_state["obj_ids"]:
                try:
                    # propagate_in_video는 제너레이터이므로 next()로 1프레임 실행
                    # max_frame_num_to_track=1로 설정하여 현재 프레임만 처리
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                        self.inference_state, 
                        start_frame_idx=frame_idx,
                        max_frame_num_to_track=1
                    ):
                        if out_frame_idx == frame_idx:
                            current_obj_ids = out_obj_ids
                            current_masks = []
                            # Logits -> Mask 변환
                            for i in range(len(out_obj_ids)):
                                mask = (out_mask_logits[i].cpu().numpy() > 0.0).squeeze()
                                current_masks.append(mask)
                            
                            print(f"  SAM2 Tracking: {len(current_obj_ids)} objects")
                except Exception as e:
                    print(f"  ⚠ SAM2 Propagation Error: {e}")
            else:
                print(f"  SAM2 Tracking: No objects tracked yet")

                # 2. YOLO로 새 객체 감지 (주기에 따라)
            if frame_idx % yolo_interval == 0:
                # person 클래스(0)만 감지
                detections = self.detect_objects_yolo(frame_idx, conf_threshold=0.5, target_classes=[0])
                
                # 디버깅용 이미지 복사
                debug_img = cv2.imread(os.path.join(self.video_dir, f"{frame_idx+1:08d}.jpg"))
                
                for class_id, class_name, conf, bbox in detections:
                    # YOLO 박스 그리기 (빨간색)
                    cv2.rectangle(debug_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                    cv2.putText(debug_img, f"YOLO {conf:.2f}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    best_iou = 0
                    best_containment = 0
                    matched_obj_id = None
                    matched_mask_idx = -1
                    
                    # 현재 추적 중인 객체들과 비교
                    for mask_idx, mask in enumerate(current_masks):
                        tracked_box = self.mask_to_box(mask)
                        if tracked_box is not None:
                            iou = self.compute_iou(bbox, tracked_box)
                            containment = self.compute_containment(bbox, tracked_box)
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_containment = containment
                                matched_obj_id = current_obj_ids[mask_idx]
                                matched_mask_idx = mask_idx
                            
                            # 시각화용: SAM2 박스 (초록색)
                            cv2.rectangle(debug_img, (int(tracked_box[0]), int(tracked_box[1])), (int(tracked_box[2]), int(tracked_box[3])), (0, 255, 0), 1)
                    
                    # 매칭 판단 로직
                    # 1. Refine: IoU > 0.7 or Containment > 0.8
                    if best_iou > 0.7 or best_containment > 0.8:
                        print(f"  [Refine] Update Object {matched_obj_id} (IoU: {best_iou:.2f}, Cont: {best_containment:.2f})")
                        cv2.putText(debug_img, f"Refine Obj{matched_obj_id}", (int(bbox[0]), int(bbox[1]-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # SAM2 상태 업데이트 (기존 ID 유지)
                        try:
                            box = np.array(bbox, dtype=np.float32)
                            _, _, _ = self.predictor.add_new_points_or_box(
                                inference_state=self.inference_state,
                                frame_idx=frame_idx,
                                obj_id=matched_obj_id,
                                box=box,
                            )
                        except Exception as e:
                            print(f"  ⚠ Refine Failed for Object {matched_obj_id}: {e}")
                        
                    # 2. Ignore: 0.3 < IoU <= 0.7
                    elif best_iou > 0.3:
                        print(f"  [Ignore] Ambiguous match with Object {matched_obj_id} (IoU: {best_iou:.2f})")
                        cv2.putText(debug_img, f"Ignore (IoU {best_iou:.2f})", (int(bbox[0]), int(bbox[1]-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                        
                    # 3. New Object: IoU <= 0.3
                    else:
                        print(f"  [New] Add new object {self.current_obj_id} (Max IoU: {best_iou:.2f})")
                        cv2.putText(debug_img, f"New Obj{self.current_obj_id}", (int(bbox[0]), int(bbox[1]-25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # SAM2에 새 객체 추가
                        try:
                            box = np.array(bbox, dtype=np.float32)
                            _, _, _ = self.predictor.add_new_points_or_box(
                                inference_state=self.inference_state,
                                frame_idx=frame_idx,
                                obj_id=self.current_obj_id,
                                box=box,
                            )
                            self.current_obj_id += 1
                        except Exception as e:
                            print(f"  ⚠ New Object Addition Failed: {e}")

                # 디버그 이미지 저장
                debug_dir = os.path.join(self.output_dir, "yolo_debug")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"debug_match_{frame_idx:08d}.jpg"), debug_img)
            
            # 4. 시각화 저장
            # 현재 프레임의 마스크 정보를 모아서 저장
            # 주의: 새로 추가된 객체는 current_masks에 없을 수 있음 (다음 프레임부터 반영)
            # 하지만 add_new_points_or_box를 호출하면 해당 프레임의 마스크가 생성됨.
            # 이를 반영하려면 구조를 조금 더 개선해야 하지만, 일단 기존 추적 결과만 시각화.
            
            # 시각화를 위한 텐서 재구성 (기존 + 신규는 복잡하므로 기존 것만 표시하거나, 
            # propagate 결과를 사용. 신규 추가된 건 다음 프레임부터 보임)
            
            if current_masks:
                # 리스트를 텐서로 변환하여 save_visualization에 전달
                # save_visualization은 [N, H, W] 형태의 logits나 mask를 기대함
                # 여기서는 binary mask 리스트이므로 변환 필요
                
                # current_masks는 numpy array list
                combined_masks = np.stack(current_masks) # [N, H, W]
                combined_masks = torch.from_numpy(combined_masks).float().to(self.predictor.device)
                # Logits 형태로 변환하지 않고 바로 넘기려면 save_visualization 수정 필요
                # 하지만 save_visualization은 logits > 0을 체크함.
                # 0/1 마스크를 logits처럼 쓰려면 0 -> -10, 1 -> 10 등으로 변환하거나
                # save_visualization을 수정.
                # 여기서는 간단히 logits 형태로 근사: 1 -> 10.0, 0 -> -10.0
                combined_logits = (combined_masks * 20.0) - 10.0
                
                self.save_visualization(frame_idx, combined_logits)
            else:
                # 객체가 없으면 원본 이미지 저장
                img_path = os.path.join(self.video_dir, f"{frame_idx+1:08d}.jpg")
                img = cv2.imread(img_path)
                
                vis_dir = os.path.join(self.output_dir, "sam_vis")
                os.makedirs(vis_dir, exist_ok=True)
                
                save_path = os.path.join(vis_dir, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(save_path, img)
        
        print(f"\n✓ 추적 완료: {end_frame - start_frame + 1} 프레임\n")
    
    def save_visualization(self, frame_idx, mask_logits):
        """시각화 저장 - 모든 객체를 하나의 프레임에 겹쳐서 표시"""
        img_path = os.path.join(self.video_dir, f"{frame_idx+1:08d}.jpg")
        img = np.array(Image.open(img_path))
        
        num_objects = mask_logits.shape[0]
        
        # 하나의 이미지에 모든 마스크 겹치기
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img)
        
        # 각 객체의 마스크를 다른 색상으로 겹쳐서 표시
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        color_names = ['Red', 'Blue', 'Green', 'Purple', 'Orange', 'Cyan', 'Magenta', 'Yellow']
        
        for i in range(num_objects):
            mask = (mask_logits[i].cpu().numpy() > 0).squeeze()
            
            # 마스크를 컬러로 변환
            colored_mask = np.zeros((*mask.shape, 4))
            if colors[i % len(colors)] == 'red':
                colored_mask[mask] = [1, 0, 0, 0.5]
            elif colors[i % len(colors)] == 'blue':
                colored_mask[mask] = [0, 0, 1, 0.5]
            elif colors[i % len(colors)] == 'green':
                colored_mask[mask] = [0, 1, 0, 0.5]
            elif colors[i % len(colors)] == 'purple':
                colored_mask[mask] = [0.5, 0, 0.5, 0.5]
            elif colors[i % len(colors)] == 'orange':
                colored_mask[mask] = [1, 0.5, 0, 0.5]
            elif colors[i % len(colors)] == 'cyan':
                colored_mask[mask] = [0, 1, 1, 0.5]
            elif colors[i % len(colors)] == 'magenta':
                colored_mask[mask] = [1, 0, 1, 0.5]
            elif colors[i % len(colors)] == 'yellow':
                colored_mask[mask] = [1, 1, 0, 0.5]
            
            ax.imshow(colored_mask)
        
        # 범례 추가
        legend_text = ', '.join([f"Obj{i+1} ({color_names[i % len(color_names)]})" 
                                 for i in range(num_objects)])
        ax.set_title(f"Frame {frame_idx} - {num_objects} object(s): {legend_text}", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.tight_layout()
        
        # 시각화 저장 디렉토리 생성
        vis_dir = os.path.join(self.output_dir, "sam_vis")
        os.makedirs(vis_dir, exist_ok=True)
        
        plt.savefig(os.path.join(vis_dir, f"frame_{frame_idx:04d}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_video(self, fps=30):
        """저장된 프레임들을 동영상으로 생성"""
        print(f"\n{'='*80}")
        print("동영상 생성 중...")
        print(f"{'='*80}\n")
        
        # 프레임 파일 리스트
        vis_dir = os.path.join(self.output_dir, "sam_vis")
        if not os.path.exists(vis_dir):
            print(f"⚠ 시각화 디렉토리가 없습니다: {vis_dir}")
            return None
            
        frame_files = sorted([
            os.path.join(vis_dir, f) 
            for f in os.listdir(vis_dir) 
            if f.startswith('frame_') and f.endswith('.png')
        ])
        
        if not frame_files:
            print(f"⚠ 프레임 파일이 없습니다.")
            return None
        
        print(f"총 {len(frame_files)}개 프레임 발견")
        
        # 첫 번째 프레임으로 크기 확인
        first_frame = cv2.imread(frame_files[0])
        height, width, _ = first_frame.shape
        
        # 동영상 작성기 생성
        video_path = os.path.join(self.output_dir, "yolo_sam2_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # 프레임 추가
        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(frame_file)
            video_writer.write(frame)
            
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(frame_files)} 프레임 처리됨")
        
        video_writer.release()
        
        # 파일 크기 확인
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        print(f"\n{'='*80}")
        print(f"✓ 동영상 생성 완료!")
        print(f"{'='*80}")
        print(f"파일: {video_path}")
        print(f"총 프레임: {len(frame_files)}")
        print(f"FPS: {fps}")
        print(f"해상도: {width}x{height}")
        print(f"파일 크기: {file_size_mb:.2f} MB")
        print(f"재생 시간: {len(frame_files)/fps:.2f}초")
        print(f"{'='*80}\n")
        
        return video_path


def main():
    # 설정
    video_dir = "datasets/VOT_Workspace/sequences/handball1/color"
    sam_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    sam_config = "sam21pp_hiera_l.yaml"
    output_dir = "output_yolo_sam2"
    
    print("=" * 80)
    print("YOLOv11 + SAM2 자동 추적 시스템")
    print("=" * 80)
    print()
    
    # 트래커 초기화
    tracker = YOLOSAMTracker(
        video_dir=video_dir,
        sam_checkpoint=sam_checkpoint,
        sam_config=sam_config,
        output_dir=output_dir,
        yolo_model="yolo11n.pt"  # nano 모델 (빠름)
    )
    
    # 1단계 ~ 3단계 통합: 추적 및 자동 객체 추가
    # YOLO가 매 프레임(또는 주기적으로) 실행되어 새로운 객체를 찾고 추가합니다.
    tracker.track_video(
        start_frame=0, 
        end_frame=100,  # 끝까지
        yolo_interval=5  # 5 프레임마다 YOLO 실행 (메모리 절약)
    )
    
    # 4단계: 동영상 생성
    tracker.create_video(fps=30)
    
    print("=" * 80)
    print("✓ 모든 작업 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

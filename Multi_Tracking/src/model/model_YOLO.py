import cv2
import numpy as np

def use_YOLO(image_path,yolo_det,yolo_seg):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    det = yolo_det(img_rgb,imgsz=(640, 416),conf=0.38,iou=0.3)[0]
    
    person_boxes = []
    for box in det.boxes:
        if int(box.cls.item()) != 0:
            continue
        score = float(box.conf.item())
        if score < 0.45:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        if (x2 - x1) < 40 or (y2 - y1) < 60:
            continue
        person_boxes.append([x1, y1, x2, y2])
        
    seg_masks = []
    for (x1, y1, x2, y2) in person_boxes:
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            seg_masks.append(None)
            continue
        seg = yolo_seg(crop, imgsz=(640,416))[0]
        if seg.masks is None:
            seg_masks.append(None)
            continue
        masks_np = seg.masks.data.cpu().numpy()
        areas = [np.sum(m > 0.5) for m in masks_np]
        idx_max = int(np.argmax(areas))
        mask_crop = (masks_np[idx_max] > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask_crop, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        mask_full = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        mask_full[y1:y2, x1:x2] = mask_resized
        seg_masks.append(mask_full)
        
    return person_boxes, seg_masks
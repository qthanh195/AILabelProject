from ultralytics import YOLO
import numpy as np
import cv2

model_segment_label = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\detect_label_segment.pt")
model_classifi_label = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_label_classification.pt")
model_detect_logo_tdc = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_logo_tdc.pt")
model_detect_khoiluong_tdc = YOLO("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project/ai_service\model\model_detect_khoiluong.pt")

def detectLabel(image):
    crop, rect_label = None, None
    results = model_segment_label.predict(image)
    for idx, result in enumerate(results):
        if result.masks is None:
            continue
        for i, (seg, cls) in enumerate(zip(result.masks.xy, result.boxes.cls)):
            polygon = np.array(seg, dtype=np.int32)
            
            x, y, w, h = cv2.boundingRect(polygon)
            rect_label = ((x, y), (x + w, y + h))

            # 1. Tìm hình chữ nhật xoay bao quanh polygon
            rect = cv2.minAreaRect(polygon)
            box = cv2.boxPoints(rect)
            box = np.int8(box)

            # 2. Lấy ma trận xoay
            center, size, angle = rect
            size = tuple([int(s) for s in size])
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # 3. Xoay toàn ảnh
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            # 4. Crop vùng rectangle đã xoay
            x, y = int(center[0] - size[0] / 2), int(center[1] - size[1] / 2)
            w, h = size
            crop = rotated[y:y+h, x:x+w]
    return crop, rect_label
    
def classifiLabel(image):
    id, class_name, confidence = None, None, None
    results = model_classifi_label(image)
    id = results[0].probs.top1
    class_name = results[0].names[id]
    confidence = results[0].probs.top1conf.item()
    return id, class_name, confidence
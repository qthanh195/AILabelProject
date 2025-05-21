from camera_handler import BaslerCamera
import cv2
import numpy as np
import base64


class ApiHandler(BaslerCamera):
    def __init__(self):
        super().__init__()
    
    def analyze_image(self, name_a, name_b, name_c, 
                            name_d, name_e, name_f, 
                            thresh_a, thresh_b, thresh_c, 
                            thresh_d, thresh_e, thresh_f):
        origin_image = cv2.imread("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project\data_test\img_20250507_095427.jpg")
        crop_label = cv2.imread("E:/2. GE/22. Vedan Vision Ocr\code/ai_label_project\data_test\img_20250507_095317_obj.jpg")
        
        return "label C", 0.83, self._image_to_base64(origin_image), self._image_to_base64(crop_label)
        
    def _image_to_base64(self, image_np: np.ndarray) -> str:
        # Chuyển ndarray sang buffer ảnh (dạng JPEG)
        _, buffer = cv2.imencode('.jpg', image_np)
        # Mã hóa sang base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
import cv2
import numpy as np
import base64
import re
from camera_handler import BaslerCamera
from ai_hander import detectLabel, classifiLabel
from ocr_engine import classifi_tdc_with_ocr


class ApiHandler(BaslerCamera):
    def __init__(self):
        super().__init__()
    
    def analyze_image(self, name_a, name_b, name_c, 
                            name_d, name_e, name_f, 
                            thresh_a, thresh_b, thresh_c, 
                            thresh_d, thresh_e, thresh_f):
        # 1. Chụp một ảnh gốc
        # 2. Phát hiện có nhãn hay không
        # 3. Phát hiện là nhãn nào
        # 4. kiểm tra là nhãn nào trong 6 nhãn
        image = None
        label_detect = "other label"
        conf = 0.00
        origin_image = np.ones((480, 640), dtype=np.uint8) * 255
        label_image = origin_image.copy()
        # Chụp một ảnh từ camera 
        image = self.get_image()
        if image is None:
            print("Không lấy được ảnh từ camera!")
            return "No Image", 0.0, self._image_to_base64(origin_image),self._image_to_base64(label_image)
        if len(image.shape) == 2:  # ảnh grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:  # ảnh có shape (H, W, 1)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Phát hiện có nhãn trong ảnh hay không
        label_image, rect_label = detectLabel(image)
        if image is None:
            print("No Label!")
            origin_image = cv2.putText(image, f"None Labels",(100,100),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
        else:
            id, class_name, confidence = classifiLabel(label_image)
            # Nếu là nhãn tdc thì kiểm tra xem là nhãn tdc nào
            if id == 22: # 22: 'image30_1' -> tdc
               class_name, label_image = classifi_tdc_with_ocr(label_image)
            print(class_name)
            origin_image = cv2.rectangle(image, rect_label[0], rect_label[1], (0,255,0), thickness= 6)
            cv2.putText(
    origin_image,
    f"Label {re.search(r'image(\d+)_1', class_name).group(1)} - {confidence:.2f}",
    (rect_label[0][0], rect_label[0][1]-30),
    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6
)
            conf = float(f"{confidence:.2f}")
            # kiểm tra xem là palet nào?
            match_class = re.search(r"image(\d+)_1", class_name)
            if match_class:
                number_classname = int(match_class.group(1))
            
            label_names = ["A", "B", "C", "D", "E", "F"]
            list_name_label = [name_a, name_b, name_c, name_d, name_e, name_f]
            list_threshold = [thresh_a, thresh_b, thresh_c, thresh_d, thresh_e, thresh_f]
            for idx, name_label in enumerate(list_name_label):
                match_label = re.search(r"label(\d+)", name_label)
                if match_label:
                    number_name_label = int(match_label.group(1))
                    if number_name_label == number_classname and confidence >= list_threshold[idx]:
                        label_detect = f"Pallet {label_names[idx]}"
                        return label_detect, conf, self._image_to_base64(origin_image),self._image_to_base64(label_image)

        return label_detect, conf, self._image_to_base64(origin_image),self._image_to_base64(label_image)
            
    def _image_to_base64(self, image_np: np.ndarray) -> str:
        # Chuyển ndarray sang buffer ảnh (dạng JPEG)
        _, buffer = cv2.imencode('.jpg', image_np)
        # Mã hóa sang base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def api_open_camera(self):
        self.open_camera()
        if self.is_open:
            print("Đã mở camera.")
        else:
            print("Không mở được camera.")
        
    def api_close_camera(self):
        self.close_camera()
        if not self.is_open:
            print("Đã tắt camera.")
        else:
            print("Đang mở camera.")
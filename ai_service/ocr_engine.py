from rapidfuzz import process, fuzz
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from ai_hander import model_detect_logo_tdc, model_detect_khoiluong_tdc
from process_image import rotate_image, transform_point, crop_rotated_contour, get_best_match, draw_text_with_pillow

valid_tdc = ["でん粉「TW-100」", 
             "食品用タピオカでん粉「BK-V」", 
             "食品用タピオカでん粉「BK-V3」",
             "イモのちから",
             "食品用タピオカでん粉「ES-5」",
             "食品用タピオカでん粉「SK-08」", 
             "食品用タピオカでん粉「タピオカV3」",
             "食品用タピオカでん粉「タピオカV」", 
             "食品用タピオカでん粉「FM-5」", 
             "食品用タピオカでん粉「RT-90」",
             "食品用タピオカでん粉「タピオカV2」", 
             "食品用タピオカでん粉「BK-V7」",]

valid_tdc_kg = ["20kg", "25kg", "12.5kg", "18kg", "12kg",]

def classifi_tdc_with_ocr(image): 
    # image = cv2.imread(image)
    w, h = image.shape[1], image.shape[0]
    # tạo mask với kích thước lớn hơn ảnh gốc 10
    mask = np.zeros((h+10, w+10, 3), dtype=np.uint8)
    
    #ghép ảnh vào giữa mask
    mask[5:h+5, 5:w+5] = image
    
    results = model_detect_logo_tdc.predict(source=image)
    if results is None or len(results) == 0 or results[0].boxes is None:
        return False, None
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if center_x < w // 2 and center_y < h // 2:
                print("Nhan tren trai")
                new_img = mask
            elif center_x > w // 2 and center_y < h // 2:
                print("Nhan tren phai")
                new_img = rotate_image(mask, 90)
                center_x, center_y = transform_point((center_x, center_y), image, 90)
            elif center_x < w // 2 and center_y > h // 2:
                print("Nhan duoi trai")
                new_img = rotate_image(mask, -90)
                center_x, center_y = transform_point((center_x, center_y), image, -90)
            elif center_x > w // 2 and center_y > h // 2:
                print("Nhan duoi phai")
                new_img = rotate_image(mask, 180)
                center_x, center_y = transform_point((center_x, center_y), image, 180)

            # Chuyển đổi ảnh sang grayscale
            gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            # Áp dụng threshold
            _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Tìm contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Lọc contours theo diện tích nhỏ hơn max_area
            max_area = (w+10) * (h+10)
            filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]

            if filtered_contours:
                # Tìm contour lớn nhất trong danh sách đã lọc
                contour = max(filtered_contours, key=cv2.contourArea)
            else:
                print("Không có contour nào thỏa mãn điều kiện.")
                contour = None
                
            if contour is not None:
                image_crop = crop_rotated_contour(new_img, contour)
                results = model_detect_khoiluong_tdc.predict(source=image_crop)
                if results is None or len(results) == 0 or results[0].boxes is None:
                    return False, None
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        image_crop1 = image_crop[y1-4:y2+4, x1:x2]
                        image_crop2 = image_crop[y1-98:y2-80, x1-5:x2+400]

                        text1 = pytesseract.image_to_string(image_crop1, config=r'--oem 3 --psm 8 -l eng')
                        text2 = pytesseract.image_to_string(image_crop2, config=r'--oem 3 --psm 7 -l jpn')

                        text1 = get_best_match(''.join(text1.split()).strip().lower(), valid_tdc_kg)
                        text2 = get_best_match(''.join(text2.split()).strip().upper(), valid_tdc)

                        if text1 is not None:
                            image_crop = draw_text_with_pillow(image_crop, text1, (x1, y1-20), font_path="simsun.ttc", font_size=20, color=(0, 255, 0))
                        if text2 is not None:
                            image_crop = draw_text_with_pillow(image_crop, text2, (x1, y1-110), font_path="simsun.ttc", font_size=20, color=(0, 255, 0))
                        
                        print("text1:", text1)
                        print("text2:", text2)
from ai_service.api_handler import ApiHandler
import cv2


api = ApiHandler()
def main():
    label_detected, pallet_detect, confidence, origin_image, cropped_label = api.analyze_image("label1.jpg", "label2.jpg", "label3.jpg", "label4.jpg", "label5.jpg", "label6.jpg", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    print("label_detected: ", label_detected)   
    print("pallet_detect: ", pallet_detect)
    print("confidence: ", confidence)
    cv2.imwrite("origin_image.jpg", origin_image)
    cv2.imwrite("cropped_label.jpg", cropped_label)
    
if __name__ == '__main__':
    main()
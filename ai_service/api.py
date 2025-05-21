from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from schemas.image_schemas import ImageCaptureRequest
from api_handler import ApiHandler

api_handel = ApiHandler()
app = FastAPI()

@app.post("/image_capture")
def capture_image(req: ImageCaptureRequest):
    label_detected, confidence, origin_image, cropped_label = api_handel.analyze_image(req.name_a, req.name_b, req.name_c, 
                                                                                req.name_d, req.name_e, req.name_f, 
                                                                                req.thresh_a, req.thresh_b, req.thresh_c, 
                                                                                req.thresh_d, req.thresh_e, req.thresh_f)
    
    return JSONResponse(content={
        "label_detected": label_detected,
        "confidence": confidence,
        "origin_image": origin_image,
        "cropped_image": cropped_label
    })
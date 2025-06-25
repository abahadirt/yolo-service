from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse

from app.model_utils import predict
from app.processing_utils import *


router = APIRouter()


@router.post("/detect/")
async def detect_all(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    # Apply letterbox and generate image tensor:
    input_tensor, original_shape, letterboxed_shape = preprocess_image(image_bytes)

    # Run inference and get raw predictions (bounding boxes, class scores, confidences):
    # Note: Bounding box coordinates are relative to the letterboxed image dimensions
    model_output = predict(request.app.state.onnx_session, request.app.state.onnx_input_name, input_tensor)

    # Postprocess raw predictions: rescale bounding boxes from letterboxed image back to original image size:
    final_detections = postprocess_predictions(model_output, original_shape, letterboxed_shape)

    # Draw boxes on original image:
    response_data = draw_and_encode_image(image_bytes, final_detections)

    return JSONResponse(content=response_data)


@router.post("/detect/{label}")
async def detect_by_label(request: Request, label: str, file: UploadFile = File(...)):
    label = label.lower()
    if label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid label. Please use one of the supported class names.")

    class_id = CLASS_NAMES.index(label)

    image_bytes = await file.read()

    input_tensor, original_shape, letterboxed_shape = preprocess_image(image_bytes)

    model_output = predict(request.app.state.onnx_session, request.app.state.onnx_input_name, input_tensor, filter_label_id=class_id)

    final_detections = postprocess_predictions(model_output, original_shape, letterboxed_shape)

    response_data = draw_and_encode_image(image_bytes, final_detections)

    return JSONResponse(content=response_data)
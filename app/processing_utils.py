import cv2
from app.config import *
from ultralytics.utils import ops
from fastapi import HTTPException
import numpy as np
import base64

def letterbox(im, new_shape=ONNX_MODEL_IMAGE_SIZE, color=(114, 114, 114)):

    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


    return im, ratio, (dw, dh)


def preprocess_image(image_bytes):

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Given image can not be decoded!")

    img_np_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    original_shape = img_bgr.shape[:2]



    img_letterboxed, ratio, (dw, dh) = letterbox(img_np_rgb)
    letterboxed_shape = img_letterboxed.shape[:2] # H, W of letterboxed

    img_tensor = img_letterboxed.transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = img_tensor.astype(np.float32) / 255.0
    if img_tensor.ndim == 3:
        img_tensor = np.expand_dims(img_tensor, 0)

    return img_tensor, original_shape, letterboxed_shape




def postprocess_predictions(model_output, original_shape, letterboxed_shape):

    raw_detections = model_output
    confidences_all = raw_detections[:, 4]
    mask = confidences_all > 0
    raw_detections = raw_detections[mask]


    boxes = []
    confidences = []
    class_ids = []


    for det in raw_detections:
        confidence = det[4]
        boxes.append(det[:4]) # x1, y1, x2, y2
        confidences.append(confidence)
        class_ids.append(int(det[5]))

    if not boxes:
        return []

    boxes_np = np.array(boxes)

    # Scale the boxes to the original image size
    scaled_boxes = ops.scale_boxes(
        img1_shape=letterboxed_shape, # (h,w)
        boxes=boxes_np,               # NMS boxes
        img0_shape=original_shape     # (h, w)
    ).round().astype(int)

    processed_predictions = []

    for i in range(len(scaled_boxes)):
        x1, y1, x2, y2 = scaled_boxes[i]
        class_id = class_ids[i]
        conf = confidences[i]

        try:
            label_name = CLASS_NAMES[class_id]
        except IndexError:
            label_name = f"unknown_class_{class_id}"

        processed_predictions.append({
            "label_id": int(class_id),
            "label": str(label_name),
            "x": int(x1), # LEFT CORNER
            "y": int(y1), # LEFT CORNER
            # "x2": int(x2),
            # "y2": int(y2),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "confidence": round(float(conf), 2)
        })

    return processed_predictions



def draw_and_encode_image(original_image_bytes, processed_predictions):
    """
        Decodes an image from bytes, draws bounding boxes and labels for the
        processed predictions onto it, and then encodes the modified image
        to a base64 string.

        Args:
            original_image_bytes (bytes): The raw bytes of the original image.
            processed_predictions (list): A list of prediction dictionaries, where each
                                          dictionary contains bounding box coordinates,
                                          label, and confidence.

        Returns:
            dict: A dictionary containing the base64 encoded image string with
                  drawings, the list of processed predictions, and the count
                  of detected objects.

        Raises:
            HTTPException: If the image cannot be decoded or if encoding to PNG/JPEG fails.

        Warning:
            This function internally decodes `original_image_bytes` into a NumPy array.
            The drawing operations (rectangles, text) are performed **in-place** on this
            decoded image array before it is encoded to base64.

            !!!! If the decoded image (NumPy array) were to be used elsewhere after this
            function call (which is not the case in the current implementation as
            the decoded array is local to this function), it would be modified.
            For safety and to prevent unintended side effects if the internal logic
            changes, consider that the input's visual representation is altered by this function.!!!!
        """

    nparr = np.frombuffer(original_image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # BGR HWC

    for obj in processed_predictions:
        x1, y1= obj["x"], obj["y"]
        x2, y2 = obj["width"] + x1 , obj["height"] + y1
        label = obj["label"]
        conf_text = f'{obj["confidence"]:.2f}'

        # Draw Boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Write Label Text:
        label_size, base_line = cv2.getTextSize(f'{label} {conf_text}', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1_label = max(y1, label_size[1] + 10) # # Ensure the label does not go outside the image boundaries

        cv2.rectangle(image, (x1, y1_label - label_size[1] - base_line), (x1 + label_size[0], y1_label + base_line//2), (0,0,0), cv2.FILLED) # Black background
        cv2.putText(image, f'{label} {conf_text}', (x1, y1_label-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA) # White text


    retval, buffer_cv2 = cv2.imencode(".png", image)  # Encode Type
    if not retval:
        raise HTTPException(status_code=500, detail="Failed to encode image to PNG/JPEG.")

    base64_img = base64.b64encode(buffer_cv2.tobytes()).decode("utf-8")  # Convert NP to Byte
    return {
        "image": base64_img,
        "objects": processed_predictions,
        "count": len(processed_predictions)
    }

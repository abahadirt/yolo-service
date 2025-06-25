# config.py

"""
The SELECTED_MODEL parameter must be chosen directly from the models defined in MODEL_CONFIGS.
To add a new model, make sure to define its configuration in MODEL_CONFIGS.

"""

# Model attributes can be extended or specified further depending on future changes in the code structure or the requirements of the model or service
# ------------------------------------------------------------
MODEL_CONFIGS = {
    "temp": {
        "path": "onnx_models/yolo11s_temp1.onnx",
        "image_size": (640, 640),
        "class_id_column_index": 5,
        # "confidence_threshold": 0.5,
    },
    "default": {
        "path": "onnx_models/yolo11s_default.onnx",
        "image_size": (640, 640),
        "class_id_column_index": 5,
        # "confidence_threshold": 0.4,
    },
    "balanced": {
        "path": "onnx_models/yolo11s_balanced.onnx",
        "image_size": (640, 640),
        "class_id_column_index": 5,
        # "confidence_threshold": 0.45,
    },
}


# ------------------------------------------------------------
# SELECT MODEL:
SELECTED_MODEL = "balanced"


# ------------------------------------------------------------

# ASSIGNING PARAMETERS:
MODEL_PATH = MODEL_CONFIGS[SELECTED_MODEL]["path"]
ONNX_MODEL_IMAGE_SIZE = MODEL_CONFIGS[SELECTED_MODEL]["image_size"]
CLASS_ID_COLUMN_INDEX = MODEL_CONFIGS[SELECTED_MODEL]["class_id_column_index"]
# CONF_THRESH = MODEL_CONFIGS[SELECTED_MODEL]["confidence_threshold"]

# ------------------------------------------------------------

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
] # COCO classes

NUM_CLASSES = len(CLASS_NAMES)
# ------------------------------------------------------------
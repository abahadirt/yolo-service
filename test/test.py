import onnx
from app.config import *


PROJECT_PATH =r"C:\Users\alibahadir\PycharmProjects\yolo_service_tractus/"
model_path = PROJECT_PATH+MODEL_PATH

model = onnx.load(model_path)
print("Opset version:", model.opset_import[0].version)
print(model.stride)
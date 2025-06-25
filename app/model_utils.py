import onnxruntime
from app.config import MODEL_PATH, CLASS_ID_COLUMN_INDEX


""" 
WARNING: The onnxruntime package and its dependencies (e.g., sympy, etc.) occupy approximately 150 MB in total. 
This can be significant in IoT and other resource-constrained environments.
Therefore, onnxruntime should be removed before production release.
"""
def load_model(model_path=MODEL_PATH):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session, input_name


"""
WARNING: Changing the model export type or model version may change the output tensor shape!
the shape (1, 1, 300, 6) may vary. Update this code accordingly if needed.
"""
def predict(session, input_name, input_tensor, filter_label_id=None):
    # Run model
    outputs = session.run(None, {input_name: input_tensor})[0][0] #(1, 1, 300, 6) -> (300,6)
    if filter_label_id is not None:

        detected_class_ids = outputs[:, CLASS_ID_COLUMN_INDEX]

        mask = (detected_class_ids == filter_label_id)

        outputs = outputs[mask]

    return outputs
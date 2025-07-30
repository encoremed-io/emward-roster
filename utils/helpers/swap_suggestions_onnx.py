import onnxruntime as ort
import numpy as np
import pandas as pd

# Load the model once
try:
    session = ort.InferenceSession("models/swap_suggestions/trained_model.onnx")
    input_name = session.get_inputs()[0].name
    print(f"[ONNX] Loaded model")
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {e}")


# Run predictions
def run_model_on(features_df: pd.DataFrame) -> np.ndarray:
    features = features_df.astype(np.float32).values
    predictions = session.run(None, {input_name: features})[0]
    return np.asarray(predictions)

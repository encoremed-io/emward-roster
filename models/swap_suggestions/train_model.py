# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
import os

# Create dummy data
np.random.seed(42)
n = 100
data = pd.DataFrame({
    "icu_certified": np.random.randint(0, 2, n),
    "prefers_morning": np.random.randint(0, 2, n),
    "shifts_this_week": np.random.randint(0, 7, n),
    "recent_night_shift": np.random.randint(0, 2, n),
})

# Create labels
data["swap_score"] = (
    (data["icu_certified"] & data["prefers_morning"]) &
    (data["shifts_this_week"] < 5) &
    (~data["recent_night_shift"].astype(bool))
).astype(int)

X = data.drop(columns=["swap_score"])
y = data["swap_score"]

# Train the model (80% train, 20% test)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# ML model that uses decision trees (10 trees, 42 random seeds)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Export ONNX
initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save Model to Disk
os.makedirs("models", exist_ok=True)
with open("models/swap_suggestions/trained_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

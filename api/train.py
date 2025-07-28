from fastapi import APIRouter, Body
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from typing import List
import os
from docs.train.candidates import train_candidates_description
from schemas.train.candidates import CandidatesTrainingRecord
from utils.helpers.swap_suggestions import extract_preference_features

router = APIRouter(prefix="/train", tags=["Train Models"])


@router.post(
    "/candidates",
    description=train_candidates_description,
    summary="Train Candidates Model",
)
def train_roster(records: List[CandidatesTrainingRecord] = Body(...)):
    all_candidates = []

    # Loops candidates data
    for record in records:
        base = record.model_dump()
        pref_features = extract_preference_features(base["preferences"])
        base.update(pref_features)
        all_candidates.append(base)

    # Checks for empty data
    if not all_candidates:
        return {"error": "No candidate data provided."}

    df = pd.DataFrame(all_candidates)

    # Checks if there are enough records to train the model
    if len(df) < 5:
        return {"error": "Insufficient data to train model."}

    # Feature matrix (input)
    X = df[
        [
            "isSenior",
            "isSpecialist",
            "shiftsThisWeek",
            "recentNightShift",
            "totalHoursThisWeek",
            "consecutiveDaysWorked",
            "dayAfterOffDay",
            "totalPreferredShifts",
            "uniquePreferredShiftTypes",
            "avgPreferredShiftId",
        ]
    ].astype(np.float32)

    # Target vector
    y = df["wasChosen"].astype(int)

    # ML model that uses decision trees (10 trees, 42 random seeds)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Export ONNX
    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    if isinstance(onnx_model, tuple):
        onnx_model = onnx_model[0]  # Extract the actual ONNX model

    # Save Model to Disk
    os.makedirs("models/swap_suggestions", exist_ok=True)
    with open("models/swap_suggestions/trained_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    return {
        "message": "Model trained and saved successfully.",
        "samplesTrainedOn": len(df),
        "featureImportances": dict(zip(X.columns, model.feature_importances_)),
    }

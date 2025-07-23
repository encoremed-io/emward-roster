from fastapi import APIRouter, Body
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from typing import List
from pydantic import BaseModel
import os

router = APIRouter(prefix="/train")

class CandidateFeatures(BaseModel):
    nurse_id: str
    icu_certified: bool
    prefers_morning: bool
    shifts_this_week: int
    recent_night_shift: bool
    was_chosen: bool

class ShiftInfo(BaseModel):
    date: str
    type: str
    department: str

class SwapTrainingRecord(BaseModel):
    target_shift: ShiftInfo
    target_nurse_id: str
    candidates: List[CandidateFeatures]

@router.post("/roster")
def train_roster(records: List[SwapTrainingRecord] = Body(...)):
    all_candidates = []

    # Loops candidates data
    for record in records:
        for c in record.candidates:
            all_candidates.append(c.dict())

    # Checks for empty data
    if not all_candidates:
        return {"error": "No candidate data provided."}

    df = pd.DataFrame(all_candidates)

    # Feature matrix (input)
    X = df[["icu_certified", "prefers_morning", "shifts_this_week", "recent_night_shift"]].astype(np.float32)

    # Target vector
    y = df["was_chosen"].astype(int)

    # ML model that uses decision trees (10 trees, 42 random seeds)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Export ONNX
    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save Model to Disk
    os.makedirs("models/swap_suggestions", exist_ok=True)
    with open("models/swap_suggestions/trained_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    return {
        "message": "Model trained and saved successfully.",
        "samples_trained_on": len(df)
    }

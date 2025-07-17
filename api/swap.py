from fastapi import APIRouter, Body
import pandas as pd
from utils.helpers.swap_suggestions import parse_date, generate_warning, is_conflicting, preprocess_nurse
from utils.helpers.swap_suggestions_onnx import run_model_on

router = APIRouter(prefix="/swap")

# get suggestions for swapping nurses shift
@router.post("/suggestions")
def suggest_swap(data: dict = Body(...)):
    target_shift = data["target_shift"]
    target_date = parse_date(target_shift["date"])
    shift_type = target_shift["type"]
    required_skill = target_shift.get("required_skill", "icu_certified")
    target_nurse_id = data.get("target_nurse_id")
    settings = data.get("settings", {
        "max_shifts_per_week": 5,
        "warn_recent_night_shift": True,
        "recent_night_window_days": 2
    })

    roster = data["roster"]

    # Find the nurse to be replaced
    assigned_nurse = next((n for n in roster if n["nurse_id"] == target_nurse_id), None)

    if not assigned_nurse:
        return {"message": f"Nurse '{target_nurse_id}' not found in roster."}

    target_role = assigned_nurse.get("role", "junior")

    # Count other seniors on the same shift
    other_seniors = [
        n for n in roster
        if n["nurse_id"] != target_nurse_id and
           n.get("role") == "senior" and
           any(s["date"] == target_shift["date"] and s["type"] == shift_type for s in n.get("shifts", []))
    ]

    must_replace_with_senior = target_role == "senior" and not other_seniors

    # Filter nurses into groups
    strict, relaxed, last_resort = [], [], []

    for nurse in roster:
        if nurse["nurse_id"] == target_nurse_id:
            continue

        if must_replace_with_senior and nurse.get("role") != "senior":
            continue
            
        is_busy = is_conflicting(nurse, target_shift)
        processed = preprocess_nurse(nurse, target_date, settings)

        if not is_busy and nurse.get(required_skill, False):
            strict.append(processed)
        elif not is_busy:
            relaxed.append(processed)
        else:
            last_resort.append(processed)
    
    # Apply fallback filtering
    top_candidates = None
    filter_level = ""

    for group, label in [(strict, "strict"), (relaxed, "relaxed"), (last_resort, "last_resort")]:
        if not group:
            continue
        df = pd.DataFrame(group)
        features = df[["icu_certified", "prefers_morning", "shifts_this_week", "recent_night_shift"]]
        df["swap_score"] = run_model_on(features)
        top = df.sort_values(by="swap_score", ascending=False).head(3)
        if not top.empty:
            top_candidates = top
            filter_level = label
            break

    # Format result
    results = []
    if top_candidates is not None and not top_candidates.empty:
        for _, row in top_candidates.iterrows():
            results.append({
                "nurse_id": row["nurse_id"],
                "swap_score": float(row["swap_score"]),
                "shifts_this_week": int(row["shifts_this_week"]),
                "recent_night_shift": int(row["recent_night_shift"]),
                "message": generate_warning(row, settings)
            })

    return {
        "original_nurse": assigned_nurse["nurse_id"],
        "replacement_for": {
            "date": target_shift["date"],
            "type": shift_type,
            "department": target_shift.get("department")
        },
        "filter_level": filter_level if filter_level else "none_available",
        "top_candidates": results
    }
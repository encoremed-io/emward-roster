from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from datetime import date
import numpy as np

from build_model import (
    load_nurse_profiles,
    load_shift_preferences,
    validate_nurse_data,
    build_schedule_model,
)
from nurse_env import compute_total_penalty

app = FastAPI(title="Nurse Roster Scheduler API")


class Override(BaseModel):
    nurse:      str = Field(..., description="Name of the nurse (must match profiles)")
    el_date:    date = Field(..., description="Which day to override (YYYY-MM-DD)")
    type:       str = Field(..., description="'EL' for emergency leave or one of SHIFT_LABELS")


class ScheduleRequest(BaseModel):
    profiles_path: str = Field(default=..., description="Path to nurse_profiles.xlsx")
    prefs_path:    str = Field(default=..., description="Path to nurse_preferences.xlsx")
    start_date:    date = Field(default=..., description="Schedule start date (YYYY-MM-DD)")
    num_days:      int  = Field(default=..., ge=1, description="Number of days to schedule")
    rl_assignment: Optional[List[List[List[int]]]] = Field(
        default=None,
        description="Optional RL warm-start: N×D×3 array of 0/1"
    )

    overrides: Optional[List[Override]] = Field(
        default=None,
        description="Optional list of per-nurse, per-day overrides (e.g. EL)"
    )

    class Config:
        schema_extra = {
            "example": {
                "profiles_path": "data/nurse_profiles.xlsx",
                "prefs_path":    "data/nurse_preferences.xlsx",
                "start_date":    "2025-06-01",
                "num_days":      14,
                "rl_assignment": [],
                "overrides": [
                    {"nurse":"S01","el_date":"2025-06-03","type":"EL"}
                ]
            }
        }


class ScheduleResponse(BaseModel):
    schedule: dict
    summary:  List[dict]
    penalty:  Optional[int] = None
    used_rl: bool


@app.post("/schedule", response_model=ScheduleResponse)
def make_schedule(req: ScheduleRequest):
    # 1) Load and validate data
    try:
        profiles = load_nurse_profiles(req.profiles_path)
        prefs    = load_shift_preferences(req.prefs_path)
        validate_nurse_data(profiles, prefs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) Build schedule
    rl_arr = np.array(req.rl_assignment, dtype=int) if req.rl_assignment else None
    
    if rl_arr is not None:
        rl_arr = np.array(req.rl_assignment, dtype=int)
        print("🔁 Warm-starting with RL assignment!")
    else:
        print("🔨 No warm start — CP-SAT only.")

    try:
        fixed: dict[tuple[str,int], str] = {}
        for ov in req.overrides or []:
            name = ov.nurse.strip().upper()
            if name not in profiles['Name'].str.upper().tolist():
                raise HTTPException(400, f"Unknown nurse in overrides: {ov.nurse}")
            day_idx = (ov.el_date - req.start_date).days
            if not (0 <= day_idx < req.num_days):
                raise HTTPException(400, f"Override date {ov.el_date} out of scheduling window")
            typ = ov.type.strip().upper()
            if typ != "EL":
                raise HTTPException(400, "Overrides currently only support type='EL'")
            fixed[(name, day_idx)] = "EL"

        sched_df, summ_df = build_schedule_model(
            profiles,
            prefs,
            pd.to_datetime(req.start_date),
            req.num_days,
            rl_assignment=rl_arr,
            fixed_assignments=fixed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scheduling failed: {e}")

    # 3) Compute penalty (optional)
    penalty = None
    try:
        nurse_names = sched_df.index.tolist()
        dates       = sched_df.columns.tolist()
        N, D = len(nurse_names), len(dates)
        assign = np.zeros((N, D, 3), dtype=int)
        label_to_idx = {"AM": 0, "PM": 1, "NIGHT": 2}
        for i, nurse in enumerate(nurse_names):
            for j, label in enumerate(sched_df.loc[nurse]):
                if isinstance(label, str) and label.upper() in label_to_idx:
                    assign[i, j, label_to_idx[label.upper()]] = 1
        penalty = int(compute_total_penalty(assign, profiles, prefs, pd.to_datetime(req.start_date), fixed))
    except Exception:
        penalty = None

    used_rl = used_rl = bool(req.rl_assignment and len(req.rl_assignment) > 0)

    return ScheduleResponse(
        schedule=sched_df.to_dict(),
        summary=summ_df.to_dict(orient="records"),
        penalty=penalty,
        used_rl = used_rl,
    )

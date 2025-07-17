from datetime import date
from typing import List, Optional, Any
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator, Field
import datetime as dt
from scheduler.builder import build_schedule_model, validate_data, InputMismatchError
from utils.constants import *
from exceptions.custom_errors import NoFeasibleSolutionError
import traceback
import logging

app = FastAPI()

# Define data models
class NurseProfile(BaseModel):
    name: str
    title: str
    years_experience: int

    @model_validator(mode="before")
    @classmethod
    def extract_years_experience(cls, values: Any) -> Any:
        if "years_experience" not in values:
            for key in list(values.keys()):
                lowered = key.lower()
                if "year" in lowered or "experience" in lowered:
                    values["years_experience"] = values.pop(key)
                    break
        return values

class NursePreference(BaseModel):
    nurse: str
    date: date
    shift: str
    timestamp: Optional[dt.datetime] = None

class NurseTraining(BaseModel):
    nurse: str
    date: date
    training: str

class FixedAssignment(BaseModel):
    nurse: str
    day_index: int
    shift: str

class ScheduleRequest(BaseModel):
    start_date: date
    num_days: int
    shift_durations: List[int] = Field(default=SHIFT_DURATIONS)
    min_nurses_per_shift: int = Field(default=MIN_NURSES_PER_SHIFT)
    min_seniors_per_shift: int = Field(default=MIN_SENIORS_PER_SHIFT)
    max_weekly_hours: int = Field(default=MAX_WEEKLY_HOURS)
    preferred_weekly_hours: int = Field(default=PREFERRED_WEEKLY_HOURS)
    pref_weekly_hours_hard: bool = False
    min_acceptable_weekly_hours: int = Field(default=MIN_ACCEPTABLE_WEEKLY_HOURS)
    activate_am_cov: bool = True
    am_coverage_min_percent: int = Field(default=AM_COVERAGE_MIN_PERCENT)
    am_coverage_min_hard: bool = False
    am_coverage_relax_step: int = Field(default=AM_COVERAGE_RELAX_STEP)
    am_senior_min_percent: int = Field(default=AM_SENIOR_MIN_PERCENT)
    am_senior_min_hard: bool = False
    am_senior_relax_step: int = Field(default=AM_SENIOR_RELAX_STEP)
    weekend_rest: bool = True
    back_to_back_shift: bool = False
    use_sliding_window: bool = False
    shift_balance: bool = False
    fixed_assignments: Optional[List[FixedAssignment]] = None


def standardize_profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    From any df whose columns include (case-insensitive) "name", "title", and 
    "experience"/"year", pick out those three, rename to exactly 
    ["Name", "Title", "Years of experience"], uppercase Name, and return the result.
    """
    col_map = {col.lower().strip(): col for col in df.columns}

    def find_col(*candidates: str) -> str:
        # Try exact match first
        for c in candidates:
            if c in col_map:
                return col_map[c]
        # Then try substring match
        for lower, original in col_map.items():
            if any(c in lower for c in candidates):
                return original
        raise ValueError(f"No column matching {candidates} in {list(df.columns)}")

    name_src  = find_col("name")
    title_src = find_col("title")
    exp_src   = find_col("experience", "year")

    out = df[[name_src, title_src, exp_src]].copy()
    out.columns = ["Name", "Title", "Years of experience"]
    out["Name"] = out["Name"].astype(str).str.strip().str.upper()
    return out


@app.post("/schedule/generate/", response_model=dict)
async def generate_schedule(
    profiles: List[NurseProfile],
    preferences: List[NursePreference],
    training_shifts: List[NurseTraining],
    request: ScheduleRequest
):
    try:
        # Convert array inputs to DataFrames
        # profiles_df = (
        #     pd.DataFrame([p.model_dump() for p in profiles])
        #     .rename(columns={
        #         'name': 'Name',
        #         'title': 'Title',
        #         'years_experience': 'YearsExperience',
        #     })
        # )

        # Convert array inputs to raw DataFrame
        raw = pd.DataFrame([p.model_dump() for p in profiles])
        # Standardize it to exactly Name/Title/Years of experience
        profiles_df = standardize_profile_columns(raw)

        # Handle preferences
        if preferences:
            pref_dicts = [p.model_dump() for p in preferences]
            pref_df = pd.DataFrame(pref_dicts)
            if 'timestamp' in pref_df.columns:
                pref_df.sort_values(by='timestamp', ascending=False, inplace=True)
            prefs_df = (
                pref_df
                .drop_duplicates(subset=['nurse', 'date'])
                .pivot(index='nurse', columns='date', values='shift')
                .rename_axis(None, axis=0)
                .rename_axis(None, axis=1)
            )
            prefs_df.index.name = "Name"  # <-- Set index name here
        else:
            prefs_df = pd.DataFrame(index=profiles_df["Name"].str.strip())
            prefs_df.index.name = "Name"

        # Handle training shifts
        if training_shifts:
            training_df = (
                pd.DataFrame([t.model_dump() for t in training_shifts])
                .pivot(index='nurse', columns='date', values='training')
                .rename_axis(None, axis=0)
                .rename_axis(None, axis=1)
            )
            training_df.index.name = "Name"  # <-- Set index name here
        else:
            training_df = pd.DataFrame(index=profiles_df["Name"].str.strip())
            training_df.index.name = "Name"
        
        # Ensure indices are string type for validation
        if not prefs_df.empty and prefs_df.index.dtype != 'object':
            prefs_df.index = prefs_df.index.astype(str)
        if not training_df.empty and training_df.index.dtype != 'object':
            training_df.index = training_df.index.astype(str)
        
        # === Execute the original validation ===
        validate_data(profiles_df, prefs_df, "profiles", "preferences", False)
        validate_data(profiles_df, training_df, "profiles", "training shifts", False)
        
        # Handle fixed assignments
        fixed_assignments_dict = None
        if request.fixed_assignments:
            fixed_assignments_dict = {(fa.nurse, fa.day_index): fa.shift 
                                     for fa in request.fixed_assignments}
            
        # Convert hours→minutes so the CP‑SAT model sees minutes everywhere
        dur_minutes = [h * 60 for h in request.shift_durations]
        
        # Call scheduling function

        logging.info("=== API inputs for build_schedule_model ===")
        logging.info("profiles_df.shape:         %s", profiles_df.shape)
        logging.info("profiles_df.columns:       %s", profiles_df.columns.tolist())
        logging.info("prefs_df.shape:            %s, index sample: %r",
                     prefs_df.shape, list(prefs_df.index)[:5])
        logging.info("prefs_df.columns sample:   %r", list(prefs_df.columns)[:5])
        logging.info("training_df.shape:         %s, index sample: %r",
                     training_df.shape, list(training_df.index)[:5])
        logging.info("training_df.columns sample:%r", list(training_df.columns)[:5])
        logging.info("start_date:                %s", request.start_date)
        logging.info("num_days:                  %d", request.num_days)
        logging.info("shift_durations (hrs):     %r", request.shift_durations)
        logging.info("shift_durations (mins):    %r", dur_minutes)
        logging.info("min_nurses_per_shift:      %d", request.min_nurses_per_shift)
        logging.info("min_seniors_per_shift:     %d", request.min_seniors_per_shift)
        logging.info("max_weekly_hours (hrs):    %d", request.max_weekly_hours)
        logging.info("max_weekly_hours (mins):   %d", request.max_weekly_hours * 60)
        logging.info("preferred_weekly_hours (hrs):  %d", request.preferred_weekly_hours)
        logging.info("preferred_weekly_hours (mins): %d", request.preferred_weekly_hours * 60)
        logging.info("min_accept_weekly_hours (hrs):  %d", request.min_acceptable_weekly_hours)
        logging.info("min_accept_weekly_hours (mins): %d", request.min_acceptable_weekly_hours * 60)
        logging.info("activate_am_cov:           %s", request.activate_am_cov)
        logging.info("am_coverage_min_percent:   %d", request.am_coverage_min_percent)
        logging.info("am_coverage_min_hard:      %s", request.am_coverage_min_hard)
        logging.info("am_senior_min_percent:     %d", request.am_senior_min_percent)
        logging.info("am_senior_min_hard:        %s", request.am_senior_min_hard)
        logging.info("weekend_rest:              %s", request.weekend_rest)
        logging.info("back_to_back_shift:        %s", request.back_to_back_shift)
        logging.info("use_sliding_window:        %s", request.use_sliding_window)
        logging.info("shift_balance:             %s", request.shift_balance)
        logging.info("fixed_assignments count:   %s", len(fixed_assignments_dict or {}))

        schedule, summary, violations, metrics = build_schedule_model(
            profiles_df=profiles_df,
            preferences_df=prefs_df,
            training_shifts_df=training_df,
            start_date=pd.Timestamp(request.start_date),
            num_days=request.num_days,
            shift_durations=dur_minutes,
            min_nurses_per_shift=request.min_nurses_per_shift,
            min_seniors_per_shift=request.min_seniors_per_shift,
            max_weekly_hours=request.max_weekly_hours,
            preferred_weekly_hours=request.preferred_weekly_hours,
            pref_weekly_hours_hard=request.pref_weekly_hours_hard,
            min_acceptable_weekly_hours=request.min_acceptable_weekly_hours,
            activate_am_cov=request.activate_am_cov,
            am_coverage_min_percent=request.am_coverage_min_percent,
            am_coverage_min_hard=request.am_coverage_min_hard,
            am_coverage_relax_step=request.am_coverage_relax_step,
            am_senior_min_percent=request.am_senior_min_percent,
            am_senior_min_hard=request.am_senior_min_hard,
            am_senior_relax_step=request.am_senior_relax_step,
            weekend_rest=request.weekend_rest,
            back_to_back_shift=request.back_to_back_shift,
            use_sliding_window=request.use_sliding_window,
            shift_balance=request.shift_balance,
            fixed_assignments=fixed_assignments_dict
        )

        # Convert DataFrames to JSON-friendly format
        response = {
            "schedule": schedule.reset_index().to_dict(orient="records"),
            "summary": summary.reset_index().to_dict(orient="records"),
            "violations": violations,
            "metrics": metrics
        }
        
        return response
        
    except InputMismatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NoFeasibleSolutionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{tb}")
    

@app.get("/")
def root():
    return {"message": "Nurse Roster Scheduling API is running. Visit /docs for the Swagger UI."}

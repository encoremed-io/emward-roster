from datetime import date
from typing import List, Optional, Any
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, model_validator, Field, ConfigDict
import datetime as dt
from scheduler.builder import build_schedule_model
from utils.validate import validate_data
from utils.constants import *
from exceptions.custom_errors import *
import traceback
import logging

router = APIRouter(prefix="/schedule")

CUSTOM_ERRORS = {
    NoFeasibleSolutionError: 422,
    InvalidMCError: 400,
    ConsecutiveMCError: 400,
    InputMismatchError: 400
}

# Define data models
class NurseProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    title: str
    years_experience: int

    @model_validator(mode="before")
    @classmethod
    def extract_years_experience(cls, values: Any) -> Any:
        """
        Model validator to extract years_experience from other keys in the input data if not present.

        This validator is needed to handle the case where the column name for years of experience is not exactly "years_experience", e.g. "Years of Experience", "Year(s) Experience", etc.

        If the "years_experience" key is not present, the validator iterates over all the keys in the input data and checks if the key contains the words "year" or "experience". If a matching key is found, the value associated with that key is moved to the "years_experience" key, and the original key is removed from the input data.
        """
        if "years_experience" not in values:
            for key in list(values.keys()):
                lowered = key.lower()
                if "year" in lowered or "experience" in lowered:
                    values["years_experience"] = values.pop(key)
                    break
        return values

class NursePreference(BaseModel):
    model_config = ConfigDict(extra="allow")

    nurse: str
    date: date
    shift: str
    timestamp: Optional[dt.datetime] = None

class NurseTraining(BaseModel):
    model_config = ConfigDict(extra="allow")

    nurse: str
    date: date
    training: str

class FixedAssignment(BaseModel):
    model_config = ConfigDict(extra="allow")

    nurse: str
    date: date
    fixed: str

class ScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

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
    Standardize the column names of a nurse profile DataFrame to "Name", "Title", and "Years of experience".
    
    The function takes a DataFrame with columns representing nurse names, titles, and years of experience.
    It returns a new DataFrame with the same data, but with standardized column names.
    
    The function first builds a dictionary mapping lower-case, stripped column names to the original column names.
    It then uses this dictionary to find the columns in the DataFrame that match the candidates.
    If no exact match is found, it tries a substring match.
    If no match is found, it raises a ValueError.
    
    The function then copies the relevant columns into a new DataFrame and renames them.
    Finally, it strips and upper-cases the Name column and returns the new DataFrame.
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


@router.post("/schedule/generate/", response_model=dict)
async def generate_schedule(
    profiles: List[NurseProfile],
    preferences: List[NursePreference],
    training_shifts: List[NurseTraining],
    request: ScheduleRequest
):
    """
    Generate a schedule based on the given nurse profiles, shift preferences, and other parameters
    
    The API endpoint takes the following parameters:
    
    - `profiles`: List of `NurseProfile` objects, which contain the following information:
        - `name`: Name of the nurse
        - `title`: Title of the nurse (e.g. "Senior Nurse", "Junior Nurse")
        - `years_experience`: Number of years of experience the nurse has
    - `preferences`: List of `NursePreference` objects, which contain the following information:
        - `nurse`: Name of the nurse
        - `date`: Date of the shift
        - `shift`: Shift preference (e.g. "AM", "PM", "Night")
        - `timestamp`: Timestamp of the preference (optional, used for sorting preferences)
    - `training_shifts`: List of `NurseTraining` objects, which contain the following information:
        - `nurse`: Name of the nurse
        - `date`: Date of the training shift
        - `training`: Shift on training (e.g. "AM", "PM")
    - `request`: `ScheduleRequest` object, which contains the following information:
        - `start_date`: Start date of the schedule
        - `num_days`: Number of days in the schedule
        - `shift_durations`: List of shift durations in hours
        - `min_nurses_per_shift`: Minimum number of nurses per shift
        - `min_seniors_per_shift`: Minimum number of senior nurses per shift
        - `max_weekly_hours`: Maximum weekly hours for each nurse
        - `preferred_weekly_hours`: Preferred weekly hours for each nurse
        - `min_acceptable_weekly_hours`: Minimum acceptable weekly hours for each nurse
        - `activate_am_cov`: Whether to activate AM coverage constraints
        - `am_coverage_min_percent`: Minimum percentage of AM shifts that must be covered
        - `am_coverage_min_hard`: Whether the minimum percentage is a hard constraint
        - `am_coverage_relax_step`: Relaxation step for the minimum percentage
        - `am_senior_min_percent`: Minimum percentage of senior nurses that must be assigned to AM shifts
        - `am_senior_min_hard`: Whether the minimum percentage is a hard constraint
        - `am_senior_relax_step`: Relaxation step for the minimum percentage
        - `weekend_rest`: Whether to ensure that each nurse has a weekend rest
        - `back_to_back_shift`: Whether to prevent back-to-back shifts
        - `use_sliding_window`: Whether to use a sliding window for shift assignments
        - `shift_balance`: Whether to balance the number of shifts between nurses
        - `fixed_assignments`: List of fixed shift assignments (optional), with the following fields:
            - `nurse`: Name of the nurse
            - `date`: Date of the shift
            - `fixed`: Fixed declaration (e.g. "EL", "MC")
        
    The API endpoint returns a JSON object with the following keys:
    
    - `schedule`: List of shift assignments, where each assignment is a dictionary with the following keys:
        - `nurse`: Name of the nurse
        - `date`: Date of the shift
        - `shift`: Shift assignment (e.g. "AM", "PM", "Night")
    - `summary`: List of summary statistics, where each statistic is a dictionary with the following keys:
        - `metric`: Name of the metric (e.g. "Hours_Week1_Real", "Prefs_Unmet")
        - `value`: Value of the metric
    - `violations`: List of constraint violations, where each violation is a dictionary with the following keys:
        - `constraint`: Name of the constraint (e.g. "Low Hours Nurses", "Low Senior AM Days")
        - `value`: Value of the constraint
     - `metrics`: A dictionary containing evaluation metrics. Keys include "Preference Unmet" and "Fairness Gap", with each value representing the corresponding metric score.

    The API endpoint raises an HTTPException with a status code of 400 if the input is invalid and raises an HTTPException with a status code of 422 if no feasible solution is found.
    """
    try:
        # Convert array inputs to raw DataFrame
        raw = pd.DataFrame([p.model_dump() for p in profiles])
        # Standardize it to exactly Name/Title/Years of experience
        profiles_df = standardize_profile_columns(raw)

        # Handle preferences
        if preferences:
            # 1) build raw DataFrame
            pref_df = pd.DataFrame([p.model_dump() for p in preferences])

            if 'timestamp' in pref_df.columns:
                # 2) coerce to datetime & sort so earliest come first
                pref_df['timestamp'] = pd.to_datetime(pref_df['timestamp'])
                pref_df.sort_values(
                    by=['date','shift','timestamp'],
                    ascending=[True, True, True],
                    inplace=True
                )

            # 3) pack shift+ts into one column
            pref_df['cell'] = list(zip(pref_df['shift'], pref_df['timestamp']))

            # 4) drop any duplicate nurse+date, keeping that earliest row, then pivot
            prefs_df = (
                pref_df
                .drop_duplicates(subset=['nurse','date'], keep='first')
                .pivot(index='nurse', columns='date', values='cell')
                .rename_axis(None, axis=0)
                .rename_axis(None, axis=1)
            )

            # normalize to match profiles_df["Name"]
            prefs_df.index = prefs_df.index.str.strip().str.upper()
            prefs_df.index.name = "Name"

        else:
            prefs_df = pd.DataFrame(index=profiles_df["Name"].str.upper())
            prefs_df.index.name = "Name"

        # Handle training shifts
        if training_shifts:
            raw_train = pd.DataFrame([t.model_dump() for t in training_shifts])
            training_df = (
                raw_train
                .pivot(index='nurse', columns='date', values='training')
            )
            # normalize to match profiles_df["Name"]
            training_df.index = (
                training_df
                .index
                .astype(str)
                .str.strip()
                .str.upper()
            )
            training_df.index.name = "Name"
        else:
            # build an empty table with the same normalized index
            training_df = pd.DataFrame(index=profiles_df["Name"])
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
            # build a dict keyed by (nurse, date)
            fixed_assignments_dict = {(fa.nurse, fa.date): fa.fixed 
                                    for fa in request.fixed_assignments}
            # now convert dates to day‐indices
            fixed_idx_dict = {}
            for (nurse, dt), shift in fixed_assignments_dict.items():
                idx = (pd.Timestamp(dt) - pd.Timestamp(request.start_date)).days
                if idx < 0 or idx >= request.num_days:
                    raise HTTPException(400,
                        f"Fixed assignment for {nurse} on {dt} is outside the scheduling window")
                fixed_idx_dict[(nurse, idx)] = shift
            
        # Convert hours→minutes so the CP‑SAT model sees minutes everywhere
        dur_minutes = [h * 60 for h in request.shift_durations]
        
        # Call scheduling function

        # Log the inputs for debugging
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
            fixed_assignments=fixed_idx_dict if fixed_assignments_dict else None
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
    except tuple(CUSTOM_ERRORS) as e:
        raise HTTPException(status_code=CUSTOM_ERRORS[type(e)], detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{tb}")
    

@router.get("/")
def root():
    """
    Simple root endpoint to indicate the API is running.

    Returns a JSON response with a "message" key, containing a string
    indicating the API is running and providing a pointer to the Swagger UI
    documentation at /docs.
    """
    return {"message": "Nurse Roster Scheduling API is running. Visit /docs for the Swagger UI."}
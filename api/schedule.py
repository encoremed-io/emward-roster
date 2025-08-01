from schemas.schedule.generate import (
    NurseProfile,
    NursePreference,
    NurseTraining,
    PrevSchedule,
    ScheduleRequest,
)
from typing import List
import pandas as pd
from fastapi import APIRouter, HTTPException
from scheduler.builder import build_schedule_model
from utils.validate import validate_data
from utils.constants import *
from exceptions.custom_errors import *
import re
import traceback
from docs.schedule.roster import schedule_roster_description
from utils.helpers.schedule_roster import standardize_profile_columns
import logging

router = APIRouter(prefix="/schedule", tags=["Roster"])


# generate roster
@router.post(
    "/generate",
    response_model=dict,
    description=schedule_roster_description,
    summary="Generate Roster",
)
async def generate_schedule(
    profiles: List[NurseProfile],
    preferences: List[NursePreference],
    training_shifts: List[NurseTraining],
    previous_schedule: List[PrevSchedule],
    request: ScheduleRequest,
):
    try:
        # Convert array inputs to raw DataFrame
        raw = pd.DataFrame([p.model_dump() for p in profiles])
        # Standardize it to exactly Name/Title/Years of experience
        profiles_df = standardize_profile_columns(raw)

        # Handle preferences
        if preferences:
            # 1) build raw DataFrame
            pref_df = pd.DataFrame([p.model_dump() for p in preferences])

            if "timestamp" in pref_df.columns:
                # 2) coerce to datetime & sort so earliest come first
                pref_df["timestamp"] = pd.to_datetime(pref_df["timestamp"])
                pref_df.sort_values(
                    by=["date", "shift", "timestamp"],
                    ascending=[True, True, True],
                    inplace=True,
                )

            # 3) pack shift+ts into one column
            pref_df["cell"] = list(zip(pref_df["shift"], pref_df["timestamp"]))

            # 4) drop any duplicate nurse+date, keeping that earliest row, then pivot
            prefs_df = (
                pref_df.drop_duplicates(subset=["nurse", "date"], keep="first")
                .pivot(index="nurse", columns="date", values="cell")
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
            training_df = raw_train.pivot(
                index="nurse", columns="date", values="training"
            )
            # normalize to match profiles_df["Name"]
            training_df.index = training_df.index.astype(str).str.strip().str.upper()
            training_df.index.name = "Name"
        else:
            # build an empty table with the same normalized index
            training_df = pd.DataFrame(index=profiles_df["Name"])
            training_df.index.name = "Name"

        if previous_schedule:
            prev_sched_df = pd.DataFrame([p.model_dump() for p in previous_schedule])
            # set nurse index
            if "index" not in prev_sched_df.columns:
                raise HTTPException(
                    400, detail="Each prev_schedule row requires an 'index' field"
                )
            prev_sched_df = prev_sched_df.set_index("index")
            prev_sched_df.index = (
                prev_sched_df.index.astype(str).str.strip().str.upper()
            )
            prev_sched_df.index.name = "Name"
            # robust date‑column parsing
            converted = {}
            for col in prev_sched_df.columns:
                m = re.search(r"\d{4}-\d{2}-\d{2}", col)
                if not m:
                    raise HTTPException(
                        400,
                        detail=f"Could not parse date in prev-schedule column '{col}'",
                    )
                converted[col] = pd.to_datetime(m.group(0))
            prev_sched_df = prev_sched_df.rename(columns=converted)
            prev_schedule_df = prev_sched_df
        else:
            prev_schedule_df = pd.DataFrame(index=profiles_df["Name"])
            prev_schedule_df.index.name = "Name"

        # Ensure indices are string type for validation
        if not prefs_df.empty and prefs_df.index.dtype != "object":
            prefs_df.index = prefs_df.index.astype(str)
        if not training_df.empty and training_df.index.dtype != "object":
            training_df.index = training_df.index.astype(str)

        # === Execute the original validation ===
        validate_data(profiles_df, prefs_df, "profiles", "preferences", False)
        validate_data(profiles_df, training_df, "profiles", "training shifts", False)
        validate_data(
            profiles_df, prev_schedule_df, "profiles", "previous schedule", False
        )

        # Handle fixed assignments
        fixed_assignments_dict = None
        if request.fixed_assignments:
            # build a dict keyed by (nurse, date)
            fixed_assignments_dict = {
                (fa.nurse, fa.date): fa.fixed for fa in request.fixed_assignments
            }
            # now convert dates to day‐indices
            fixed_idx_dict = {}
            for (nurse, dt), shift in fixed_assignments_dict.items():
                idx = (pd.Timestamp(dt) - pd.Timestamp(request.start_date)).days
                if idx < 0 or idx >= request.num_days:
                    raise HTTPException(
                        400,
                        f"Fixed assignment for {nurse} on {dt} is outside the scheduling window",
                    )
                fixed_idx_dict[(nurse, idx)] = shift

        # Convert hours→minutes so the CP‑SAT model sees minutes everywhere
        dur_minutes = [h * 60 for h in request.shift_durations]

        # Call scheduling function

        # # Log the inputs for debugging
        # logging.info("=== API inputs for build_schedule_model ===")
        # logging.info("profiles_df.shape:         %s", profiles_df.shape)
        # logging.info("profiles_df.columns:       %s", profiles_df.columns.tolist())
        # logging.info("prefs_df.shape:            %s, index sample: %r",
        #              prefs_df.shape, list(prefs_df.index)[:5])
        # logging.info("prefs_df.columns sample:   %r", list(prefs_df.columns)[:5])
        # logging.info("training_df.shape:         %s, index sample: %r",
        #              training_df.shape, list(training_df.index)[:5])
        # logging.info("training_df.columns sample:%r", list(training_df.columns)[:5])
        # logging.info("prev_schedule_df.shape:    %s, index sample: %r",
        #              prev_schedule_df.shape, list(prev_schedule_df.index)[:5])
        # logging.info("start_date:                %s", request.start_date)
        # logging.info("num_days:                  %d", request.num_days)
        # logging.info("shift_durations (hrs):     %r", request.shift_durations)
        # logging.info("shift_durations (mins):    %r", dur_minutes)
        # logging.info("min_nurses_per_shift:      %d", request.min_nurses_per_shift)
        # logging.info("min_seniors_per_shift:     %d", request.min_seniors_per_shift)
        # logging.info("max_weekly_hours (hrs):    %d", request.max_weekly_hours)
        # logging.info("max_weekly_hours (mins):   %d", request.max_weekly_hours * 60)
        # logging.info("preferred_weekly_hours (hrs):  %d", request.preferred_weekly_hours)
        # logging.info("preferred_weekly_hours (mins): %d", request.preferred_weekly_hours * 60)
        # logging.info("min_accept_weekly_hours (hrs):  %d", request.min_acceptable_weekly_hours)
        # logging.info("min_accept_weekly_hours (mins): %d", request.min_acceptable_weekly_hours * 60)
        # logging.info("activate_am_cov:           %s", request.activate_am_cov)
        # logging.info("am_coverage_min_percent:   %d", request.am_coverage_min_percent)
        # logging.info("am_coverage_min_hard:      %s", request.am_coverage_min_hard)
        # logging.info("am_senior_min_percent:     %d", request.am_senior_min_percent)
        # logging.info("am_senior_min_hard:        %s", request.am_senior_min_hard)
        # logging.info("weekend_rest:              %s", request.weekend_rest)
        # logging.info("back_to_back_shift:        %s", request.back_to_back_shift)
        # logging.info("use_sliding_window:        %s", request.use_sliding_window)
        # logging.info("shift_balance:             %s", request.shift_balance)
        # logging.info("fixed_assignments count:   %s", len(fixed_assignments_dict or {}))

        schedule, summary, violations, metrics = build_schedule_model(
            profiles_df=profiles_df,
            preferences_df=prefs_df,
            training_shifts_df=training_df,
            prev_schedule_df=prev_schedule_df,
            start_date=pd.Timestamp(request.start_date),
            num_days=request.num_days,
            shift_durations=dur_minutes,
            min_nurses_per_shift=request.min_nurses_per_shift,
            min_seniors_per_shift=request.min_seniors_per_shift,
            max_weekly_hours=request.max_weekly_hours,
            preferred_weekly_hours=request.preferred_weekly_hours,
            pref_weekly_hours_hard=request.pref_weekly_hours_hard,
            min_acceptable_weekly_hours=request.min_acceptable_weekly_hours,
            min_weekly_rest=request.min_weekly_rest,
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
            priority_setting=request.priority_setting,
            fixed_assignments=fixed_idx_dict if fixed_assignments_dict else None,
            shift_details=request.shift_details,
        )

        # Convert DataFrames to JSON-friendly format
        response = {
            "schedule": schedule.reset_index().to_dict(orient="records"),
            "summary": summary.reset_index().to_dict(orient="records"),
            "violations": violations,
            "metrics": metrics,
        }

        return response

    except tuple(CUSTOM_ERRORS) as e:
        raise HTTPException(status_code=CUSTOM_ERRORS[type(e)], detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{tb}")

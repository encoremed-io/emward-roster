from schemas.schedule.generate import (
    NurseProfile,
    NursePreference,
    NurseTraining,
    PrevSchedule,
    ScheduleRequest,
    Shifts,
)
from typing import List
import pandas as pd
from fastapi import APIRouter, HTTPException
from scheduler.builder import build_schedule_model
from utils.validate import validate_data
from utils.constants import *
from exceptions.custom_errors import *
import traceback
from docs.schedule.roster import schedule_roster_description
from utils.helpers.schedule_roster import standardize_profile_columns, normalize_names
from utils.shift_utils import parse_duration

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
    trainingShifts: List[NurseTraining],
    previousSchedule: List[PrevSchedule],
    shifts: List[Shifts],
    request: ScheduleRequest,
):
    try:
        # Convert array inputs to raw DataFrame
        raw = pd.DataFrame([p.model_dump() for p in profiles])
        # Standardize it to exactly Name/Title/Years of experience
        profiles_df = standardize_profile_columns(raw)
        profiles_df.columns = profiles_df.columns.str.lower()

        if "id" in profiles_df.columns:
            profiles_df["id"] = profiles_df["id"].astype(str)

        # Handle preferences
        pref_df = pd.DataFrame()

        if preferences:
            pref_df = pd.DataFrame([p.model_dump() for p in preferences])

            # Ensure id and shift are strings
            if "id" in pref_df.columns:
                pref_df["id"] = pref_df["id"].astype(str)
            if "shiftId" in pref_df.columns:
                pref_df["shiftId"] = pref_df["shiftId"].astype(str)

            # Convert timestamp to datetime and sort
            if "timestamp" in pref_df.columns:
                pref_df["timestamp"] = pd.to_datetime(
                    pref_df["timestamp"], errors="coerce"
                )
                pref_df.sort_values(
                    by=["date", "shiftId", "timestamp"],
                    ascending=[True, True, True],
                    inplace=True,
                )

            # Pack shift+ts into one column
            pref_df["cell"] = list(zip(pref_df["shiftId"], pref_df["timestamp"]))

            # Pivot with nurse id as index
            prefs_df = pref_df.drop_duplicates(
                subset=["id", "date"], keep="first"
            ).pivot(index="id", columns="date", values="cell")

            # Ensure index is str (id, not nurse name)
            prefs_df.index = prefs_df.index.astype(str)
            prefs_df.index.name = "id"

            # Add nurse name for readability
            id_to_name = pref_df.set_index("id")["nurse"].to_dict()
            prefs_df.insert(0, "name", prefs_df.index.map(id_to_name))
        else:
            prefs_df = pd.DataFrame(index=profiles_df["id"].astype(str))
            prefs_df.insert(0, "name", profiles_df["name"])

        # Handle training shifts
        if trainingShifts:
            raw_train = pd.DataFrame([t.model_dump() for t in trainingShifts])

            # Normalize nurse names if column exists
            if "nurse" in raw_train.columns:
                raw_train["nurse"] = normalize_names(raw_train["nurse"])

            # Normalize IDs if column exists
            if "id" in raw_train.columns:
                raw_train["id"] = raw_train["id"].astype(str)

            if "shiftId" not in raw_train.columns:
                raw_train["shiftId"] = None

            # Pivot with id as index if available, else fallback to nurse
            if "id" in raw_train.columns:

                training_df = raw_train.drop_duplicates(
                    subset=["id", "date"], keep="first"
                )[["id", "nurse", "date", "shiftId"]]

                training_df = training_df.rename(columns={"nurse": "name"})

            else:
                training_df = raw_train.drop_duplicates(
                    subset=["nurse", "date"], keep="first"
                )[["nurse", "date", "shiftId"]].rename(columns={"nurse": "name"})

        else:
            # Empty training_df with same structure as prefs_df
            training_df = pd.DataFrame(index=prefs_df.index)
            training_df.insert(0, "name", prefs_df["name"])

        # Previous schedule
        if previousSchedule:
            records = []
            for nurse_entry in previousSchedule:
                for item in nurse_entry.schedule:
                    records.append(
                        {
                            "id": nurse_entry.id,
                            "nurse": nurse_entry.nurse,
                            "date": pd.to_datetime(item.date),
                            "shiftId": item.shiftId,
                            "shift": item.shift,
                        }
                    )

            prev_sched_df = pd.DataFrame(records)

            if prev_sched_df.empty:
                prev_schedule_df = pd.DataFrame(index=profiles_df["id"].astype(str))
            else:
                # Pivot: id as row, date as column, shift name as value
                prev_schedule_df = prev_sched_df.pivot_table(
                    index="id", columns="date", values="shiftId", aggfunc="first"
                )

                # ensure index is string for consistency
                prev_schedule_df.index = prev_schedule_df.index.astype(str)
                prev_schedule_df.index.name = "id"
        else:
            prev_schedule_df = pd.DataFrame(index=profiles_df["id"].astype(str))
            prev_schedule_df.index.name = "id"

        validate_data(
            profiles_df,
            pref_df,
            "profiles",
            "preferences",
            False,
        )

        validate_data(profiles_df, training_df, "profiles", "training shifts", False)
        validate_data(
            profiles_df, prev_schedule_df, "profiles", "previous schedule", False
        )

        # Handle fixed assignments
        fixed_assignments_dict = None
        if request.fixedAssignments:
            # build a dict keyed by (nurse, date)
            fixed_assignments_dict = {
                (fa.nurse, fa.date): fa.fixed for fa in request.fixedAssignments
            }
            # now convert dates to day‐indices
            fixed_idx_dict = {}
            for (nurse, dt), shift in fixed_assignments_dict.items():
                idx = (pd.Timestamp(dt) - pd.Timestamp(request.startDate)).days
                if idx < 0 or idx >= request.numDays:
                    raise HTTPException(
                        400,
                        f"Fixed assignment for {nurse} on {dt} is outside the scheduling window",
                    )
                fixed_idx_dict[(nurse, idx)] = shift

        # Convert hours→minutes so the CP‑SAT model sees minutes everywhere
        shiftDurations = [parse_duration(s.duration) for s in shifts]
        dur_minutes = [h * 60 for h in shiftDurations]

        # Call scheduling function
        schedule, summary, violations, metrics = build_schedule_model(
            profiles_df=profiles_df,
            preferences_df=prefs_df,
            training_shifts_df=training_df,
            prev_schedule_df=prev_schedule_df,
            start_date=pd.Timestamp(request.startDate),
            num_days=request.numDays,
            shift_durations=dur_minutes,
            max_weekly_hours=request.maxWeeklyHours,
            preferred_weekly_hours=request.preferredWeeklyHours,
            pref_weekly_hours_hard=request.prefWeeklyHoursHard,
            min_acceptable_weekly_hours=request.minWeeklyHours,
            min_weekly_rest=request.minWeeklyRest,
            weekend_rest=request.weekendRest,
            back_to_back_shift=request.backToBackShift,
            use_sliding_window=request.useSlidingWindow,
            shift_balance=request.shiftBalance,
            priority_setting=request.prioritySetting,
            shift_details=request.shiftDetails,
            shifts=shifts,
            allow_double_shift=request.allowDoubleShift,
            # Uncomment if you want to use AM coverage constraints
            # fixed_assignments=fixed_idx_dict if fixed_assignments_dict else None,
            # activate_am_cov=request.activate_am_cov,
            # am_coverage_min_percent=request.am_coverage_min_percent,
            # am_coverage_min_hard=request.am_coverage_min_hard,
            # am_coverage_relax_step=request.am_coverage_relax_step,
            # am_senior_min_percent=request.am_senior_min_percent,
            # am_senior_min_hard=request.am_senior_min_hard,
            # am_senior_relax_step=request.am_senior_relax_step,
        )

        # Normalize UUIDs to lowercase for mapping
        id_to_name = dict(
            zip(profiles_df["id"].astype(str).str.lower(), profiles_df["name"])
        )

        # ---- schedule ----
        sched_df = schedule.reset_index().rename(columns={"index": "id"})
        sched_df["id"] = sched_df["id"].astype(str).str.lower()

        sched_df.insert(1, "name", sched_df["id"].map(id_to_name).fillna("UNKNOWN"))

        # ---- summary ----
        sum_df = summary.reset_index()

        # If "Nurse" column exists, rename it
        if "Nurse" in sum_df.columns:
            sum_df = sum_df.rename(columns={"Nurse": "id"})
        elif "index" in sum_df.columns and "id" not in sum_df.columns:
            sum_df = sum_df.rename(columns={"index": "id"})
        elif "name" in sum_df.columns and "id" not in sum_df.columns:
            # shift UUIDs from 'name' → 'id'
            sum_df = sum_df.rename(columns={"name": "id"})

        # Normalize IDs for mapping
        sum_df["id"] = sum_df["id"].astype(str).str.lower()

        # Add proper human name column
        sum_df.insert(1, "name", sum_df["id"].map(id_to_name).fillna("UNKNOWN"))

        # ==== final response ====
        response = {
            "schedule": sched_df.to_dict(orient="records"),
            "summary": sum_df.to_dict(orient="records"),
            "violations": violations,
            "metrics": metrics,
        }
        return response

    except tuple(CUSTOM_ERRORS) as e:
        raise HTTPException(status_code=CUSTOM_ERRORS[type(e)], detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{tb}")

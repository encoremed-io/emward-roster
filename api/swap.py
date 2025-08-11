from fastapi import APIRouter, Body
import pandas as pd
from utils.helpers.swap_suggestions import (
    parse_date,
    generate_warning,
    preprocess_nurse,
)
from utils.helpers.swap_suggestions_onnx import run_model_on
from datetime import datetime, timedelta
from typing import Dict, List
from docs.swap.suggestions import swap_suggestions_description
from schemas.swap.suggestions import SwapSuggestionRequest, SwapCandidate
import sys
from pprint import pprint

router = APIRouter(prefix="/swap", tags=["Suggestions"])


# get suggestions for swapping nurses shift
@router.post(
    "/suggestions",
    description=swap_suggestions_description,
    summary="Suggest Swap Candidates",
)
def suggest_swap(data: SwapSuggestionRequest = Body(...)):
    # pprint(data, sort_dicts=False, width=100)

    # data
    target_nurses = data.targetNurseId
    settings = data.settings
    shift_duration = settings.shiftDurations
    max_hours = settings.maxWeeklyHours
    min_seniors = settings.minSeniorsPerShift
    back_to_back = settings.backToBackShift

    print(f"Settings: {settings}")
    print(f"Target Nurses: {target_nurses}")
    # assign roster
    roster = data.roster

    results = []

    # loop through the nurses who are taking leaves
    for target_entry in target_nurses:
        nurseId = target_entry.nurseId
        assigned_nurse = next((n for n in roster if n.nurseId == nurseId), None)

        # return error if nurse not found
        if not assigned_nurse:
            results.append(
                {"originalNurse": nurseId, "error": "Nurse not found in roster."}
            )
            continue

        # get nurse role
        targetSenior = assigned_nurse.isSenior

        # loop through the target shifts
        for shift in target_entry.targetShift:
            date = shift.date

            # loop through the shift id
            for shiftType in shift.shiftTypeId:
                # Get nurses already assigned to the same date/shift
                same_day_assignments = [
                    n
                    for n in roster
                    for s in n.shifts
                    if s.date == date and s.shiftTypeId == shiftType
                ]

                # get current seniors
                current_seniors = sum(1 for n in same_day_assignments if n.isSenior)

                # replace with senior if the min seniors requirement is not met
                must_replace_with_senior = (
                    targetSenior and current_seniors < min_seniors
                )

                # set rule rigidity
                strict, fallback = [], []

                # direct swap option
                directSwap = None

                # loop through roster
                for nurse in roster:

                    # skip the nurse that is applying leave
                    if nurse.nurseId == nurseId:
                        continue

                    # check for swap suggestion
                    for shift in nurse.shifts:
                        if shift.date == date:
                            continue  # Skip if it's the same day as the target

                        # Check if nurse is already working on the target date
                        if any(s.date == date for s in nurse.shifts):
                            print(f"  ❌ Already has shift on {date}")
                            continue

                        # Enforce back-to-back restriction
                        if not back_to_back:
                            target_dt = datetime.strptime(date, "%Y-%m-%d")
                            if any(
                                abs(
                                    (
                                        target_dt
                                        - datetime.strptime(s.date, "%Y-%m-%d")
                                    ).days
                                )
                                == 1
                                for s in nurse.shifts
                            ):
                                print(
                                    f"  ❌ Back-to-back conflict with shift on adjacent day"
                                )
                                continue

                        # Optional: Check seniority if required
                        if must_replace_with_senior and not nurse.isSenior:
                            print(f"  ❌ Not senior and senior required")
                            continue

                        # Passed all checks — suggest this as a direct swap
                        directSwap = {
                            "nurseId": nurse.nurseId,
                            "swapFrom": shift,  # original shift (any type)
                            "swapTo": {
                                "date": date,
                                "shiftTypeId": shiftType,  # target shift needing replacement
                            },
                            "note": (
                                "Cross-shift swap allowed ({} → {}).".format(
                                    shift.shiftTypeId, shiftType
                                )
                                if shift.shiftTypeId != shiftType
                                else "Same-shift direct swap"
                            ),
                        }
                        break

                    if directSwap:
                        break

                    # check if the replacement require senior
                    if must_replace_with_senior and not nurse.isSenior:
                        continue

                    # check if nurse has conflict on the same date shift
                    has_conflict = any(
                        s.date == date and s.shiftTypeId == shiftType
                        for s in nurse.shifts
                    )
                    if has_conflict:
                        continue

                    # check for back to back shift
                    if not back_to_back:
                        shift_dates = [s.date for s in nurse.shifts]
                        dt = datetime.strptime(date, "%Y-%m-%d")
                        if any(
                            abs((dt - datetime.strptime(d, "%Y-%m-%d")).days) == 1
                            for d in shift_dates
                        ):
                            continue

                    # get the total hours the nurse works
                    weekly_hours = len(nurse.shifts) * shift_duration

                    # compile the shift details of the nurse
                    processed = preprocess_nurse(
                        nurse, datetime.strptime(date, "%Y-%m-%d"), settings
                    )

                    # potential replacement candidate
                    entry = SwapCandidate(
                        nurseId=nurse.nurseId,
                        isSenior=nurse.isSenior,
                        currentHours=weekly_hours,
                        violatesMaxHours=weekly_hours + shift_duration > max_hours,
                        message=generate_warning(processed, settings),
                    )

                    # append to fallback list if rules being violated
                    if entry.violatesMaxHours:
                        fallback.append(entry)
                    else:
                        strict.append(entry)

                candidates = strict if strict else fallback
                filterLevel = (
                    "strict" if strict else "fallback" if fallback else "none_available"
                )

                # append candidates to results
                results.append(
                    {
                        "originalNurse": nurseId,
                        "replacementFor": {"date": date, "shiftTypeId": shiftType},
                        "filterLevel": filterLevel,
                        "topCandidates": candidates,
                        "directSwapCandidate": directSwap,
                    }
                )

    # return all candidates suggestions
    return {"results": results}

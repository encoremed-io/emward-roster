from fastapi import APIRouter, Body
from datetime import datetime
from docs.swap.suggestions import swap_suggestions_description
from schemas.swap.suggestions import SwapSuggestionRequest, SwapCandidate
from utils.helpers.swap_suggestions import (
    generate_warning,
    preprocess_nurse,
)
from utils.shift_utils import parse_duration

router = APIRouter(prefix="/swap", tags=["Suggestions"])


@router.post(
    "/suggestions",
    description=swap_suggestions_description,
    summary="Suggest Swap Candidates",
)
def suggest_swap(data: SwapSuggestionRequest = Body(...)):
    # Build lookup map of shiftId → hours
    shift_hours = {s.id: parse_duration(s.duration) for s in data.shifts}

    max_hours = data.settings.maxWeeklyHours
    min_seniors = data.settings.minSeniorsPerShift
    back_to_back = data.settings.backToBackShift

    results = []

    # loop through the nurses who are taking leaves
    for target_entry in data.targetNurseId:
        nurseId = target_entry.nurseId
        assigned_nurse = next((n for n in data.roster if n.nurseId == nurseId), None)

        # return error if nurse not found
        if not assigned_nurse:
            results.append(
                {"originalNurse": nurseId, "error": "Nurse not found in roster."}
            )
            continue

        # get nurse role
        targetSenior = target_entry.isSenior

        # loop through the target shifts
        for target_shift in target_entry.targetShift:
            date = target_shift.date

            # loop through each shiftTypeId in the list
            for shiftType in target_shift.shiftIds:
                # Nurses already assigned to this date/shift
                same_day_assignments = [
                    n
                    for n in data.roster
                    for s in n.shifts
                    if s.date == date and shiftType in s.shiftIds
                ]

                # count seniors already assigned
                current_seniors = sum(1 for n in same_day_assignments if n.isSenior)

                # require senior replacement if not enough seniors
                must_replace_with_senior = (
                    targetSenior and current_seniors < min_seniors
                )

                strict, fallback = [], []
                directSwap = None

                # loop through roster
                for nurse in data.roster:
                    if nurse.nurseId == nurseId:
                        continue  # skip the nurse on leave

                    for s in nurse.shifts:
                        if s.date == date:
                            continue  # Skip if it's the same day as the target

                        # nurse already working that day?
                        if any(date == ns.date for ns in nurse.shifts):
                            continue

                        # check back-to-back restriction
                        if not back_to_back:
                            target_dt = datetime.strptime(date, "%Y-%m-%d")
                            if any(
                                abs(
                                    (
                                        target_dt
                                        - datetime.strptime(ns.date, "%Y-%m-%d")
                                    ).days
                                )
                                == 1
                                for ns in nurse.shifts
                            ):
                                continue

                        # seniority required?
                        if must_replace_with_senior and not nurse.isSenior:
                            continue

                        # Passed checks — direct swap candidate
                        directSwap = {
                            "nurseId": nurse.nurseId,
                            "swapFrom": s,
                            "swapTo": {
                                "date": date,
                                "shiftTypeId": shiftType,
                            },
                            "note": (
                                f"Cross-shift swap allowed ({s.shiftIds} → {shiftType})."
                                if shiftType not in s.shiftIds
                                else "Same-shift direct swap"
                            ),
                        }
                        break

                    if directSwap:
                        break

                    # check seniority again for non-direct swaps
                    if must_replace_with_senior and not nurse.isSenior:
                        continue

                    # conflict: already working this shift
                    if any(
                        ns.date == date and shiftType in ns.shiftIds
                        for ns in nurse.shifts
                    ):
                        continue

                    # back-to-back restriction
                    if not back_to_back:
                        dt = datetime.strptime(date, "%Y-%m-%d")
                        shift_dates = [ns.date for ns in nurse.shifts]
                        if any(
                            abs((dt - datetime.strptime(d, "%Y-%m-%d")).days) == 1
                            for d in shift_dates
                        ):
                            continue

                    # calculate nurse hours
                    weekly_hours = sum(
                        shift_hours.get(str(shiftId), 0)
                        for ns in nurse.shifts
                        for shiftId in ns.shiftIds
                    )
                    candidate_shift_hours = shift_hours.get(shiftType, 0)

                    processed = preprocess_nurse(
                        nurse, datetime.strptime(date, "%Y-%m-%d"), data.settings
                    )

                    entry = SwapCandidate(
                        nurseId=nurse.nurseId,
                        isSenior=nurse.isSenior,
                        currentHours=weekly_hours,
                        violatesMaxHours=weekly_hours + candidate_shift_hours
                        > max_hours,
                        message=generate_warning(processed, data.settings),
                    )

                    if entry.violatesMaxHours:
                        fallback.append(entry)
                    else:
                        strict.append(entry)

                candidates = strict if strict else fallback
                filterLevel = (
                    "strict" if strict else "fallback" if fallback else "none_available"
                )

                results.append(
                    {
                        "originalNurse": nurseId,
                        "replacementFor": {"date": date, "shiftTypeId": shiftType},
                        "filterLevel": filterLevel,
                        "topCandidates": candidates,
                        "directSwapCandidate": directSwap,
                    }
                )

    return {"results": results}

from fastapi import APIRouter, Body
from datetime import datetime
from docs.swap.suggestions import swap_suggestions_description
from schemas.swap.suggestions import SwapSuggestionRequest, SwapCandidate
from ortools.linear_solver import pywraplp
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
    shift_hours = {s.id: parse_duration(s.duration) for s in data.shifts}
    shift_names = {s.id: s.name for s in data.shifts}
    results = []

    for target_entry in data.targetNurseId:
        nurseId = target_entry.nurseId
        assigned_nurse = next((n for n in data.roster if n.nurseId == nurseId), None)

        if not assigned_nurse:
            results.append(
                {"originalNurse": nurseId, "error": "Nurse not found in roster."}
            )
            continue

        targetSenior = target_entry.isSenior

        for target_shift in target_entry.targetShift:
            date = target_shift.date

            for shiftType in target_shift.shiftIds:
                # Count seniors already assigned
                same_day_assignments = [
                    n
                    for n in data.roster
                    for s in n.shifts
                    if s.date == date and shiftType in s.shiftIds
                ]

                current_total = len(same_day_assignments)
                current_seniors = sum(1 for n in same_day_assignments if n.isSenior)

                shift_config = next(
                    (s for s in data.shifts if str(s.id) == str(shiftType)), None
                )

                # get min nurses and seniors by shifts
                min_nurses_required = (
                    shift_config.minNursesPerShift if shift_config else 0
                )
                min_seniors_required = (
                    shift_config.minSeniorsPerShift if shift_config else 0
                )

                # enforce per-shift minimums
                must_replace_with_nurse = current_total < min_nurses_required
                must_replace_with_senior = current_seniors < min_seniors_required

                # Use solver with direct swap candidates
                candidates, direct_swap = optimize_candidates(
                    data,
                    date,
                    shiftType,
                    shift_hours,
                    nurseId,
                    must_replace_with_nurse,
                    must_replace_with_senior,
                    shift_names,
                )

                results.append(
                    {
                        "originalNurse": nurseId,
                        "replacementFor": {"date": date, "shiftId": shiftType},
                        "filterLevel": "optimized",
                        "topCandidates": candidates[:3],  # best 3 candidates
                        "directSwapCandidate": direct_swap,
                    }
                )

    return {"results": results}


def optimize_candidates(
    data,
    target_date,
    shiftType,
    shift_hours,
    exclude_nurse,
    must_replace_with_nurse,
    must_replace_with_senior,
    shift_names,
):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("Solver not available")

    nurses = [n for n in data.roster if n.nurseId != exclude_nurse]
    x = {n.nurseId: solver.BoolVar(n.nurseId) for n in nurses}

    objective = solver.Objective()

    best_direct_swap = None
    scored = []

    for nurse in nurses:
        penalty = 0
        weekly_hours = sum(
            shift_hours.get(str(shiftId), 0)
            for s in nurse.shifts
            for shiftId in s.shiftIds
        )

        candidate_shift_hours = shift_hours.get(shiftType, 0)

        # direct Swap: if nurse has another shift on the same date → priority 0
        direct_shift = next((s for s in nurse.shifts if s.date == target_date), None)
        if direct_shift:
            penalty = 0

            swap_from_names = ", ".join(
                shift_names.get(sid, sid) for sid in direct_shift.shiftIds
            )
            swap_to_name = shift_names.get(shiftType, shiftType)

            best_direct_swap = {
                "nurseId": nurse.nurseId,
                "swapFrom": {
                    "date": direct_shift.date,
                    "shiftIds": direct_shift.shiftIds,
                },
                "swapTo": {"date": target_date, "shiftId": shiftType},
                "note": (
                    f"Cross-shift swap allowed ({swap_from_names} → {swap_to_name})."
                    if shiftType not in direct_shift.shiftIds
                    else f"Same-shift direct swap ({swap_to_name})"
                ),
            }
        else:
            # --- Rule penalties ---
            if any(
                s.date == target_date and shiftType in s.shiftIds for s in nurse.shifts
            ):
                penalty += 1000  # already working same shift

            if must_replace_with_nurse:
                # strongly penalize skipping this shift if already understaffed
                penalty -= 200

            if must_replace_with_senior and not nurse.isSenior:
                penalty += 500  # must be senior

            # min weekly hours constraint
            if weekly_hours + candidate_shift_hours < data.settings.minWeeklyHours:
                penalty += (
                    data.settings.minWeeklyHours
                    - (weekly_hours + candidate_shift_hours)
                ) * 20

            # max weekly hours constraint
            if weekly_hours > data.settings.maxWeeklyHours:
                penalty += (weekly_hours - data.settings.maxWeeklyHours) * 10

            # preferred weekly hours constraint
            penalty += abs(weekly_hours - data.settings.preferredWeeklyHours)

        objective.SetCoefficient(x[nurse.nurseId], penalty)

        processed = preprocess_nurse(
            nurse, datetime.strptime(target_date, "%Y-%m-%d"), data.settings
        )
        scored.append(
            (
                penalty,
                SwapCandidate(
                    nurseId=nurse.nurseId,
                    isSenior=nurse.isSenior,
                    currentHours=weekly_hours,
                    violatesMaxHours=weekly_hours > data.settings.maxWeeklyHours,
                    message=generate_warning(processed, data.settings),
                ),
            )
        )

    objective.SetMinimization()
    solver.Solve()

    # sort by penalty score
    scored.sort(key=lambda x: x[0])
    candidates = [entry for _, entry in scored]

    return candidates, best_direct_swap

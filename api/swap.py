from fastapi import APIRouter, Body
from datetime import datetime, timedelta
from docs.swap.suggestions import swap_suggestions_description
from schemas.swap.suggestions import SwapSuggestionRequest, SwapCandidate
from ortools.linear_solver import pywraplp
from utils.helpers.swap_suggestions import (
    generate_warning,
    preprocess_nurse,
    build_back_to_back_rules,
    parse_date,
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

                current_seniors = sum(1 for n in same_day_assignments if n.isSenior)

                shift_config = next(
                    (s for s in data.shifts if str(s.id) == str(shiftType)), None
                )

                # get min nurses and seniors by shifts
                min_seniors_required = (
                    shift_config.minSeniorsPerShift if shift_config else 0
                )

                # enforce per-shift minimums
                must_replace_with_senior = current_seniors < min_seniors_required

                # Use solver with direct swap candidates
                result = optimize_candidates(
                    data,
                    date,
                    shiftType,
                    shift_hours,
                    nurseId,
                    must_replace_with_senior,
                    shift_names,
                )

                results.append(
                    {
                        "originalNurse": nurseId,
                        "replacementFor": {"date": date, "shiftId": shiftType},
                        "filterLevel": "optimized",
                        "topCandidates": result["candidates"][:3],
                        "directSwapCandidate": result["bestDirectSwap"],
                    }
                )

    return {"results": results}


def optimize_candidates(
    data,
    target_date,
    shiftType,
    shift_hours,
    exclude_nurse,
    must_replace_with_senior,
    shift_names,
):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("Solver not available")

    # helper
    def count_nurses_on_shift(roster, date, shift_id):
        return sum(
            1
            for nurse in roster
            for s in nurse.shifts
            if s.date == date and shift_id in s.shiftIds
        )

    nurses = [n for n in data.roster if n.nurseId != exclude_nurse]
    x = {n.nurseId: solver.BoolVar(n.nurseId) for n in nurses}

    objective = solver.Objective()

    best_direct_swap = None
    scored = []

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    week_start = target_dt - timedelta(days=6)
    back_to_back_rules = build_back_to_back_rules(data.shifts)

    # enforce min nurses per shift (hard constraint)
    # current_count = count_nurses_on_shift(data.roster, target_date, shiftType)
    # min_required = getattr(data.settings, "minNursesPerShift", 1)

    # if current_count <= min_required:
    #     raise ValueError(
    #         f"Cannot swap: shift {shiftType} on {target_date} "
    #         f"already at minimum coverage ({current_count} ≤ {min_required})."
    #     )

    for nurse in nurses:
        # calculate hours in rolling 7-day window
        weekly_hours = sum(
            shift_hours.get(str(shiftId), 0)
            for s in nurse.shifts
            if week_start <= parse_date(s.date) <= target_dt
            for shiftId in s.shiftIds
        )

        worked_days = {s.date for s in nurse.shifts}
        weekly_rest_days = 7 - len(worked_days)
        candidate_shift_hours = shift_hours.get(shiftType, 0)

        # direct swap handling
        direct_shift = next((s for s in nurse.shifts if s.date == target_date), None)
        if direct_shift and shiftType in direct_shift.shiftIds:
            # if current_count <= min_required:
            #     continue

            # if valid, record direct swap
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

            # direct swaps always lowest penalty
            scored.append(
                (
                    -1,
                    SwapCandidate(
                        nurseId=nurse.nurseId,
                        isSenior=nurse.isSenior,
                        currentHours=weekly_hours,
                        violatesMaxHours=weekly_hours > data.settings.maxWeeklyHours,
                        messages=[best_direct_swap["note"]],
                        penaltyScore=-1,
                    ),
                )
            )
            continue  # skip normal warning calc

        # preprocess
        processed = preprocess_nurse(nurse, target_dt, data.settings)

        # run warning generator
        warning_messages, warn_penalty = generate_warning(
            processed,
            data.settings,
            weekly_hours=weekly_hours,
            weekly_rest_days=weekly_rest_days,
            candidate_shift_hours=candidate_shift_hours,
            must_replace_with_senior=must_replace_with_senior,
            back_to_back_rules=back_to_back_rules,
        )

        penalty = warn_penalty

        # attach penalty to solver variable
        objective.SetCoefficient(x[nurse.nurseId], penalty)

        scored.append(
            (
                penalty,
                SwapCandidate(
                    nurseId=nurse.nurseId,
                    isSenior=nurse.isSenior,
                    currentHours=weekly_hours,
                    violatesMaxHours=weekly_hours > data.settings.maxWeeklyHours,
                    messages=warning_messages if warning_messages != "OK" else [],
                    penaltyScore=penalty,
                ),
            )
        )

    # if you want solver to pick exactly one candidate:
    solver.Add(sum(x.values()) == 1)
    objective.SetMinimization()
    solver.Solve()

    scored.sort(key=lambda x: x[0])
    candidates = [entry for _, entry in scored]

    return {"candidates": candidates, "bestDirectSwap": best_direct_swap}

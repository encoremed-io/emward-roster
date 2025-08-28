from datetime import datetime, timedelta
from schemas.swap.suggestions import SwapCandidateFeatures
import numpy as np


# format date
def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")


# format time
def parse_time(tstr):
    """Convert 'HHMM' string to minutes since midnight."""
    return int(tstr[:2]) * 60 + int(tstr[2:])


# check for weekend
def is_weekend(date_str: str) -> tuple[bool, int | None]:
    """
    Return (is_weekend, weekday_index).
    weekday_index: 5 = Saturday, 6 = Sunday, None otherwise.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.weekday() in (5, 6):
        return True, dt.weekday()
    return False, None


# validation check with warning messages
def generate_warning(
    nurse: SwapCandidateFeatures,
    settings,
    weekly_hours,
    weekly_rest_days,
    candidate_shift_hours,
    must_replace_with_senior,
    back_to_back_rules,
    target_date=None,
    worked_weekends=None,
):
    messages = []
    penalty = 0

    # configurable weights
    penalties = getattr(
        settings,
        "penalties",
        {
            "over_max_hours": 10,
            "under_min_hours": 20,
            "preferred_hours": 1,
            "insufficient_rest": 100,
            "must_be_senior": 500,
            "already_working": 1000,
            "back_to_back": 500,
            "recent_night": 200,
            "weekend_rest": 800,
        },
    )

    # calculate shift-based hours
    min_hours = settings.minWeeklyHours
    max_hours = settings.maxWeeklyHours
    preferred_hours = settings.preferredWeeklyHours
    min_rest = settings.minWeeklyRest
    candidate_total = weekly_hours + candidate_shift_hours

    # warn if over max hours
    if candidate_total > max_hours:
        messages.append(
            f"This staff would exceed max weekly hours ({candidate_total} > {max_hours})."
        )
        penalty += (candidate_total - max_hours) * penalties["over_max_hours"]

    # warn if under min hours
    if candidate_total < min_hours:
        messages.append(
            f"This staff would be below min weekly hours ({candidate_total} < {min_hours})."
        )
        penalty += (min_hours - candidate_total) * penalties["under_min_hours"]

    # warn if under preferred
    if candidate_total < preferred_hours:
        messages.append(
            f"This staff would be below preferred weekly hours ({candidate_total} < {preferred_hours})."
        )
        penalty += (preferred_hours - candidate_total) * penalties["preferred_hours"]

    # rest day warning
    if weekly_rest_days < min_rest:
        messages.append(
            f"This staff has only {weekly_rest_days} rest days, needs at least {min_rest}."
        )
        penalty += (min_rest - weekly_rest_days) * penalties["insufficient_rest"]

    # warn if recent night shift (if enabled, default is True)
    if getattr(settings, "warnRecentNightShift", False) and nurse.recentNightShift == 1:
        messages.append(f"{nurse.nurseId} recently worked a night shift.")
        penalty += penalties["recent_night"]

    # senior requirement
    if must_replace_with_senior and not nurse.isSenior:
        messages.append("Shift requires a senior nurse.")
        penalty += penalties["must_be_senior"]

    # back-to-back rule
    if getattr(settings, "backToBackShift", False):
        for from_id, to_id, rtype in back_to_back_rules:
            if rtype == "same_day":
                messages.append(f"Back-to-back risk: {from_id} → {to_id} (same day).")
                penalty += penalties["back_to_back"]
            elif rtype == "overnight":
                messages.append(f"Back-to-back risk: {from_id} → {to_id} (overnight).")
                penalty += penalties["back_to_back"]

    # weekend rest logic
    if getattr(settings, "weekendRest", False) and target_date and worked_weekends:
        last_weekend_date, last_weekend_day = worked_weekends
        if target_date.weekday() == last_weekend_day:  # same Sat/Sun
            messages.append(
                f"{nurse.nurseId} worked last weekend ({last_weekend_date.strftime('%Y-%m-%d')}) → must rest this {target_date.strftime('%A')}."
            )
            penalty += penalties["weekend_rest"]

    return messages, penalty


# process nurse replacement logic
def preprocess_nurse(nurse, target_date, settings):
    recent_night_window = getattr(
        settings, "recentNightWindowDays", 2
    )  # optional for rest day after night shifts
    night_shift_ids = getattr(settings, "nightShiftIds", [3])

    shift_dates = [parse_date(s.date) for s in nurse.shifts]

    # Define this week as 7 days up to the target date
    this_week_start = target_date - timedelta(days=6)
    recent_night_cutoff = target_date - timedelta(days=recent_night_window)

    # Count shifts in the current week
    shiftsThisWeek = sum(this_week_start <= d <= target_date for d in shift_dates)

    # Check if any shift in the recent night window was a night/overnight shift
    recentNightShift = any(
        parse_date(s.date) >= recent_night_cutoff
        and any(shiftId in night_shift_ids for shiftId in s.shiftIds)
        for s in nurse.shifts
    )

    return SwapCandidateFeatures(
        nurseId=nurse.nurseId,
        isSenior=nurse.isSenior,
        shiftsThisWeek=shiftsThisWeek,
        recentNightShift=int(recentNightShift),
    )


# Extract features from preferences
def extract_preference_features(pref_data: dict) -> dict:
    shifts = pref_data.get("shifts", [])

    all_shift_ids = [shift_id for day in shifts for shift_id in day.get("shiftIds", [])]

    total_preferred = len(all_shift_ids)
    unique_shift_types = len(set(all_shift_ids))
    avg_shift_id = float(np.mean(all_shift_ids)) if all_shift_ids else 0.0

    return {
        "totalPreferredShifts": total_preferred,
        "uniquePreferredShiftTypes": unique_shift_types,
        "avgPreferredShiftId": avg_shift_id,
    }


# build dynamic back to back rules with durations
def build_back_to_back_rules(shifts):
    """
    Build rules dynamically from shift durations.
    Rules are (sid1, sid2, type).
    """
    rules = []
    parsed = []

    shift_ids = [s.id for s in shifts]
    id_to_index = {sid: idx for idx, sid in enumerate(shift_ids)}

    # preprocess
    for s in shifts:
        start_str, end_str = s.duration.split(
            "-"
        )  # split duration format of (0700-1400)
        start = parse_time(start_str)
        end = parse_time(end_str)
        # overnight shift (end < start)
        overnight = end <= start
        parsed.append(
            {
                "id": id_to_index[s.id],  # shift index
                "start": start,
                "end": end,
                "overnight": overnight,
                "name": s.name,
            }
        )

    # same-day adjacency
    for a in parsed:
        for b in parsed:
            if a["id"] == b["id"]:
                continue
            # if shift A ends exactly when shift B starts → consecutive
            if not a["overnight"] and a["end"] == b["start"]:
                rules.append((a["id"], b["id"], "same_day"))

    # overnight adjacency
    for a in parsed:
        if a["overnight"]:
            # overnight shifts should not be followed by any shift starting in the morning
            for b in parsed:
                if not b["overnight"] and b["start"] < 12 * 60:  # morning-ish threshold
                    rules.append((a["id"], b["id"], "overnight"))

    return rules


# check violate back to back rule
def violates_back_to_back(nurse, target_date, target_shift, rules):
    """
    Check if assigning target_shift on target_date causes back-to-back violation.
    """
    for s in nurse.shifts:
        # same day case
        if s.date == target_date:
            for sid in s.shiftIds:
                for a, b, rule_type in rules:
                    if sid == a and target_shift == b and rule_type == "same_day":
                        return True

        # overnight case (yesterday's night → today's AM)
        if datetime.strptime(s.date, "%Y-%m-%d") == datetime.strptime(
            target_date, "%Y-%m-%d"
        ) - timedelta(days=1):
            for sid in s.shiftIds:
                for a, b, rule_type in rules:
                    if sid == a and target_shift == b and rule_type == "overnight":
                        return True
    return False

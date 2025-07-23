from datetime import datetime, timedelta
from schemas.swap.suggestions import SwapCandidateFeatures

# format date
def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")

# validation check with warning messages
def generate_warning(nurse, settings):
    messages = []

    # Calculate shift-based hours
    shift_duration = settings.shiftDurations
    max_hours = settings.maxWeeklyHours
    preferred_hours = settings.preferredWeeklyHours

    weekly_hours = nurse.shiftsThisWeek * shift_duration

    # Warn if over max hours
    if weekly_hours > max_hours:
        messages.append(
            f"{nurse.nurseId} has {weekly_hours} hours this week (limit: {max_hours})."
        )

    # Optional: Warn if under preferred (but not required)
    if weekly_hours < preferred_hours:
        messages.append(
            f"{nurse.nurseId} is below preferred weekly hours ({weekly_hours} < {preferred_hours})."
        )

    # Warn if recent night shift (if enabled, default is True)
    if getattr(settings, "warnRecentNightShift", False) and nurse.recentNightShift == 1:
        messages.append(f"{nurse.nurseId} recently worked a night shift.")

    return " ".join(messages) if messages else "OK"


# process nurse replacement logic
def preprocess_nurse(nurse, target_date, settings):
    shift_duration = settings.shiftDurations
    recent_night_window = getattr(settings, "recentNightWindowDays", 2) # optional for rest day after night shifts
    night_shift_ids = getattr(settings, "nightShiftIds", [3])

    shift_dates = [parse_date(s.date) for s in nurse.shifts]

    # Define this week as 7 days up to the target date
    this_week_start = target_date - timedelta(days=6)
    recent_night_cutoff = target_date - timedelta(days=recent_night_window)

    # Count shifts in the current week
    shiftsThisWeek = sum(this_week_start <= d <= target_date for d in shift_dates)

    # Check if any shift in the recent night window was a night/overnight shift
    recentNightShift = any(
        parse_date(s.date) >= recent_night_cutoff and s.shiftTypeId in night_shift_ids
        for s in nurse.shifts
    )

    return SwapCandidateFeatures(
        nurseId=nurse.nurseId,
        isSenior=nurse.isSenior,
        shiftsThisWeek=shiftsThisWeek,
        recentNightShift=int(recentNightShift)
    )
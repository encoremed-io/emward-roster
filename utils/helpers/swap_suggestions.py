from datetime import datetime, timedelta

# format date
def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")

# validation check with warning messages
def generate_warning(nurse, settings):
    messages = []

    # Calculate shift-based hours
    shift_duration = settings.get("shiftDurations", 8)
    max_hours = settings.get("maxWeeklyHours", 40)
    preferred_hours = settings.get("preferredWeeklyHours", 40)

    weekly_hours = nurse.get("shiftsThisWeek", 0) * shift_duration

    # Warn if over max hours
    if weekly_hours > max_hours:
        messages.append(
            f"{nurse['nurseId']} has {weekly_hours} hours this week (limit: {max_hours})."
        )

    # Optional: Warn if under preferred (but not required)
    if weekly_hours < preferred_hours:
        messages.append(
            f"{nurse['nurseId']} is below preferred weekly hours ({weekly_hours} < {preferred_hours})."
        )

    # Warn if recent night shift (if enabled, default is True)
    if settings.get("warnRecentNightShift", True) and nurse.get("recentNightShift", 0) == 1:
        messages.append(f"{nurse['nurseId']} recently worked a night shift.")

    return " ".join(messages) if messages else "OK"


# process nurse replacement logic
def preprocess_nurse(nurse, target_date, settings):
    shift_duration = settings.get("shiftDurations", 8)
    recent_night_window = settings.get("recentNightWindowDays", 2) # optional for rest day after night shifts
    night_shift_ids = settings.get("nightShiftIds", [3])

    shift_dates = [parse_date(s["date"]) for s in nurse.get("shifts", [])]

    # Define this week as 7 days up to the target date
    this_week_start = target_date - timedelta(days=6)
    recent_night_cutoff = target_date - timedelta(days=recent_night_window)

    # Count shifts in the current week
    shiftsThisWeek = sum(this_week_start <= d <= target_date for d in shift_dates)

    # Check if any shift in the recent night window was a night/overnight shift
    recentNightShift = any(
        parse_date(s["date"]) >= recent_night_cutoff and s.get("shiftTypeId") in night_shift_ids
        for s in nurse.get("shifts", [])
    )

    return {
        **nurse,
        "shiftsThisWeek": shiftsThisWeek,
        "recentNightShift": int(recentNightShift)
    }
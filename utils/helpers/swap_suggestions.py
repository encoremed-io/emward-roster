from datetime import datetime, timedelta

# format date
def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")

# checks for conflicting shifts
def is_conflicting(nurse, target_shift):
    for shift in nurse.get("shifts", []):
        if shift["date"] == target_shift["date"] and shift["type"] == target_shift["type"]:
            return True
    return False

# validation check with warning messages
def generate_warning(nurse, settings):
    messages = []
    if nurse["shifts_this_week"] > settings.get("max_shifts_per_week", 5):
        messages.append(
            f"{nurse['nurse_id']} has {nurse['shifts_this_week']} shifts this week (limit: {settings['max_shifts_per_week']})."
        )
    if settings.get("warn_recent_night_shift", True) and nurse["recent_night_shift"] == 1:
        messages.append(f"{nurse['nurse_id']} recently had a night shift.")
    return " ".join(messages) if messages else None

# process nurse replacement logic
def preprocess_nurse(nurse, target_date, settings):
    shift_dates = [parse_date(s["date"]) for s in nurse.get("shifts", [])]
    this_week_start = target_date - timedelta(days=6)
    recent_night_cutoff = target_date - timedelta(days=settings.get("recent_night_window_days", 2))

    shifts_this_week = sum(this_week_start <= d <= target_date for d in shift_dates)
    recent_night_shift = any(
        parse_date(s["date"]) >= recent_night_cutoff and s["type"] == "night"
        for s in nurse.get("shifts", [])
    )

    return {
        **nurse,
        "shifts_this_week": shifts_this_week,
        "recent_night_shift": int(recent_night_shift)
    }
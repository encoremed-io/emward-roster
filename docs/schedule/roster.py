schedule_roster_description = """
    Generate a schedule based on the given nurse profiles, shift preferences, and other parameters

    The API endpoint takes the following parameters:

    - `profiles`: List of `NurseProfile` objects, which contain the following information:
        - `name`: Name of the nurse
        - `title`: Title of the nurse (e.g. "Senior Nurse", "Junior Nurse")
        - `years_experience`: Number of years of experience the nurse has
    - `preferences`: List of `NursePreference` objects, which contain the following information:
        - `nurse`: Name of the nurse
        - `date`: Date of the shift
        - `shift`: Shift preference (e.g. "AM", "PM", "Night")
        - `timestamp`: Timestamp of the preference
    - `training_shifts`: List of `NurseTraining` objects, which contain the following information:
        - `nurse`: Name of the nurse
        - `date`: Date of the training shift
        - `training`: Shift on training (e.g. "AM", "PM", "Night", "FULL")
    - `previous_schedule`: List of `PrevSchedule` objects. Each object represents a nurse's past schedule and contains:
        - `index`: The name of the nurse.
        - `<Day Date>`: The assigned shift for that day, where the key is a string in the format `"Day YYYY-MM-DD"`
          (e.g., `"Mon 2025-07-07"`), and the value is one of the shift types (e.g. "AM", "PM", "Night") or no work labels (e.g. "MC", "REST", "AL", "EL").
    - `request`: `ScheduleRequest` object, which contains the following information:
        - `start_date`: Start date of the schedule
        - `num_days`: Number of days in the schedule
        - `shift_durations`: List of shift durations in hours
        - `min_nurses_per_shift`: Minimum number of nurses per shift
        - `min_seniors_per_shift`: Minimum number of senior nurses per shift
        - `max_weekly_hours`: Maximum weekly hours for each nurse
        - `preferred_weekly_hours`: Preferred weekly hours for each nurse
        - `min_acceptable_weekly_hours`: Minimum acceptable weekly hours for each nurse
        - `activate_am_cov`: Whether to activate AM coverage constraints
        - `am_coverage_min_percent`: Minimum percentage of AM shifts that must be covered
        - `am_coverage_min_hard`: Whether the minimum percentage is a hard constraint
        - `am_coverage_relax_step`: Relaxation step for the minimum percentage
        - `am_senior_min_percent`: Minimum percentage of senior nurses that must be assigned to AM shifts
        - `am_senior_min_hard`: Whether the minimum percentage is a hard constraint
        - `am_senior_relax_step`: Relaxation step for the minimum percentage
        - `weekend_rest`: Whether to ensure that each nurse has a weekend rest
        - `back_to_back_shift`: Whether to prevent back-to-back shifts
        - `use_sliding_window`: Whether to use a sliding window for shift assignments
        - `shift_balance`: Whether to balance the number of shifts between nurses
        - `priority_setting`: Priority setting for the solver (e.g. "Fairness", "Fairness-leaning", "50/50", "Preference-leaning", "Preference"). Only activated when `shift_balance` is `True`.
        - `fixed_assignments`: List of fixed shift assignments (optional), with the following fields:
            - `nurse`: Name of the nurse
            - `date`: Date of the shift
            - `fixed`: Fixed declaration (e.g. "EL", "MC")

    The API endpoint returns a JSON object with the following keys:

    - `schedule`: List of shift assignments, where each assignment is a dictionary with the following keys:
        - `nurse`: Name of the nurse
        - `date`: Date of the shift
        - `shift`: Shift assignment (e.g. "AM", "PM", "Night")
    - `summary`: List of summary statistics, where each statistic is a dictionary with the following keys:
        - `metric`: Name of the metric (e.g. "Hours_Week1_Real", "Prefs_Unmet")
        - `value`: Value of the metric
    - `violations`: List of constraint violations, where each violation is a dictionary with the following keys:
        - `constraint`: Name of the constraint (e.g. "Low Hours Nurses", "Low Senior AM Days")
        - `value`: Value of the constraint
     - `metrics`: A dictionary containing evaluation metrics. Keys include "Preference Unmet" and "Fairness Gap", with each value representing the corresponding metric score.

    The API endpoint raises an HTTPException with a status code of 400 if the input is invalid and raises an HTTPException with a status code of 422 if no feasible solution is found.
    """

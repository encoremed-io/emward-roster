from datetime import timedelta
from ortools.sat.python import cp_model
from utils.nurse_utils import (
    get_senior_set,
    get_nurse_names,
    shuffle_order,
    get_doubleShift_nurses,
    get_nurse_name_by_id,
)
from utils.shift_utils import (
    make_shift_index,
    extract_prefs_info,
    extract_leave_days,
    extract_leaves_info,
    normalize_fixed_assignments,
    make_weekend_pairs,
    get_days_with_el,
    extract_training_shifts_info,
    normalise_date,
)
from core.assumption_flags import define_hard_rules
from utils.constants import (
    PREF_MISS_PENALTY,
    FAIRNESS_GAP_PENALTY,
    FAIRNESS_GAP_THRESHOLD,
    SHIFT_IMBALANCE_PENALTY,
    SHIFT_IMBALANCE_THRESHOLD,
)
from exceptions.custom_errors import (
    InvalidPreviousScheduleError,
    InvalidPrioritySettingError,
)


def build_variables(model, nurse_names, num_days, prev_days, shift_types):
    """Builds the work[n,d,s] BoolVars for every nurse/day/shift, including previous days."""
    work = {
        (n, d, s): model.NewBoolVar(f"work_{n}_{d}_{s}")
        for n in nurse_names
        for d in range(-prev_days, num_days)
        for s in range(shift_types)
    }
    return work


def make_model():
    """Creates a new CP-SAT model instance."""
    model = cp_model.CpModel()
    return model


def setup_model(
    profiles_df,
    preferences_df,
    training_shifts_df,
    prev_schedule_df,
    leaves_df,
    start_date,
    num_days,
    no_work_labels,
    fixed_assignments=None,
    shift_details=None,
    shifts=[],
):
    """
    Sets up the scheduling model with necessary data and constraints.

    This function initializes a constraint programming model for nurse scheduling,
    processes input data, and extracts relevant information for model setup. It prepares
    the nurse names, shift indices, and various constraints such as preferences and training
    shifts. It also handles previous schedules and fixed assignments.

    Args:
        profiles_df (pd.DataFrame): DataFrame containing nurse profiles.
        preferences_df (pd.DataFrame): DataFrame containing nurse shift preferences.
        training_shifts_df (pd.DataFrame): DataFrame containing training shift information.
        prev_schedule_df (pd.DataFrame): DataFrame containing previous schedule information.
        start_date (Union[pd.Timestamp, dt_date]): Start date of the scheduling period.
        num_days (int): Number of days for the scheduling period.
        shift_labels (List[str]): List of shift labels (e.g., "AM", "PM", "Night").
        no_work_labels (List[str]): List of labels indicating no work (e.g., "REST", "AL").
        fixed_assignments (Optional[Dict[Tuple[str, int], str]]): Pre-assigned shifts.

    Returns:
        tuple: Contains initialized model, nurse names, previous schedule, senior nurse names,
               shift index mapping, normalized start date, hard constraint rules, shift preferences,
               preferences by nurse, training shifts, training by nurse, normalized fixed assignments,
               leave day sets (MC, AL, EL), days with EL, weekend pairs, number of shift types, work
               variables, previous days count, and total days count.
    """
    model = make_model()
    nurse_names = get_nurse_names(profiles_df)
    clean_prev_sched = prev_schedule_df.reindex(
        nurse_names
    )  # Add missing nurses with no previous work data
    og_nurse_names, shuffled_nurse_names = shuffle_order(nurse_names)
    senior_names = get_senior_set(
        profiles_df
    )  # Assume senior nurses have ≥3 years experience
    double_shift_nurses = get_doubleShift_nurses(
        profiles_df
    )  # Nurses who can work double shifts
    shift_str_to_idx = make_shift_index(shifts)
    # Map shift code → int index for your decision variables
    hard_rules = define_hard_rules(model)  # Track hard constraint violations

    date_start = normalise_date(
        start_date
    )  # Normalize start date to a standard date format

    training_shifts, training_by_nurse = extract_training_shifts_info(
        training_shifts_df,
        date_start,
        shuffled_nurse_names,
        num_days,
        shifts,
        no_work_labels,
    )

    shift_preferences, prefs_by_nurse = extract_prefs_info(
        preferences_df,
        date_start,
        shuffled_nurse_names,
        num_days,
        shifts,
        no_work_labels,
        training_by_nurse,
    )

    leaves_by_nurse = extract_leaves_info(
        leaves_df,
        date_start,
        shuffled_nurse_names,
        num_days,
    )

    if prev_schedule_df is None or prev_schedule_df.empty:
        prev_days = 0
    else:
        last_date = normalise_date(max(prev_schedule_df.columns))
        if last_date >= date_start:
            raise InvalidPreviousScheduleError(
                "Previous schedule end date must be before start date."
            )
        earliest_date = normalise_date(min(prev_schedule_df.columns))
        prev_days = (date_start - earliest_date).days

    total_days = prev_days + num_days

    raw_pairs = make_weekend_pairs(total_days, date_start - timedelta(days=prev_days))
    weekend_pairs = [(d1 - prev_days, d2 - prev_days) for d1, d2 in raw_pairs]

    shift_types = len(shifts)
    work = build_variables(
        model, shuffled_nurse_names, num_days, prev_days, shift_types
    )

    return (
        model,
        shuffled_nurse_names,
        og_nurse_names,
        clean_prev_sched,
        senior_names,
        double_shift_nurses,
        shift_str_to_idx,
        date_start,
        hard_rules,
        shift_preferences,
        prefs_by_nurse,
        training_shifts,
        training_by_nurse,
        fixed_assignments,
        weekend_pairs,
        shift_types,
        work,
        prev_days,
        total_days,
        shift_details,
        shifts,
        leaves_by_nurse,
    )


def adjust_low_priority_params(doAdjustment: bool, option: str):
    """
    Adjusts and returns the penalty parameters for low priority constraints based on the given option.

    Args:
        doAdjustment (bool): Flag indicating whether to adjust parameters or not.
        option (str): The option to determine the adjustment strategy. Valid options are
                      'FAIRNESS', 'FAIRNESS-LEANING', '50/50', 'PREFERENCE-LEANING', and 'PREFERENCE'.

    Returns:
        tuple: Contains the adjusted values for pref_miss_penalty, fairness_gap_penalty,
               fairness_gap_threshold, shift_imbalance_penalty, and shift_imbalance_threshold.

    Raises:
        InvalidPrioritySettingError: If an invalid option is provided.
    """
    if not doAdjustment:
        return (
            PREF_MISS_PENALTY,
            FAIRNESS_GAP_PENALTY,
            FAIRNESS_GAP_THRESHOLD,
            SHIFT_IMBALANCE_PENALTY,
            SHIFT_IMBALANCE_THRESHOLD,
        )

    else:
        match (str(option).strip().upper()):
            case "FAIRNESS":
                pref_miss_penalty = 10
                fairness_gap_penalty = 2
                fairness_gap_threshold = 0
                shift_imbalance_penalty = 10
                shift_imbalance_threshold = 1
            case "FAIRNESS-LEANING":
                pref_miss_penalty = 2
                fairness_gap_penalty = 5
                fairness_gap_threshold = FAIRNESS_GAP_THRESHOLD // 2
                shift_imbalance_penalty = 5
                shift_imbalance_threshold = 2
            case "50/50":
                pref_miss_penalty = PREF_MISS_PENALTY
                fairness_gap_penalty = FAIRNESS_GAP_PENALTY
                fairness_gap_threshold = FAIRNESS_GAP_THRESHOLD
                shift_imbalance_penalty = SHIFT_IMBALANCE_PENALTY
                shift_imbalance_threshold = SHIFT_IMBALANCE_THRESHOLD
            case "PREFERENCE-LEANING":
                pref_miss_penalty = 5
                fairness_gap_penalty = 1
                fairness_gap_threshold = FAIRNESS_GAP_THRESHOLD + (
                    (100 - FAIRNESS_GAP_THRESHOLD) // 2
                )
                shift_imbalance_penalty = 1
                shift_imbalance_threshold = 10
            case "PREFERENCE":
                pref_miss_penalty = 50
                fairness_gap_penalty = 0
                fairness_gap_threshold = 100
                shift_imbalance_penalty = 0
                shift_imbalance_threshold = 100
            case _:
                raise InvalidPrioritySettingError(
                    "Invalid priority setting. Expected 'FAIRNESS', 'FAIRNESS-LEANING', '50/50', 'PREFERENCE-LEANING', or 'PREFERENCE'."
                )

    return (
        pref_miss_penalty,
        fairness_gap_penalty,
        fairness_gap_threshold,
        shift_imbalance_penalty,
        shift_imbalance_threshold,
    )

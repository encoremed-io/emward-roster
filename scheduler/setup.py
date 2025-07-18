import pandas as pd
from ortools.sat.python import cp_model
from utils.nurse_utils import get_senior_set, get_nurse_names, shuffle_order
from utils.shift_utils import (
    make_shift_index, 
    extract_prefs_info, 
    extract_leave_days, 
    normalize_fixed_assignments,
    make_weekend_pairs,
    get_days_with_el,
    extract_training_shifts_info
)
from core.assumption_flags import define_hard_rules

def normalise_date(input_date):
    """ Convert input date to a standard date format. """
    if isinstance(input_date, pd.Timestamp):
        return input_date.date()
    return input_date


def build_variables(model, nurse_names, num_days, shift_types):
    """ Builds the work[n,d,s] BoolVars for every nurse/day/shift. """
    work = {
        (n, d, s): model.NewBoolVar(f'work_{n}_{d}_{s}')
        for n in nurse_names for d in range(num_days) for s in range(shift_types)
    }
    return work


def make_model():
    """ Creates a new CP-SAT model instance. """
    model = cp_model.CpModel()
    return model


def setup_model(profiles_df, preferences_df, training_shifts_df, start_date, num_days, shift_labels, no_work_labels, fixed_assignments=None):
    """
    Sets up a CpModel instance with all the necessary variables and constraints.
    
    :param profiles_df: A DataFrame of nurse profiles.
    :param preferences_df: A DataFrame of nurse preferences.
    :param training_shifts_df: A DataFrame of non-MC preferences.
    :param start_date: A pd.Timestamp object of the schedule start date.
    :param num_days: The number of days in the schedule.
    :param shift_labels: A list of strings of the shift labels.
    :param no_work_labels: A list of strings of no-work labels.
    :param fixed_assignments: A dictionary of fixed assignments.
    
    :return: A tuple of the model instance, shuffled nurse names, original nurse names, set of senior names, shift string to index mapping, start date, hard rules dictionary, shift preferences dictionary, preferences by nurse dictionary, training shifts dictionary, training shifts by nurse dictionary, fixed assignments dictionary, MC sets dictionary, annual leave sets dictionary, extra leave sets dictionary, days with extra leave set, weekend pairs set, number of shift types, and work variables dictionary.
    """
    model = make_model()
    nurse_names = get_nurse_names(profiles_df)
    og_nurse_names, shuffled_nurse_names = shuffle_order(nurse_names)
    senior_names = get_senior_set(profiles_df)          # Assume senior nurses have ≥3 years experience
    shift_str_to_idx = make_shift_index(shift_labels)   # Map shift code → int index for your decision variables
    hard_rules = define_hard_rules(model)               # Track hard constraint violations

    date_start = normalise_date(start_date)         # Normalize start date to a standard date format

    fixed_assignments = normalize_fixed_assignments(
        fixed_assignments,
        set(shuffled_nurse_names),
        num_days
    )

    training_shifts, training_by_nurse = extract_training_shifts_info(
        training_shifts_df, profiles_df, date_start, shuffled_nurse_names, num_days, shift_labels, no_work_labels, fixed_assignments
    )

    shift_preferences, prefs_by_nurse = extract_prefs_info(
        preferences_df, profiles_df, date_start, shuffled_nurse_names, num_days, shift_labels, no_work_labels, training_by_nurse, fixed_assignments
    )

    mc_sets, al_sets, el_sets = extract_leave_days(
        profiles_df, preferences_df, shuffled_nurse_names, date_start, num_days, fixed_assignments
    )

    days_with_el = get_days_with_el(el_sets)
    weekend_pairs = make_weekend_pairs(num_days, date_start)
    shift_types = len(shift_labels)
    work = build_variables(model, shuffled_nurse_names, num_days, shift_types)

    return model, shuffled_nurse_names, og_nurse_names, senior_names, shift_str_to_idx, date_start, hard_rules, shift_preferences, prefs_by_nurse, training_shifts, training_by_nurse, fixed_assignments, mc_sets, al_sets, el_sets, days_with_el, weekend_pairs, shift_types, work

import pandas as pd
from ortools.sat.python import cp_model
from utils.constants import SHIFT_LABELS
from utils.nurse_utils import get_senior_set, get_nurse_names, shuffle_order
from utils.shift_utils import (
    make_shift_index, 
    extract_prefs_info, 
    extract_leave_days, 
    normalize_fixed_assignments,
    make_weekend_pairs
)
from core.hard_rules import define_hard_rules

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
    """ Create a new CP-SAT model instance. """
    model = cp_model.CpModel()
    return model


def setup_model(profiles_df, preferences_df, start_date, num_days, shift_labels, fixed_assignments=None):
    model = make_model()
    nurse_names = get_nurse_names(profiles_df)
    og_nurse_names, shuffled_nurse_names = shuffle_order(nurse_names)
    senior_names = get_senior_set(profiles_df)          # Assume senior nurses have ≥3 years experience
    shift_str_to_idx = make_shift_index(SHIFT_LABELS)   # Map shift code → int index for your decision variables
    hard_rules = define_hard_rules(model)               # Track hard constraint violations

    date_start = normalise_date(start_date)         # Normalize start date to a standard date format

    fixed_assignments = normalize_fixed_assignments(
        fixed_assignments,
        set(shuffled_nurse_names),
        num_days
    )

    shift_preferences, prefs_by_nurse = extract_prefs_info(
        preferences_df, profiles_df, date_start, shuffled_nurse_names, num_days, shift_labels
    )

    mc_sets, al_sets, el_sets = extract_leave_days(
        profiles_df, preferences_df, shuffled_nurse_names, date_start, num_days, fixed_assignments
    )
    weekend_pairs = make_weekend_pairs(num_days, date_start)
    shift_types = len(shift_labels)
    work = build_variables(model, shuffled_nurse_names, num_days, shift_types)

    return model, shuffled_nurse_names, og_nurse_names, senior_names, shift_str_to_idx, date_start, hard_rules, shift_preferences, prefs_by_nurse, fixed_assignments, mc_sets, al_sets, el_sets, weekend_pairs, shift_types, work
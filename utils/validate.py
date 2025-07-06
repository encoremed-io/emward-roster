import pandas as pd
from utils.nurse_utils import get_senior_set

def validate_data(profiles_df: pd.DataFrame, preferences_df: pd.DataFrame):
    """ Validate that the nurse profiles and preferences match. """
    missing, extra = validate_nurse_data(profiles_df, preferences_df)
    if missing or extra:
        raise ValueError(
            f"Mismatch between nurse profiles and preferences:\n"
            f" • Not found in preferences: {sorted(missing)}\n"
            f" • Not found in profiles: {sorted(extra)}\n"
        )
    

def validate_nurse_data(profiles_df: pd.DataFrame, preferences_df: pd.DataFrame):
    """ Validate that the nurse profiles and preferences match. """
    profile_names = set(profiles_df['Name'].str.strip())
    preference_names = set(preferences_df.index.str.strip())
    missing = profile_names - preference_names
    extra = preference_names - profile_names
    return missing, extra       
# Returns sets of names that are missing or extra, if none returns empty sets


def validate_input_params(
        profiles_df: pd.DataFrame, 
        num_days: int, 
        min_nurses_per_shift: int,
        min_seniors_per_shift: int,
        max_working_hours: int, 
        preferred_work_hours: int, 
        min_acceptable_working_hours: int):
    """ Validate input parameters. """
    nurse_amt = len(set(profiles_df['Name'].str.strip()))
    senior_amt = len(get_senior_set(profiles_df))
    errors = []
    
    if nurse_amt < min_nurses_per_shift:
        errors.append(f" • Number of nurses ({nurse_amt}) must be greater than or equal to min nurses per shift ({min_nurses_per_shift}).\n")

    if senior_amt < min_seniors_per_shift:
        errors.append(f" • Number of seniors ({senior_amt}) must be greater than or equal to min seniors per shift ({min_seniors_per_shift}).\n")

    if num_days < 1:
        errors.append(" • End date must be after or same as start date.\n")

    if preferred_work_hours > max_working_hours:
        errors.append(" • Preferred work hours must be less than or equal to max working hours.\n")

    if preferred_work_hours < min_acceptable_working_hours:
        errors.append(" • Preferred work hours must be greater than or equal to min acceptable working hours.\n")

    if min_acceptable_working_hours > max_working_hours:
        errors.append(" • Min acceptable working hours must be less than or equal to max working hours.\n")

    if errors:
        errors.insert(0, "Recheck your inputs:\n")

    return errors


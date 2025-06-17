import pandas as pd

def validate_nurse_data(profiles_df: pd.DataFrame, preferences_df: pd.DataFrame):
    """ Validate that the nurse profiles and preferences match. """
    profile_names = set(profiles_df['Name'].str.strip())
    preference_names = set(preferences_df.index.str.strip())
    missing = profile_names - preference_names
    extra = preference_names - profile_names
    return missing, extra       
# Returns sets of names that are missing or extra, if none returns empty sets

import pandas as pd

def validate_nurse_data(profiles_df: pd.DataFrame, preferences_df: pd.DataFrame):
    profile_names = set(profiles_df['Name'].str.strip())
    preference_names = set(preferences_df.index.str.strip())
    missing = profile_names - preference_names
    extra = preference_names - profile_names
    if missing or extra:
        return missing, extra
    return None, None  # valid

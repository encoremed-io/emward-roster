import pandas as pd
import random


def get_senior_set(profiles_df):
    """Get set of senior nurses."""
    return {
        str(row["id"]).strip().upper()
        for _, row in profiles_df.iterrows()
        if row.get("title", "").upper() == "SENIOR"
        or row.get("years of experience", 0) >= 3
    }


def get_nurse_names(profiles_df: pd.DataFrame) -> list[str]:
    """Get set of nurse names from profiles DataFrame."""
    nurses = profiles_df.to_dict(orient="records")
    nurse_names = [n["id"].strip().upper() for n in nurses]
    return nurse_names


def get_doubleShift_nurses(profiles_df: pd.DataFrame) -> list[str]:
    """Get set of nurses who can work double shifts."""
    return [
        str(row["name"]).strip().upper()
        for _, row in profiles_df.iterrows()
        if row.get("double shift", False) is True
    ]


def extract_nurse_info(profiles_df):
    """
    From profiles_df compute:
     - nurse_names: shuffled, uppercase
     - og_nurse_names: original order
     - senior_names: those with ≥3 years experience
    """
    nurse_names = get_nurse_names(profiles_df)
    og_nurse_names, shuffled_nurse_names = shuffle_order(nurse_names)
    senior_names = get_senior_set(profiles_df)
    return og_nurse_names, shuffled_nurse_names, senior_names


def shuffle_order(lst):
    """Shuffle the order of a list and return both original and shuffled lists."""
    og_lst = lst.copy()  # Save original order of list
    random.shuffle(lst)  # Shuffle order for random shift assignments
    return og_lst, lst

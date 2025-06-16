import pandas as pd
from datetime import date as dt_date
from collections import defaultdict

def get_shift_prefs_and_mc_days(preferences_df, profiles_df, start_date, active_days):
    """Extract shift preferences and MC days."""
    if isinstance(start_date, pd.Timestamp):
        date_start = start_date.date()
    else:
        date_start = start_date

    nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
    shift_str_to_idx = {'AM': 0, 'PM': 1, 'NIGHT': 2}
    
    shift_prefs = {n: {} for n in nurse_names}
    mc_days = {n: set() for n in nurse_names}
    
    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        for label, val in row.items():
            if isinstance(label, pd.Timestamp):
                d = label.date()
            elif isinstance(label, dt_date):
                d = label
            else:
                d = pd.to_datetime(str(label)).date()

            offset = (d - date_start).days
            if pd.notna(val) and 0 <= offset < active_days:
                v = str(val).strip().upper()
                if v == 'MC':
                    mc_days[nm].add(offset)
                elif v in shift_str_to_idx:
                    shift_prefs[nm][offset] = shift_str_to_idx[v]
                    
    return shift_prefs, mc_days


def get_senior_set(profiles_df):
    """Get set of senior nurses."""
    return {
        str(row["Name"]).strip().upper()
        for _, row in profiles_df.iterrows()
        if row.get("Title", "").upper() == "SENIOR"
    }


def get_el_days(fixed_assignments):
    """Extract EL days from fixed assignments."""
    el_days = defaultdict(set)
    for (nurse, d), label in fixed_assignments.items():
        if label.upper() == "EL":
            el_days[nurse].add(d)
    return el_days
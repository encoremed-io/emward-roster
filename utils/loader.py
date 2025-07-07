import pandas as pd
from typing import Union, IO

def load_nurse_profiles(
    path_or_buffer: Union[str, bytes, IO, None] = 'data/nurse_profiles.xlsx') -> pd.DataFrame:
    """
    Load nurse profiles from an Excel file.

    Parameters:
        path_or_buffer: Path to the Excel file (str), a file-like object (e.g., Streamlit UploadedFile),
                        or bytes. Defaults to 'data/nurse_profiles.xlsx'.

    Returns:
        pd.DataFrame: DataFrame with nurse names standardized (stripped and uppercased).
    """
    try:
        df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise ValueError(f"Error loading nurse profiles: {e}")
    df['Name'] = df['Name'].str.strip().str.upper()
    return df


def load_shift_preferences(
    path_or_buffer: Union[str, bytes, IO, None] = 'data/nurse_preferences.xlsx') -> pd.DataFrame:
    """
    Load nurse shift preferences from an Excel file.

    Parameters:
        path_or_buffer: Path to the Excel file (str), a file-like object (e.g., Streamlit UploadedFile),
                        or bytes. Defaults to 'data/nurse_preferences.xlsx'.

    Returns:
        pd.DataFrame: DataFrame with nurse names as index (stripped and uppercased), and columns as dates.
    """
    try:
        df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise ValueError(f"Error loading nurse preferences: {e}")
    df.rename(columns={df.columns[0]: 'Name'}, inplace=True)
    df.set_index('Name', inplace=True)
    # Parse date columns
    cleaned = []
    for col in df.columns:
        # Assume format contains YYYY-MM-DD
        try:
            dt = pd.to_datetime(str(col).strip().split()[-1], format="%Y-%m-%d").date()
            cleaned.append(dt)
        except ValueError as e:
            raise ValueError(f"Invalid date format in column '{col}': {e}")
    df.columns = cleaned
    df.index = df.index.str.strip().str.upper()
    return df
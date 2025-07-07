import pandas as pd
import re
from typing import Union, IO

def load_nurse_profiles(
    path_or_buffer: Union[str, bytes, IO, None] = 'data/nurse_profiles.xlsx',
    drop_duplicates: bool = True,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    Load nurse profiles flexibly by matching columns containing 'name', 'title',  'experience'.

    Parameters:
        path_or_buffer: Path to Excel file or file-like object.
        drop_duplicates: Remove duplicate names if True.
        allow_missing: Allow rows with missing values if True.

    Returns:
        Cleaned DataFrame with standardized columns: Name, Title, Years of experience.
    """
    try:
        df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise ValueError(f"Error loading nurse profiles: {e}")

    # Lowercase map of original column names
    col_map = {col.lower().strip(): col for col in df.columns}

    def find_col(keyword: str) -> str:
        for key, original in col_map.items():
            if keyword in key:
                return original
        raise ValueError(f"No column found containing '{keyword}'")

    try:
        name_col = find_col("name")
        title_col = find_col("title")
        exp_col = find_col("experience") or find_col("year")
    except ValueError as e:
        raise ValueError(f"Missing expected column: {e}")

    df = df[[name_col, title_col, exp_col]]
    df.columns = ["Name", "Title", "Years of experience"]

    df["Name"] = df["Name"].astype(str).str.strip().str.upper()

    if not allow_missing:
        df.dropna(subset=["Name", "Title", "Years of experience"], inplace=True)

    if drop_duplicates:
        df.drop_duplicates(subset=["Name"], inplace=True)
    else:
        if df.duplicated(subset=["Name"]).any():
            raise ValueError("Duplicate nurse names found.")

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

    cleaned_cols = []
    for col in df.columns:
        try:
            col_str = str(col).strip()

            # Remove weekday prefix if exists: match "Mon", "Tues", "Thu", etc.
            col_str = re.sub(r'^[A-Za-z]{3,9}\s+', '', col_str)

            dt = pd.to_datetime(col_str, errors='raise').date()
            cleaned_cols.append(dt)
        except Exception as e:
            raise ValueError(f"Invalid date format in column '{col}': {e}")

    df.columns = cleaned_cols
    df.index = df.index.str.strip().str.upper()
    return df

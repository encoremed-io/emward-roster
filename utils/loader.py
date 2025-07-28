import pandas as pd
import re
from typing import Union, IO
from pathlib import Path
from config.paths import DATA_DIR
from exceptions.custom_errors import FileContentError, FileReadingError

def load_nurse_profiles(
    path_or_buffer: Union[str, Path, bytes, IO, None] = None,
    drop_duplicates: bool = True,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    Load nurse profiles flexibly by matching columns containing 'name', 'title', or 'experience/year'.

    Parameters:
        path_or_buffer: Path to Excel file or file-like object. Defaults to 'data/nurse_profiles.xlsx'.
        drop_duplicates: Remove duplicate names if True.
        allow_missing: Allow rows with missing values if True.

    Returns:
        Cleaned DataFrame with standardized columns: Name, Title, Years of experience.
    """
    if path_or_buffer is None:
        path_or_buffer = DATA_DIR / "nurse_profiles.xlsx"

    try:
        df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise FileReadingError(f"Error loading nurse profiles: {e}")

    col_map = {col.lower().strip(): col for col in df.columns}

    def find_col(*keywords: str) -> str:
        """Find column containing any of the keywords."""
        for key, original in col_map.items():
            if any(k in key for k in keywords):
                return original
        raise FileContentError(f"No column found containing {keywords}")

    try:
        name_col = find_col("name")
        title_col = find_col("title")
        exp_col = find_col("experience", "year")
    except Exception as e:
        raise FileContentError(f"Missing expected column: {e}")

    df = df[[name_col, title_col, exp_col]]
    df.columns = ["Name", "Title", "Years of experience"]

    df["Name"] = df["Name"].astype(str).str.strip().str.upper()

    if not allow_missing:
        df.dropna(subset=["Name", "Title", "Years of experience"], inplace=True)

    if drop_duplicates:
        df.drop_duplicates(subset=["Name"], inplace=True)
    else:
        if df.duplicated(subset=["Name"]).any():
            raise FileContentError("Duplicate nurse names found.")

    return df


def load_shift_preferences(
    path_or_buffer: Union[str, Path, bytes, IO, None] = None) -> pd.DataFrame:
    """
    Load nurse shift preferences from an Excel file.

    Parameters:
        path_or_buffer: Path to the Excel file (str), a file-like object (e.g., Streamlit UploadedFile),
                        or bytes. Defaults to 'data/nurse_preferences.xlsx'.

    Returns:
        pd.DataFrame: DataFrame with nurse names as index (stripped and uppercased), and columns as dates.
    """
    if path_or_buffer is None:
        path_or_buffer = DATA_DIR / "nurse_preferences.xlsx"

    try:
        df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise FileReadingError(f"Error loading nurse preferences: {e}")

    df.rename(columns={df.columns[0]: 'Name'}, inplace=True)
    df.set_index('Name', inplace=True)

    invalid_cols = []
    cleaned_cols = []
    for col in df.columns:
        try:
            col_str = str(col).strip()

            # Remove weekday prefix if exists: match "Mon", "Tues", "Thu", etc.
            col_str = re.sub(r'^[A-Za-z]{3,9}\s+', '', col_str)

            dt = pd.to_datetime(col_str, errors='raise').date()
            cleaned_cols.append(dt)
        except Exception as e:
            invalid_cols.append(col)
            
    if invalid_cols:
        raise FileContentError(f"Invalid date format in columns: {', '.join(invalid_cols)}")

    df.columns = cleaned_cols
    df.index = df.index.str.strip().str.upper()
    if df.index.has_duplicates:
        raise FileContentError(f"Duplicate nurse names found in preferences file in rows. Duplicated values: {df.index[df.index.duplicated()].tolist()}.")
    return df


def load_training_shifts(
    path_or_buffer: Union[str, Path, bytes, IO, None] = None) -> pd.DataFrame:
    """
    Load nurse training shifts from an Excel file.

    Parameters:
        path_or_buffer: Path to the Excel file (str), a file-like object (e.g., Streamlit UploadedFile),
                        or bytes. Defaults to 'data/training_shifts.xlsx'.

    Returns:
        pd.DataFrame: DataFrame with nurse names as index (stripped and uppercased), and columns as dates.
    """
    if path_or_buffer is None:
        path_or_buffer = DATA_DIR / "training_shifts.xlsx"

    try:
        df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise FileReadingError(f"Error loading nurse training shifts: {e}")

    df.rename(columns={df.columns[0]: 'Name'}, inplace=True)
    df.set_index('Name', inplace=True)

    invalid_cols = []
    cleaned_cols = []
    for col in df.columns:
        try:
            col_str = str(col).strip()

            # Remove weekday prefix if exists: match "Mon", "Tues", "Thu", etc.
            col_str = re.sub(r'^[A-Za-z]{3,9}\s+', '', col_str)

            dt = pd.to_datetime(col_str, errors='raise').date()
            cleaned_cols.append(dt)
        except Exception as e:
            invalid_cols.append(col)
            
    if invalid_cols:
        raise FileContentError(f"Invalid date format in columns: {', '.join(invalid_cols)}")

    df.columns = cleaned_cols
    df.index = df.index.str.strip().str.upper()
    if df.index.has_duplicates:
        raise FileContentError(f"Duplicate nurse names found in training shifts file in rows. Duplicated values: {df.index[df.index.duplicated()].tolist()}.")
    return df


def load_prev_schedule(
    path_or_buffer: Union[str, Path, bytes, IO, None] = None) -> pd.DataFrame:
    """
    Load nurse training shifts from an Excel file.

    Parameters:
        path_or_buffer: Path to the Excel file (str), a file-like object (e.g., Streamlit UploadedFile),
                        or bytes. Defaults to 'data/previous_schedule.xlsx'.

    Returns:
        pd.DataFrame: DataFrame with nurse names as index (stripped and uppercased), and columns as dates.
    """
    if path_or_buffer is None:
        path_or_buffer = DATA_DIR / "previous_schedule.xlsx"

    try:
        df = pd.read_excel(path_or_buffer)
    except Exception as e:
        raise FileReadingError(f"Error loading nurse training shifts: {e}")

    df.rename(columns={df.columns[0]: 'Name'}, inplace=True)
    df.set_index('Name', inplace=True)

    invalid_cols = []
    cleaned_cols = []
    for col in df.columns:
        try:
            col_str = str(col).strip()

            # Remove weekday prefix if exists: match "Mon", "Tues", "Thu", etc.
            col_str = re.sub(r'^[A-Za-z]{3,9}\s+', '', col_str)

            dt = pd.to_datetime(col_str, errors='raise').date()
            cleaned_cols.append(dt)
        except Exception as e:
            invalid_cols.append(col)
            
    if invalid_cols:
        raise FileContentError(f"Invalid date format in columns: {', '.join(invalid_cols)}")

    df.columns = cleaned_cols
    df.index = df.index.str.strip().str.upper()
    if df.index.has_duplicates:
        raise FileContentError(f"Duplicate nurse names found in previous schedule file in rows. Duplicated values: {df.index[df.index.duplicated()].tolist()}.")
    return df

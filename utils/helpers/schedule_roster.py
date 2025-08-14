import pandas as pd


def standardize_profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the column names of a nurse profile DataFrame to "Name", "Title", "Years of experience", "Double Shift".

    The function takes a DataFrame with columns representing nurse names, titles, years of experience, and double shift eligibility.
    It returns a new DataFrame with the same data, but with standardized column names.

    The function first builds a dictionary mapping lower-case, stripped column names to the original column names.
    It then uses this dictionary to find the columns in the DataFrame that match the candidates.
    If no exact match is found, it tries a substring match.
    If no match is found, it raises a ValueError.

    The function then copies the relevant columns into a new DataFrame and renames them.
    Finally, it strips and upper-cases the Name column and returns the new DataFrame.
    """
    col_map = {col.lower().strip(): col for col in df.columns}

    def find_col(*candidates: str) -> str:
        # Try exact match first
        for c in candidates:
            if c in col_map:
                return col_map[c]
        # Then try substring match
        for lower, original in col_map.items():
            if any(c in lower for c in candidates):
                return original
        raise ValueError(f"No column matching {candidates} in {list(df.columns)}")

    id_src = find_col("id")
    name_src = find_col("name")
    title_src = find_col("title")
    exp_src = find_col("experience", "year")
    double_shift_src = find_col(
        "double shift", "doubleshift", "can double", "can work double"
    )

    out = df[[id_src, name_src, title_src, exp_src, double_shift_src]].copy()
    out.columns = ["Id", "Name", "Title", "Years of experience", "Double Shift"]
    out["Name"] = out["Name"].astype(str).str.strip().str.upper()
    out["Double Shift"] = out["Double Shift"].astype(bool)

    return out


def normalize_names(series):
    """Trim, uppercase, ensure string dtype."""
    return series.astype(str).str.strip().str.upper()

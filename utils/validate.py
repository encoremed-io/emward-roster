import pandas as pd
from .nurse_utils import get_senior_set
from exceptions.custom_errors import InputMismatchError


def validate_data(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    file1_name: str = "file 1",
    file2_name: str = "file 2",
    exact_match: bool = False,
):
    """
    Validate name consistency between two DataFrames.

    If exact_match is True, ensures df1 and df2 contain the same set of names.
    If False, only checks that all names in df2 exist in df1. Extra names raise an error;
    missing names produce a warning.

    Args:
        df1 (pd.DataFrame): Reference DataFrame (e.g. profiles, with names in 'Name' column).
        df2 (pd.DataFrame): Secondary DataFrame (e.g. preferences, with names in index).
        file1_name (str): Label for df1 (used in messages).
        file2_name (str): Label for df2 (used in messages).
        exact_match (bool): Whether to require exact name match (default: False).

    Returns:
        Optional[str]: Warning message if names found in df1 but not in df2 and exact_match is False.

    Raises:
        InputMismatchError: If name mismatch is found based on the selected mode.
    """
    msg = None
    missing, extra = validate_nurse_data(df1, df2)
    # if missing, found in df1 but not in df2
    # if extra, found in df2 but not in df1
    if exact_match:
        if missing or extra:
            msg = [f"⚠️ Mismatch between {file1_name} and {file2_name}:\n"]
            if missing:
                msg.append(
                    f"     • Not found in {file2_name}: {', '.join(sorted(missing))}\n"
                )
            if extra:
                msg.append(
                    f"     • Not found in {file1_name}: {', '.join(sorted(extra))}"
                )
            msg = "\n".join(msg)
            raise InputMismatchError(msg)

    else:
        if extra:
            msg = [f"⚠️ Extra names in {file2_name} not found in {file1_name}:\n"]
            msg.append(f"     • {', '.join(sorted(extra))}\n")
            msg = "\n".join(msg)
            raise InputMismatchError(msg)

        if missing:
            msg = [
                f"Note: {file1_name!r} has {len(missing)} entries with no data in {file2_name!r}.\n"
            ]
            msg.append(f"     • {', '.join(sorted(missing))}\n")
            msg.append("They will be treated as having no data in the schedule.\n")
            msg = "\n".join(msg)
            return msg


def validate_nurse_data(df1: pd.DataFrame, df2: pd.DataFrame):
    """Validates that the nurse names in two DataFrames match."""
    df1_names = set(df1["Name"].str.strip())
    if df2.empty:
        df2_names = set()
    else:
        # Convert index to string if needed
        if not (isinstance(df2.index, pd.Index) and df2.index.dtype.kind in "OS"):
            df2.index = df2.index.astype(str)
        df2_names = set(df2.index.str.strip())
    missing = df1_names - df2_names  # If found in df1 but not in df2
    extra = df2_names - df1_names  # If found in df2 but not in df1
    return missing, extra


# Returns sets of names that are missing or extra, if none returns empty sets


def validate_input_params(
    profiles_df: pd.DataFrame,
    num_days: int,
    min_nurses_per_shift: int,
    min_seniors_per_shift: int,
    max_working_hours: int,
    preferred_work_hours: int,
    min_acceptable_working_hours: int,
):
    """Validate input parameters."""
    nurse_amt = len(set(profiles_df["Name"].str.strip()))
    senior_amt = len(get_senior_set(profiles_df))
    errors = []

    if nurse_amt < min_nurses_per_shift:
        errors.append(
            f" • Number of nurses ({nurse_amt}) must be greater than or equal to min nurses per shift ({min_nurses_per_shift}).\n"
        )

    if senior_amt < min_seniors_per_shift:
        errors.append(
            f" • Number of seniors ({senior_amt}) must be greater than or equal to min seniors per shift ({min_seniors_per_shift}).\n"
        )

    if num_days < 1:
        errors.append(" • End date must be after or same as start date.\n")

    if preferred_work_hours > max_working_hours:
        errors.append(
            " • Preferred work hours must be less than or equal to max working hours.\n"
        )

    if preferred_work_hours < min_acceptable_working_hours:
        errors.append(
            " • Preferred work hours must be greater than or equal to min acceptable working hours.\n"
        )

    if min_acceptable_working_hours > max_working_hours:
        errors.append(
            " • Min acceptable working hours must be less than or equal to max working hours.\n"
        )

    if errors:
        errors.insert(0, "Recheck your inputs:\n")

    return errors

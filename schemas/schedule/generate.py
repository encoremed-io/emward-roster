from pydantic import BaseModel, model_validator, Field, ConfigDict
from typing import List, Optional, Any
from datetime import date
import datetime as dt
from utils.constants import *


# Define data models
class NurseProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    title: str
    years_experience: int

    @model_validator(mode="before")
    @classmethod
    def extract_years_experience(cls, values: Any) -> Any:
        """
        Model validator to extract years_experience from other keys in the input data if not present.

        This validator is needed to handle the case where the column name for years of experience is not exactly "years_experience", e.g. "Years of Experience", "Year(s) Experience", etc.

        If the "years_experience" key is not present, the validator iterates over all the keys in the input data and checks if the key contains the words "year" or "experience". If a matching key is found, the value associated with that key is moved to the "years_experience" key, and the original key is removed from the input data.
        """
        if "years_experience" not in values:
            for key in list(values.keys()):
                lowered = key.lower()
                if "year" in lowered or "experience" in lowered:
                    values["years_experience"] = values.pop(key)
                    break
        return values


class NursePreference(BaseModel):
    model_config = ConfigDict(extra="allow")

    nurse: str
    date: date
    shift: str
    timestamp: dt.datetime


class NurseTraining(BaseModel):
    model_config = ConfigDict(extra="allow")

    nurse: str
    date: date
    training: str


class PrevSchedule(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: str  # <nurse>
    # <date>: <shift> fields will be handled via internal logic


class FixedAssignment(BaseModel):
    model_config = ConfigDict(extra="allow")

    nurse: str
    date: date
    fixed: str


class ShiftDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    shiftType: str
    maxWorkingShift: int
    restDayEligible: int


class ScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    start_date: date
    num_days: int
    shift_durations: List[int] = Field(default=SHIFT_DURATIONS)
    min_nurses_per_shift: int = Field(default=MIN_NURSES_PER_SHIFT)
    min_seniors_per_shift: int = Field(default=MIN_SENIORS_PER_SHIFT)
    max_weekly_hours: int = Field(default=MAX_WEEKLY_HOURS)
    preferred_weekly_hours: int = Field(default=PREFERRED_WEEKLY_HOURS)
    pref_weekly_hours_hard: bool = False
    min_acceptable_weekly_hours: int = Field(default=MIN_ACCEPTABLE_WEEKLY_HOURS)
    min_weekly_rest: int = Field(default=MIN_WEEKLY_REST)
    activate_am_cov: bool = True
    am_coverage_min_percent: int = Field(default=AM_COVERAGE_MIN_PERCENT)
    am_coverage_min_hard: bool = False
    am_coverage_relax_step: int = Field(default=AM_COVERAGE_RELAX_STEP)
    am_senior_min_percent: int = Field(default=AM_SENIOR_MIN_PERCENT)
    am_senior_min_hard: bool = False
    am_senior_relax_step: int = Field(default=AM_SENIOR_RELAX_STEP)
    weekend_rest: bool = True
    back_to_back_shift: bool = False
    use_sliding_window: bool = False
    shift_balance: bool = False
    priority_setting: str = "50/50"
    fixed_assignments: Optional[List[FixedAssignment]] = None
    shift_details: Optional[List[ShiftDetails]] = None

    @model_validator(mode="after")
    def validate_shift_details(self) -> "ScheduleRequest":
        seen_keys = set()
        seen_shift_types = set()

        if self.shift_details is not None:
            for shift in self.shift_details:
                key = (shift.shiftType, shift.maxWorkingShift)

                # Check 1: no duplicate (shiftType, maxWorkingShift) rules
                if key in seen_keys:
                    raise ValueError(
                        f"Duplicate rule for shiftType {shift.shiftType} and maxWorkingShift {shift.maxWorkingShift}"
                    )
                seen_keys.add(key)

                # Check 2: only one rule per shiftType allowed (optional)
                if shift.shiftType in seen_shift_types:
                    raise ValueError(
                        f"Only one rule per shiftType is allowed. Found duplicate for shiftType {shift.shiftType}"
                    )
                seen_shift_types.add(shift.shiftType)

        return self

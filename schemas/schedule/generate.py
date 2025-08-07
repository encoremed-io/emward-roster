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
    doubleShift: bool
    yearsExperience: int

    @model_validator(mode="before")
    @classmethod
    def extract_yearsExperience(cls, values: Any) -> Any:
        """
        Model validator to extract yearsExperience from other keys in the input data if not present.

        This validator is needed to handle the case where the column name for years of experience is not exactly "yearsExperience", e.g. "Years of Experience", "Year(s) Experience", etc.

        If the "yearsExperience" key is not present, the validator iterates over all the keys in the input data and checks if the key contains the words "year" or "experience". If a matching key is found, the value associated with that key is moved to the "yearsExperience" key, and the original key is removed from the input data.
        """
        if "yearsExperience" not in values:
            for key in list(values.keys()):
                lowered = key.lower()
                if "year" in lowered or "experience" in lowered:
                    values["yearsExperience"] = values.pop(key)
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


class StaffAllocations(BaseModel):
    model_config = ConfigDict(extra="allow")

    seniorStaffAllocation: bool = False
    seniorStaffPercentage: int = Field(default=SENIOR_STAFF_PERCENTAGE)
    seniorStaffAllocationRefinement: bool = False
    # must have a value if seniorStaffAllocationRefinement is True
    seniorStaffAllocationRefinementValue: Optional[int] = None

    @model_validator(mode="after")
    def check_refinement_value(self) -> "StaffAllocations":
        if (
            self.seniorStaffAllocationRefinement
            and self.seniorStaffAllocationRefinementValue is None
        ):
            raise ValueError(
                "seniorStaffAllocationRefinementValue is required when seniorStaffAllocationRefinement is True."
            )
        return self


class ScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    startDate: date
    numDays: int
    shiftDurations: List[int] = Field(default=SHIFT_DURATIONS)
    minNursesPerShift: int = Field(default=MIN_NURSES_PER_SHIFT)
    minSeniorsPerShift: int = Field(default=MIN_SENIORS_PER_SHIFT)
    maxWeeklyHours: int = Field(default=MAX_WEEKLY_HOURS)
    preferredWeeklyHours: int = Field(default=PREFERRED_WEEKLY_HOURS)
    prefWeeklyHoursHard: bool = False
    minAcceptableWeeklyHours: int = Field(default=MIN_ACCEPTABLE_WEEKLY_HOURS)
    minWeeklyRest: int = Field(default=MIN_WEEKLY_REST)
    weekendRest: bool = True
    backToBackShift: bool = False
    useSlidingWindow: bool = False
    shiftBalance: bool = False
    prioritySetting: str = "50/50"
    fixedAssignments: Optional[List[FixedAssignment]] = None
    shiftDetails: Optional[List[ShiftDetails]] = None
    staffAllocation: Optional[StaffAllocations] = None
    allowDoubleShift: bool = False

    # Uncomment if you want to use AM coverage constraints
    # activate_am_cov: bool = True
    # am_coverage_min_percent: int = Field(default=AM_COVERAGE_MIN_PERCENT)
    # am_coverage_min_hard: bool = False
    # am_coverage_relax_step: int = Field(default=AM_COVERAGE_RELAX_STEP)
    # am_senior_min_percent: int = Field(default=AM_SENIOR_MIN_PERCENT)
    # am_senior_min_hard: bool = False
    # am_senior_relax_step: int = Field(default=AM_SENIOR_RELAX_STEP)

    @model_validator(mode="after")
    def validate_shift_details(self) -> "ScheduleRequest":
        seen_keys = set()
        seen_shift_types = set()

        if self.shiftDetails is not None:
            for shift in self.shiftDetails:
                key = (shift.shiftType, shift.maxWorkingShift)

                # Check 1: no duplicate (shiftType, maxWorkingShift) rules
                if key in seen_keys:
                    raise ValueError(
                        f"Duplicate rule for shiftType {shift.shiftType} and maxWorkingShift {shift.maxWorkingShift}"
                    )
                seen_keys.add(key)

                # Check 2: only one rule per shiftType allowed (optional)
                # if shift.shiftType in seen_shift_types:
                #     raise ValueError(
                #         f"Only one rule per shiftType is allowed. Found duplicate for shiftType {shift.shiftType}"
                #     )
                # seen_shift_types.add(shift.shiftType)

        return self

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime, timezone
from schemas.schedule.generate import StaffAllocations, NurseLeave


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class PreferenceShift(BaseModel):
    date: str
    shiftIds: List[int]


class Preferences(BaseModel):
    submissionDate: int = Field(default_factory=now_ts)
    shifts: List[PreferenceShift] = Field(default_factory=list)


class ShiftAssignment(BaseModel):
    id: str
    date: str
    shiftIds: List[str]


class ShiftEntry(BaseModel):
    id: str
    name: str
    duration: str
    minNursesPerShift: int
    minSeniorsPerShift: int
    staffAllocation: Optional[StaffAllocations] = Field(default=None, exclude=True)


class ShiftInfo(BaseModel):
    date: str
    shiftIds: List[str]


class TargetNurse(BaseModel):
    nurseId: str
    isSenior: bool
    isSpecialist: bool
    targetShift: List[ShiftInfo]


class Settings(BaseModel):
    minWeeklyHours: int
    maxWeeklyHours: int
    preferredWeeklyHours: int
    minWeeklyRest: int
    weekendRest: bool
    backToBackShift: bool
    # allowDoubleShift: bool
    # shiftBalance: bool
    # prioritySetting: str


class NurseLeaveSwap(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    date: str


class Nurse(BaseModel):
    nurseId: str
    isSenior: bool
    isSpecialist: bool
    shifts: List[ShiftAssignment]
    leaves: List[NurseLeaveSwap]


class SwapCandidate(BaseModel):
    nurseId: str
    isSenior: bool
    currentHours: int
    violatesMaxHours: bool
    messages: List[str] = []
    penaltyScore: int


class SwapCandidateFeatures(BaseModel):
    nurseId: str
    isSenior: bool
    shiftsThisWeek: int
    recentNightShift: int


class SwapSuggestionRequest(BaseModel):
    targetNurseId: List[TargetNurse]
    settings: Settings
    shifts: List[ShiftEntry]
    roster: List[Nurse]

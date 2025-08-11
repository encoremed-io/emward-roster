from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone


def now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


class PreferenceShift(BaseModel):
    date: str
    shiftTypeId: List[int]


class Preferences(BaseModel):
    submissionDate: int = Field(default_factory=now_ts)
    shifts: List[PreferenceShift] = Field(default_factory=list)


class ShiftAssignment(BaseModel):
    id: int
    date: str
    shiftTypeId: int


class ShiftEntry(BaseModel):
    id: int
    name: str
    duration: str


class ShiftInfo(BaseModel):
    date: str
    shiftTypeId: List[int]


class TargetNurse(BaseModel):
    nurseId: str
    targetShift: List[ShiftInfo]


class Settings(BaseModel):
    shiftDurations: int
    minSeniorsPerShift: int
    maxWeeklyHours: int
    preferredWeeklyHours: int
    minWeeklyHours: int
    enforceWeekendRest: bool
    backToBackShift: bool
    balanceShiftAssignments: bool


class Nurse(BaseModel):
    nurseId: str
    isSenior: bool
    isSpecialist: bool
    preferences: Optional[Preferences] = None
    shifts: List[ShiftAssignment]


class SwapCandidate(BaseModel):
    nurseId: str
    isSenior: bool
    currentHours: int
    violatesMaxHours: bool
    message: str


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

    class Config:
        json_schema_extra = {
            "example": {
                "targetNurseId": [
                    {
                        "nurseId": "N001",
                        "targetShift": [{"date": "2025-07-20", "shiftTypeId": [1, 2]}],
                    }
                ],
                "settings": {
                    "shiftDurations": 8,
                    "minSeniorsPerShift": 2,
                    "maxWeeklyHours": 40,
                    "preferredWeeklyHours": 40,
                    "minWeeklyHours": 30,
                    "enforceWeekendRest": False,
                    "backToBackShift": False,
                    "balanceShiftAssignments": False,
                },
                "shifts": [
                    {"id": 1, "name": "Morning", "duration": "0700-1700"},
                    {"id": 2, "name": "Noon", "duration": "1200-2100"},
                    {"id": 3, "name": "Overnight", "duration": "1700-0700"},
                ],
                "roster": [
                    {
                        "nurseId": "N001",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-15", "shiftTypeId": [3, 2]},
                                {"date": "2025-07-18", "shiftTypeId": [3, 2]},
                                {"date": "2025-07-25", "shiftTypeId": [2]},
                                {"date": "2025-07-21", "shiftTypeId": [3]},
                                {"date": "2025-07-24", "shiftTypeId": [1]},
                            ],
                        },
                        "isSenior": False,
                        "isSpecialist": True,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 2},
                            {"id": 2, "date": "2025-07-16", "shiftTypeId": 2},
                            {"id": 3, "date": "2025-07-17", "shiftTypeId": 2},
                            {"id": 4, "date": "2025-07-18", "shiftTypeId": 3},
                            {"id": 5, "date": "2025-07-19", "shiftTypeId": 3},
                            {"id": 6, "date": "2025-07-20", "shiftTypeId": 3},
                            {"id": 7, "date": "2025-07-21", "shiftTypeId": 3},
                            {"id": 8, "date": "2025-07-22", "shiftTypeId": 2},
                            {"id": 9, "date": "2025-07-24", "shiftTypeId": 1},
                            {"id": 10, "date": "2025-07-25", "shiftTypeId": 1},
                            {"id": 11, "date": "2025-07-26", "shiftTypeId": 2},
                            {"id": 12, "date": "2025-07-27", "shiftTypeId": 1},
                        ],
                    },
                    {
                        "nurseId": "N002",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-17", "shiftTypeId": [1]},
                                {"date": "2025-07-21", "shiftTypeId": [1]},
                                {"date": "2025-07-20", "shiftTypeId": [1]},
                            ],
                        },
                        "isSenior": True,
                        "isSpecialist": True,
                        "shifts": [
                            {"id": 1, "date": "2025-07-15", "shiftTypeId": 2},
                            {"id": 2, "date": "2025-07-17", "shiftTypeId": 3},
                            {"id": 3, "date": "2025-07-18", "shiftTypeId": 2},
                            {"id": 4, "date": "2025-07-19", "shiftTypeId": 3},
                            {"id": 5, "date": "2025-07-20", "shiftTypeId": 3},
                            {"id": 6, "date": "2025-07-21", "shiftTypeId": 3},
                            {"id": 7, "date": "2025-07-22", "shiftTypeId": 2},
                            {"id": 8, "date": "2025-07-23", "shiftTypeId": 2},
                            {"id": 9, "date": "2025-07-24", "shiftTypeId": 3},
                            {"id": 10, "date": "2025-07-25", "shiftTypeId": 3},
                            {"id": 11, "date": "2025-07-26", "shiftTypeId": 1},
                            {"id": 12, "date": "2025-07-27", "shiftTypeId": 2},
                        ],
                    },
                    {
                        "nurseId": "N003",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-21", "shiftTypeId": [1, 2]},
                                {"date": "2025-07-18", "shiftTypeId": [1, 3]},
                                {"date": "2025-07-19", "shiftTypeId": [2]},
                            ],
                        },
                        "isSenior": False,
                        "isSpecialist": True,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 2},
                            {"id": 2, "date": "2025-07-15", "shiftTypeId": 3},
                            {"id": 3, "date": "2025-07-18", "shiftTypeId": 1},
                            {"id": 4, "date": "2025-07-19", "shiftTypeId": 2},
                            {"id": 5, "date": "2025-07-20", "shiftTypeId": 1},
                            {"id": 6, "date": "2025-07-21", "shiftTypeId": 1},
                            {"id": 7, "date": "2025-07-22", "shiftTypeId": 3},
                            {"id": 8, "date": "2025-07-23", "shiftTypeId": 2},
                            {"id": 9, "date": "2025-07-24", "shiftTypeId": 1},
                            {"id": 10, "date": "2025-07-26", "shiftTypeId": 1},
                        ],
                    },
                    {
                        "nurseId": "N004",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-14", "shiftTypeId": [1, 2]},
                                {"date": "2025-07-18", "shiftTypeId": [1, 3]},
                                {"date": "2025-07-25", "shiftTypeId": [1, 2]},
                            ],
                        },
                        "isSenior": False,
                        "isSpecialist": True,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 2},
                            {"id": 2, "date": "2025-07-15", "shiftTypeId": 3},
                            {"id": 3, "date": "2025-07-17", "shiftTypeId": 2},
                            {"id": 4, "date": "2025-07-18", "shiftTypeId": 1},
                            {"id": 5, "date": "2025-07-19", "shiftTypeId": 2},
                            {"id": 6, "date": "2025-07-20", "shiftTypeId": 3},
                            {"id": 7, "date": "2025-07-21", "shiftTypeId": 2},
                            {"id": 8, "date": "2025-07-22", "shiftTypeId": 3},
                            {"id": 9, "date": "2025-07-24", "shiftTypeId": 3},
                            {"id": 10, "date": "2025-07-25", "shiftTypeId": 2},
                            {"id": 11, "date": "2025-07-26", "shiftTypeId": 3},
                        ],
                    },
                    {
                        "nurseId": "N005",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-21", "shiftTypeId": [3, 1]},
                                {"date": "2025-07-20", "shiftTypeId": [2, 1]},
                                {"date": "2025-07-23", "shiftTypeId": [2]},
                                {"date": "2025-07-27", "shiftTypeId": [2, 1]},
                                {"date": "2025-07-14", "shiftTypeId": [1, 2]},
                            ],
                        },
                        "isSenior": True,
                        "isSpecialist": False,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 1},
                            {"id": 2, "date": "2025-07-15", "shiftTypeId": 2},
                            {"id": 3, "date": "2025-07-16", "shiftTypeId": 2},
                            {"id": 4, "date": "2025-07-17", "shiftTypeId": 2},
                            {"id": 5, "date": "2025-07-20", "shiftTypeId": 3},
                            {"id": 6, "date": "2025-07-21", "shiftTypeId": 1},
                            {"id": 7, "date": "2025-07-22", "shiftTypeId": 1},
                            {"id": 8, "date": "2025-07-24", "shiftTypeId": 2},
                            {"id": 9, "date": "2025-07-25", "shiftTypeId": 3},
                            {"id": 10, "date": "2025-07-27", "shiftTypeId": 1},
                        ],
                    },
                    {
                        "nurseId": "N006",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-21", "shiftTypeId": [3, 1]},
                                {"date": "2025-07-20", "shiftTypeId": [3, 1]},
                                {"date": "2025-07-14", "shiftTypeId": [2, 3]},
                                {"date": "2025-07-22", "shiftTypeId": [1, 2]},
                                {"date": "2025-07-17", "shiftTypeId": [1, 3]},
                            ],
                        },
                        "isSenior": False,
                        "isSpecialist": True,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 1},
                            {"id": 2, "date": "2025-07-17", "shiftTypeId": 3},
                            {"id": 3, "date": "2025-07-18", "shiftTypeId": 2},
                            {"id": 4, "date": "2025-07-22", "shiftTypeId": 2},
                            {"id": 5, "date": "2025-07-23", "shiftTypeId": 3},
                            {"id": 6, "date": "2025-07-24", "shiftTypeId": 1},
                            {"id": 7, "date": "2025-07-26", "shiftTypeId": 3},
                            {"id": 8, "date": "2025-07-27", "shiftTypeId": 3},
                        ],
                    },
                    {
                        "nurseId": "N007",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-24", "shiftTypeId": [1, 2]},
                                {"date": "2025-07-25", "shiftTypeId": [2]},
                                {"date": "2025-07-20", "shiftTypeId": [3]},
                            ],
                        },
                        "isSenior": True,
                        "isSpecialist": False,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 3},
                            {"id": 2, "date": "2025-07-15", "shiftTypeId": 2},
                            {"id": 3, "date": "2025-07-16", "shiftTypeId": 1},
                            {"id": 4, "date": "2025-07-17", "shiftTypeId": 3},
                            {"id": 5, "date": "2025-07-21", "shiftTypeId": 1},
                            {"id": 6, "date": "2025-07-22", "shiftTypeId": 2},
                            {"id": 7, "date": "2025-07-23", "shiftTypeId": 2},
                            {"id": 8, "date": "2025-07-24", "shiftTypeId": 3},
                            {"id": 9, "date": "2025-07-25", "shiftTypeId": 1},
                            {"id": 10, "date": "2025-07-26", "shiftTypeId": 1},
                            {"id": 11, "date": "2025-07-27", "shiftTypeId": 2},
                        ],
                    },
                    {
                        "nurseId": "N008",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-16", "shiftTypeId": [2]},
                                {"date": "2025-07-26", "shiftTypeId": [2]},
                                {"date": "2025-07-24", "shiftTypeId": [2, 3]},
                                {"date": "2025-07-15", "shiftTypeId": [3, 1]},
                                {"date": "2025-07-14", "shiftTypeId": [1, 3]},
                                {"date": "2025-07-19", "shiftTypeId": [3]},
                            ],
                        },
                        "isSenior": False,
                        "isSpecialist": True,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 2},
                            {"id": 2, "date": "2025-07-15", "shiftTypeId": 1},
                            {"id": 3, "date": "2025-07-16", "shiftTypeId": 2},
                            {"id": 4, "date": "2025-07-17", "shiftTypeId": 1},
                            {"id": 5, "date": "2025-07-18", "shiftTypeId": 2},
                            {"id": 6, "date": "2025-07-21", "shiftTypeId": 2},
                            {"id": 7, "date": "2025-07-22", "shiftTypeId": 1},
                            {"id": 8, "date": "2025-07-23", "shiftTypeId": 2},
                            {"id": 9, "date": "2025-07-24", "shiftTypeId": 2},
                            {"id": 10, "date": "2025-07-25", "shiftTypeId": 2},
                            {"id": 11, "date": "2025-07-26", "shiftTypeId": 2},
                            {"id": 12, "date": "2025-07-27", "shiftTypeId": 3},
                        ],
                    },
                    {
                        "nurseId": "N009",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-23", "shiftTypeId": [2, 1]},
                                {"date": "2025-07-27", "shiftTypeId": [2, 3]},
                                {"date": "2025-07-26", "shiftTypeId": [3]},
                            ],
                        },
                        "isSenior": False,
                        "isSpecialist": True,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 3},
                            {"id": 2, "date": "2025-07-15", "shiftTypeId": 3},
                            {"id": 3, "date": "2025-07-16", "shiftTypeId": 2},
                            {"id": 4, "date": "2025-07-17", "shiftTypeId": 1},
                            {"id": 5, "date": "2025-07-18", "shiftTypeId": 1},
                            {"id": 6, "date": "2025-07-21", "shiftTypeId": 3},
                            {"id": 7, "date": "2025-07-22", "shiftTypeId": 3},
                            {"id": 8, "date": "2025-07-23", "shiftTypeId": 2},
                            {"id": 9, "date": "2025-07-26", "shiftTypeId": 1},
                            {"id": 10, "date": "2025-07-27", "shiftTypeId": 3},
                        ],
                    },
                    {
                        "nurseId": "N010",
                        "preferences": {
                            "submissionDate": 1753165193,
                            "shifts": [
                                {"date": "2025-07-26", "shiftTypeId": [1, 2]},
                                {"date": "2025-07-15", "shiftTypeId": [2, 1]},
                                {"date": "2025-07-20", "shiftTypeId": [1]},
                                {"date": "2025-07-27", "shiftTypeId": [2, 1]},
                            ],
                        },
                        "isSenior": True,
                        "isSpecialist": True,
                        "shifts": [
                            {"id": 1, "date": "2025-07-14", "shiftTypeId": 3},
                            {"id": 2, "date": "2025-07-15", "shiftTypeId": 2},
                            {"id": 3, "date": "2025-07-16", "shiftTypeId": 1},
                            {"id": 4, "date": "2025-07-17", "shiftTypeId": 3},
                            {"id": 5, "date": "2025-07-20", "shiftTypeId": 1},
                            {"id": 6, "date": "2025-07-21", "shiftTypeId": 3},
                            {"id": 7, "date": "2025-07-22", "shiftTypeId": 1},
                            {"id": 8, "date": "2025-07-23", "shiftTypeId": 2},
                            {"id": 9, "date": "2025-07-24", "shiftTypeId": 2},
                            {"id": 10, "date": "2025-07-25", "shiftTypeId": 1},
                            {"id": 11, "date": "2025-07-26", "shiftTypeId": 3},
                            {"id": 12, "date": "2025-07-27", "shiftTypeId": 3},
                        ],
                    },
                    {
                        "nurseId": "N011",
                        "isSenior": False,
                        "isSpecialist": False,
                        "shifts": [],
                    },
                    {
                        "nurseId": "N012",
                        "isSenior": False,
                        "isSpecialist": False,
                        "shifts": [{"id": 1, "date": "2025-07-16", "shiftTypeId": 1}],
                    },
                ],
            }
        }

from pydantic import BaseModel
from typing import List


class PreferenceShift(BaseModel):
    date: str  # Date of the shift
    shiftTypeId: List[int]  # List of shift type IDs for that date


class CandidatesPreferences(BaseModel):
    shifts: List[PreferenceShift]  # List of shifts the nurse prefers


class CandidatesTrainingRecord(BaseModel):
    nurseId: str  # Unique identifier for the nurse
    isSenior: bool  # If the nurse is a senior
    isSpecialist: bool  # If the nurse is a specialist
    preferences: CandidatesPreferences  # Nurse preferences
    shiftsThisWeek: int  # How many shifts they've already worked this week
    recentNightShift: bool  # If they worked a night shift recently
    totalHoursThisWeek: int  # Total hours worked this week
    consecutiveDaysWorked: int  # How many days worked in a row
    dayAfterOffDay: bool  # If the shift is the day after an off day
    wasChosen: bool  # If this candidate was chosen in the past

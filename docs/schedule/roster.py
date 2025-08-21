schedule_roster_description = """
Generate a schedule based on the given nurse profiles, shift preferences, and other parameters

### Request Body

The API endpoint takes the following parameters:

- `profiles`: List of `NurseProfile` objects, which contain the following information:
    - `id`: Primary key of the nurse

    - `name`: Name of the nurse
    - `title`: Title of the nurse (e.g. "Senior Nurse", "Junior Nurse")
    - `yearsExperience`: Number of years of experience the nurse has
    - `doubleShift`: Whether the nurse can work double shifts (True/False)

- `preferences`: List of `NursePreference` objects, which contain the following information: (Optional)
    - `id`: Primary key of the nurse

    - `nurse`: Name of the nurse
    - `date`: Date of the shift
    - `shiftId`: Shift id
    - `timestamp`: Timestamp of the preference

- `trainingShifts`: List of `NurseTraining` objects, which contain the following information: (Optional)
    - `id`: Primary key of the nurse

    - `nurse`: Name of the nurse
    - `date`: Date of the training shift
    - `shiftId`: Shift id

- `previousSchedule`: List of `PrevSchedule` objects. Each object represents a nurse's past schedule and contains: (Optional)
    - `id`: Primary key of the nurse

    - `nurse`: Name of the nurse
    - `schedule`: List of `PrevScheduleItem` objects, which contain the following information:
        - `date`: Date of the training shift

        - `shiftId`: Shift id
        - `shift`: Name of the shift

- `leaves`: List of `NurseLeave` objects, which contain the following information: (Optional)
    - `id`: Primary key of the nurse

    - `nurse`: Name of the nurse
    - `date`: Date of the leave
    - `leaveId`: Leave id
    - `leaveName`: Leave name

- `shifts`: List of `Shifts` objects, which contain the following information:
    - `id`: Primary key of the shift

    - `name`: Name of the shift
    - `duration`: Duration of the shift
    - `minNursesPerShift`: Minimum number of nurses per shift
    - `minSeniorsPerShift`: Minimum number of senior nurses per shift
    - `staffAllocation`: Optional `StaffAllocations` object, which contains the following information:
        - `seniorStaffAllocation`: Whether to allocate senior staff

        - `seniorStaffPercentage`: Percentage of senior staff required per shift
        - `seniorStaffAllocationRefinement`: Whether to refine the senior staff allocation
        - `seniorStaffAllocationRefinementValue`: Value for refining the senior staff allocation
            
- `request` (`ScheduleRequest` object):
    Contains the following information:
    - `startDate`: Start date of the schedule

    - `numDays`: Number of days in the schedule
    - `minWeeklyHours`: Minimum weekly hours for each nurse
    - `maxWeeklyHours`: Maximum weekly hours for each nurse
    - `preferredWeeklyHours`: Preferred weekly hours for each nurse
    - `minWeeklyRest`: Minimum number of rest days per week
    - `weekendRest`: Whether to ensure that each nurse has a weekend rest
    - `backToBackShift`: Whether to prevent back-to-back shifts
    - `shiftBalance`: Whether to balance the number of shifts between nurses
    - `prioritySetting`: Priority setting for the solver (e.g. "Fairness", "Fairness-leaning", "50/50", "Preference-leaning", "Preference"). Only activated when `shiftBalance` is `True`.
    - `allowDoubleShift`: Whether to allow double shifts
    - `shiftDetails`: List of shift details rule
        - `shiftId`: Primary key of the shift

        - `maxWorkingShift`: Max of the working shift type continuously
        - `restDayEligible`: Number of rests day after continuously working shift type

        
The API endpoint returns a JSON object with the following keys:

- `schedule`: List of shift assignments, where each assignment is a dictionary with the following keys:
    - `nurse`: Name of the nurse

    - `date`: Date of the shift
    - `shift`: Shift assignment (e.g. "AM", "PM", "Night")
- `summary`: List of summary statistics, where each statistic is a dictionary with the following keys:
    - `metric`: Name of the metric (e.g. "Hours_Week1_Real", "Prefs_Unmet")

    - `value`: Value of the metric
- `violations`: List of constraint violations, where each violation is a dictionary with the following keys:
    - `constraint`: Name of the constraint (e.g. "Low Hours Nurses", "Low Senior AM Days")

    - `value`: Value of the constraint
    - `metrics`: A dictionary containing evaluation metrics. Keys include "Preference Unmet" and "Fairness Gap", with each value representing the corresponding metric score.

The API endpoint raises an HTTPException with a status code of 400 if the input is invalid and raises an HTTPException with a status code of 422 if no feasible solution is found.
"""

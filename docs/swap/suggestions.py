swap_suggestions_description = """
Suggest replacement candidates or direct swap options for nurses who are taking leave.

### Request Body

- `targetNurseId` (List):
    A list of nurses who are requesting leave and their affected shifts.
    ```json
    [
        {
            "nurseId": "N001",
            "isSenior": false,
            "isSpecialist": false,
            "targetShift": [
                {
                    "date": "2025-07-25",
                    "shiftIds": ["1", "2"]
                }
            ]
        }
    ]
    ```

- `settings` (Object):
    Configuration for swap rules and constraints:
    - `minNursesPerShift`: Minimum number of nurses per shift

    - `minSeniorsPerShift`: Minimum number of senior nurses per shift
    - `minWeeklyHours`: Minimum allowable hours per nurse per week
    - `maxWeeklyHours`: Maximum allowable hours per nurse per week
    - `preferredWeeklyHours`: Target hours per week per nurse
    - `minWeeklyRest`: Minimum number of rest days per week
    - `weekendRest`: Whether to ensure that each nurse has a weekend rest
    - `backToBackShift`: Whether to disallow back-to-back shifts
    - `allowDoubleShift`: Whether to allow double shifts
    - `shiftBalance`: Whether to balance the number of shifts between nurses
    - `prioritySetting`: Priority setting for the solver (e.g. "Fairness", "Fairness-leaning", "50/50", "Preference-leaning", "Preference"). Only activated when `shiftBalance` is `True`.
    - Other optional constraints may apply.

- `shifts` (Array):
    Definitions of available shifts:
    ```json
    [
        { 
          "id": 1, 
          "name": "AM", 
          "duration": "0700-1400",
          "staffAllocation": {
            "seniorStaffAllocation": true,
            "seniorStaffPercentage": 50,
            "seniorStaffAllocationRefinement": true,
            "seniorStaffAllocationRefinementValue": 5
          }
        },
        { 
          "id": 2, 
          "name": "PM", 
          "duration": "1400-2100",
        },
        { 
          "id": 3, 
          "name": "Night", 
          "duration": "2100-0700",
        }
    ]
    ```

- `roster` (Array):
    Current list of nurses with assigned shifts:
    ```json
    [
        {
            "nurseId": "N001",
            "isSenior": false,
            "isSpecialist": false,
            "shifts": [
                { 
                  "id": "1", 
                  "date": "2025-07-24", 
                  "shiftIds": ["2"] 
                }
            ]
        }
    ]
    ```

---

### Response Format

Returns a list of possible candidates for each leave-affected shift:
```json
{
  "results": [
    {
      "originalNurse": "N001",
      "replacementFor": {
        "date": "2025-07-25",
        "shiftTypeId": 1
      },
      "filterLevel": "strict",
      "topCandidates": [
        {
          "nurseId": "N005",
          "isSenior": false,
          "currentHours": 32,
          "violatesMaxHours": false,
          "message": null
        }
      ],
      "directSwapCandidate": {
        "nurseId": "N012",
        "swapFrom": {
          "id": 1,
          "date": "2025-07-16",
          "shiftTypeId": 1
        },
        "swapTo": {
          "date": "2025-07-14",
          "shiftTypeId": 1
        },
        "note": "Same-shift direct swap"
      }
    }
  ]
}
"""

swap_suggestions_description = """
Suggest replacement candidates or direct swap options for nurses who are taking leave.

### Request Body

- `targetNurseId` (List):
    A list of nurses who are requesting leave and their affected shifts.
    ```json
    [
        {
            "nurseId": "N001",
            "targetShift": [
                {
                    "date": "2025-07-25",
                    "shiftTypeId": [1, 2]
                }
            ]
        }
    ]
    ```

- `settings` (Object):
    Configuration for swap rules and constraints:
    - `shiftDurations`: Duration of each shift in hours (e.g. 8)
    - `minSeniorsPerShift`: Minimum required senior nurses per shift
    - `maxWeeklyHours`: Maximum allowable hours per nurse per week
    - `preferredWeeklyHours`: Target hours per week per nurse
    - `backToBackShift`: Whether to disallow back-to-back shifts
    - Other optional constraints may apply.

- `shifts` (Array):
    Definitions of available shifts:
    ```json
    [
        { "id": 1, "name": "Morning", "duration": "0700-1700" },
        { "id": 2, "name": "Noon", "duration": "1200-2100" },
        { "id": 3, "name": "Overnight", "duration": "1700-0700" }
    ]
    ```

- `roster` (Array):
    Current list of nurses with assigned shifts:
    ```json
    [
        {
            "nurseId": "N001",
            "role": "senior",
            "shifts": [
                { "date": "2025-07-24", "shiftTypeId": 1 }
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
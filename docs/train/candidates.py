train_candidates_description = """
Train replacement candidates who are taking leaves.

### Request Body

-   A list of nurses who requested leave and their shift details.
    ```json
    [
        {
            "nurseId": "N001",
            "isSenior": true,
            "isSpecialist": false,
            "preferences": {
                "shifts": [
                    { "date": "2025-07-24", "shiftTypeId": [1, 2] },
                    { "date": "2025-07-25", "shiftTypeId": [3] }
                ]
            },
            "shiftsThisWeek": 2,
            "recentNightShift": false,
            "totalHoursThisWeek": 16,
            "consecutiveDaysWorked": 5,
            "dayAfterOffDay": false,
            "wasChosen": true,
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
      "message": "Model trained and saved successfully.",
      "samplesTrainedOn": 20,
      "featureImportances": {
        "isSenior": 0.32,
        "isSpecialist": 0.10,
        "preferences": 0.18,
        "shiftsThisWeek": 0.05,
        "recentNightShift": 0.12,
        "totalHoursThisWeek": 0.09,
        "consecutiveDaysWorked": 0.08,
        "dayAfterOffDay": 0.06
    },
  ]
}
"""

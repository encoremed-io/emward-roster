from dataclasses import dataclass
from ortools.sat.python import cp_model
from typing import Any

@dataclass
class HardRule:
    flag: Any
    message: str


def define_hard_rules(model: cp_model.CpModel) -> dict[str, HardRule]:
    return {
        "Min nurses": HardRule(
            model.NewBoolVar("assume_min_nurses"),
            "Minimum nurses per shift cannot be met."
        ),
        "Min seniors": HardRule(
            model.NewBoolVar("assume_min_seniors"),
            "Minimum seniors per shift cannot be met."
        ),
        "Max weekly hours": HardRule(
            model.NewBoolVar("assume_max_weekly"),
            "Maximum working hours per week cannot be guaranteed. Relaxing sliding window may help."
        ),
        "Min weekly hours": HardRule(
            model.NewBoolVar("assume_min_weekly"),
            "Minimum working hours per week cannot be enforced."
        ),
        "AM cov PM": HardRule(
            model.NewBoolVar("assume_am_cov_pm"),
            "AM coverage cannot always exceed PM coverage."
        ),
        "AM cov Night": HardRule(
            model.NewBoolVar("assume_am_cov_night"),
            "AM coverage cannot always exceed Night coverage."
        ),
        "AM snr PM": HardRule(
            model.NewBoolVar("assume_am_snr_pm"),
            "AM senior coverage cannot always exceed senior PM coverage."
        ),
        "AM snr Night": HardRule(
            model.NewBoolVar("assume_am_snr_night"),
            "AM senior coverage cannot always exceed senior Night coverage."
        ),
        "AM snr majority": HardRule(
            model.NewBoolVar("assume_am_snr_majority"),
            "Number of seniors on AM shift cannot always be more than juniors."
        ),
        "Weekend rest": HardRule(
            model.NewBoolVar("assume_weekend_rest"),
            "Alternating weekend rest cannot be guaranteed."
        ),
        "No b2b": HardRule(
            model.NewBoolVar("assume_no_b2b"),
            "Consecutive shift enforcement cannot be done."
        ),
        # Add others as needed
    }

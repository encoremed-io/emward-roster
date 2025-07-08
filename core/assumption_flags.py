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
            "Minimum nurses per shift cannot be met.\n"
        ),
        "Min seniors": HardRule(
            model.NewBoolVar("assume_min_seniors"),
            "Minimum seniors per shift cannot be met.\n"
        ),
        "Max weekly hours": HardRule(
            model.NewBoolVar("assume_max_weekly"),
            "Maximum working hours per week cannot be guaranteed. Relaxing sliding window may help.\n"
        ),
        "Pref weekly hours": HardRule(    
            model.NewBoolVar("assume_pref_weekly"),
            "Preferred weekly hours per week cannot be enforced.\n"
        ),
        "Min weekly hours": HardRule(
            model.NewBoolVar("assume_min_weekly"),
            "Minimum working hours per week cannot be enforced.\n"
        ),
        "AM cov min": HardRule(
            model.NewBoolVar("assume_am_cov_min"),
            "Minimum AM coverage cannot be met.\n"
        ),
        "AM cov majority": HardRule(
            model.NewBoolVar("assume_am_cov_majority"),
            "Number of nurses on AM shift cannot always be more than PM and Night respectively.\n"
        ),
        "AM snr min": HardRule(
            model.NewBoolVar("assume_am_snr_min"),
            "Minimum AM senior coverage cannot be met.\n"
        ),
        "AM snr majority": HardRule(
            model.NewBoolVar("assume_am_snr_majority"),
            "Number of seniors on AM shift cannot always be more than juniors.\n"
        ),
        "Weekend rest": HardRule(
            model.NewBoolVar("assume_weekend_rest"),
            "Alternating weekend rest cannot be guaranteed.\n\n"
        ),
        "No b2b": HardRule(
            model.NewBoolVar("assume_no_b2b"),
            "Consecutive shift enforcement cannot be done.\n"
        ),
        # Add others as needed
    }

from ortools.sat.python import cp_model
from typing import Dict, List, Tuple, Union
from utils.constants import SHIFT_LABELS

def add_rl_hints(
    model: cp_model.CpModel,
    work: Dict[Tuple[str, int, int], cp_model.IntVar],
    nurse_names: List[str],
    num_days: int,
    rl_assignment: Union[
        Dict[Tuple[str, int], int],
        List[int]
    ],
    add_zero_hints: bool = False
) -> None:
    """
    Add warm-start hints to a CP-SAT model from an RL assignment.

    rl_assignment can be:
      • dict[(n, d)] -> s                     		# shift index
      • list of length N*num_days            		# shift index per nurse-day
      • list of length N*num_days*shift_types 		# binary 0/1 per nurse-day-shift

    If add_zero_hints is True, all other (n, d, s) combinations will be hinted as 0.
    """
    N = len(nurse_names)
    D = num_days
    shift_types = len(SHIFT_LABELS)

    # 1) Normalize into dict[(n, d)] -> s
    if isinstance(rl_assignment, dict):
        hints = rl_assignment
    elif isinstance(rl_assignment, list):
        L = len(rl_assignment)
        if L == N * D:
            # one shift index per nurse-day
            hints = {
                (nurse_names[i // D], i % D): s
                for i, s in enumerate(rl_assignment)
            }
        elif L == N * D * shift_types:
            # one-hot list: bit=1 means that (n,d,s) should be on
            hints = {}
            for idx, bit in enumerate(rl_assignment):
                if not bit:
                    continue
                n_idx, rem = divmod(idx, D * shift_types)
                d, s       = divmod(rem, shift_types)
                hints[(nurse_names[n_idx], d)] = s
        else:
            raise ValueError(f"rl_assignment list has invalid length {L}")
    else:
        raise ValueError(f"rl_assignment must be dict or list, got {type(rl_assignment)}")

    # 2) Apply hints: track which (n,d,s) we set to 1
    hinted_ones = set()
    for (n, d), s in hints.items():
        model.AddHint(work[(n, d, s)], 1)
        hinted_ones.add((n, d, s))

    # 3) Optionally hint zeros for all other vars
    if add_zero_hints:
        for n in nurse_names:
            for d in range(num_days):
                for s in range(shift_types):
                    if (n, d, s) not in hinted_ones:
                        model.AddHint(work[(n, d, s)], 0)

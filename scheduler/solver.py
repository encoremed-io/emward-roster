from dataclasses import dataclass
from ortools.sat.python import cp_model
import logging
from typing import Any, Dict, Tuple
from exceptions.custom_errors import NoFeasibleSolutionError

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    def __init__(
        self,
        solver: cp_model.CpSolver,
        status: Any,
        cached_values: Dict[Tuple[str, int, int], int],
        high_penalty: float,
        low_penalty: float,
        fairness_gap: Any,
    ):
        """
        Initialize a SolverResult instance.

        Args:
            solver (cp_model.CpSolver): The CP solver used for solving the model.
            status (Any): The status of the solver after the solving attempt.
            cached_values (Dict[Tuple[str, int, int], int]): Cached values of the solution variables.
            high_penalty (float): The penalty associated with high priority constraints.
            low_penalty (float): The penalty associated with low priority constraints.
            fairness_gap (Any): The gap value representing the fairness of meeting preferences.
        """
        self.solver = solver
        self.status = status
        self.cached_values = cached_values
        self.high_penalty = high_penalty
        self.low_penalty = low_penalty
        self.fairness_gap = fairness_gap


def configure_solver(timeout: float = 600.0, seed: int = 42) -> cp_model.CpSolver:
    """Configure the CP solver."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.random_seed = seed
    solver.parameters.relative_gap_limit = 0.01
    solver.parameters.num_search_workers = 8
    solver.parameters.randomize_search = True
    solver.parameters.log_search_progress = False
    return solver


def get_model_size(model: cp_model.CpModel) -> Tuple[int, int]:
    """Get the number of constraints and boolean variables in the model."""
    proto = model.Proto()
    num_constraints = len(proto.constraints)
    num_bool_vars = len(proto.variables)
    return num_constraints, num_bool_vars


def run_phase1(model, state) -> SolverResult:
    """
    Run Phase 1 of the solver.

    Phase 1 has two steps: 1A and 1B. In Phase 1A, the solver first checks if there is a feasible solution
    by maximizing the number of satisfied hard rules. If a feasible solution is found, Phase 1B is run to
    minimize the total penalty of high priority constraints.

    Args:
        model (cp_model.CpModel): The CP model.
        state (ScheduleState): The state of the scheduling problem.

    Returns:
        SolverResult: A SolverResult instance containing the solver, status, cached values, high priority
            penalty, low priority penalty, and fairness gap.
    """
    # Phase 1A: Check feasibility
    logger.info("🚀 Phase 1A: checking feasibility...")
    # 1. Tell the model to minimize penalty sum
    model.Maximize(sum(r.flag for r in state.hard_rules.values()))

    # debug: print model size
    num_constraints, num_bool_vars = get_model_size(model)
    logger.info(f"→ #constraints_p1 = {num_constraints},  #bool_vars = {num_bool_vars}")

    solver = configure_solver()
    status = solver.Solve(model)

    total_hards = len(state.hard_rules)
    logger.info(f"Total hard constraint flags: {total_hards}")
    satisfied_hards = int(solver.ObjectiveValue())
    logger.info(f"Satisfied hard constraint flags: {satisfied_hards}")
    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
    if satisfied_hards != total_hards or status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        dropped = [
            r.message for r in state.hard_rules.values() if solver.Value(r.flag) == 0
        ]
        error_msg = ["❌ No feasible solution. Identified issues:\n"]
        error_msg.append("\n".join(f"    • {m.strip()}" for m in dropped))
        error_msg = "\n".join(error_msg)
        logger.info("⚠️ No feasible solution found with minimal constraints.")
        raise NoFeasibleSolutionError(error_msg)

    logger.info("✅ Feasible solution found with minimal constraints.")

    # Phase 1B: Minimise high priority penalties
    logger.info("🚀 Phase 1B: minimising penalties...")
    model.Add(sum(r.flag for r in state.hard_rules.values()) == total_hards)
    model.Minimize(sum(state.high_priority_penalty))

    # debug: print model size
    num_constraints, num_bool_vars = get_model_size(model)
    logger.info(f"→ #constraints_p1 = {num_constraints},  #bool_vars = {num_bool_vars}")

    solver = configure_solver()
    status = solver.Solve(model)
    high_penalty = solver.ObjectiveValue()
    low_penalty = solver.Value(sum(state.low_priority_penalty))

    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
    logger.info(f"High Priority Penalty Phase 1B: {high_penalty}")
    logger.info(f"Low Priority Penalty Phase 1B: {low_penalty}")

    cached = {
        (n, d, s): solver.Value(state.work[n, d, s])
        for n in state.nurse_names
        for d in range(state.num_days)
        for s in range(state.shift_types)
    }

    fairness_gap = solver.Value(state.gap_pct) if state.gap_pct is not None else None
    logger.info(
        f"▶️ Phase 1 complete: best total penalty = {high_penalty + low_penalty}; best fairness gap = {fairness_gap if fairness_gap is not None else 'N/A'}"
    )
    return SolverResult(solver, status, cached, high_penalty, low_penalty, fairness_gap)


def run_phase2(model, state, p1: SolverResult) -> SolverResult:
    """
    Phase 2: Minimize low-priority penalties (preferences/fairness/imbalance)
    while keeping the high-priority cost fixed at the Phase-1 optimum.
    """
    logger.info("🚀 Phase 2: optimizing low-priority penalties...")

    # Keep all hard rules enforced
    model.Add(sum(r.flag for r in state.hard_rules.values()) == len(state.hard_rules))

    # Pin high-priority cost to the optimum found in Phase 1
    high_cost = sum(state.high_priority_penalty) if state.high_priority_penalty else 0
    model.Add(high_cost == round(p1.high_penalty))

    # Only bind fairness if fairness penalties are active
    if getattr(state, "fairness_gap_penalty", 0) and p1.fairness_gap is not None:
        model.Add(state.gap_pct <= int(p1.fairness_gap))

    # Objective: minimize low-priority penalties (preferences, fairness, etc.)
    low_cost = sum(state.low_priority_penalty) if state.low_priority_penalty else 0
    model.Minimize(low_cost)

    # Debug info
    num_constraints, num_bool_vars = get_model_size(model)
    logger.info(f"→ #constraints_p2 = {num_constraints},  #bool_vars = {num_bool_vars}")

    # Re-solve with Phase 1 solution as a hint
    solver = configure_solver()
    for (n, d, s), val in p1.cached_values.items():
        model.AddHint(state.work[n, d, s], val)

    status = solver.Solve(model)
    high_penalty = solver.Value(high_cost) if state.high_priority_penalty else 0
    low_penalty = solver.ObjectiveValue() if state.low_priority_penalty else 0
    fairness_gap = (
        solver.Value(state.gap_pct)
        if getattr(state, "gap_pct", None) is not None
        else None
    )

    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} s")
    logger.info(f"High Priority Penalty Phase 2: {high_penalty}")
    logger.info(f"Low  Priority Penalty Phase 2: {low_penalty}")

    cached = {
        (n, d, s): solver.Value(state.work[n, d, s])
        for n in state.nurse_names
        for d in range(state.num_days)
        for s in range(state.shift_types)
    }

    return SolverResult(solver, status, cached, high_penalty, low_penalty, fairness_gap)


def run_pref_upper_bound(model, state, p1: SolverResult) -> int:
    """Maximize the total number of satisfied preferences under the Phase-1 high-cost optimum."""
    if not getattr(state, "pref_sat_vars", None):
        logger.info("No preference vars found; upper bound is 0.")
        return 0

    # Keep all hard rules ON and pin high-priority cost to phase-1 optimum
    model.Add(sum(r.flag for r in state.hard_rules.values()) == len(state.hard_rules))
    high_cost = sum(state.high_priority_penalty) if state.high_priority_penalty else 0
    model.Add(high_cost == round(p1.high_penalty))

    # Objective: maximize satisfied preferences count (pure count, not weighted)
    sat_count = cp_model.LinearExpr.Sum(state.pref_sat_vars)
    model.Maximize(sat_count)

    solver = configure_solver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.info("Upper-bound run infeasible.")
        return 0

    ub = int(solver.ObjectiveValue())
    logger.info(f"🔎 Preference upper bound (max sat prefs under hard rules): {ub}")
    return ub

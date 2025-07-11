from ortools.sat.python import cp_model
import logging
from typing import Any, Dict, Tuple
from exceptions.custom_errors import NoFeasibleSolutionError

logger = logging.getLogger(__name__)

class SolverResult:
    def __init__(
        self,
        solver: cp_model.CpSolver,
        status: Any,
        cached_values: Dict[Tuple[str,int,int], int],
        high_penalty: float,
        low_penalty: float,
        fairness_gap: Any
    ):
        self.solver = solver
        self.status = status
        self.cached_values = cached_values
        self.high_penalty = high_penalty
        self.low_penalty = low_penalty
        self.fairness_gap = fairness_gap

    
def configure_solver(timeout: float = 180.0, seed: int = 42) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.random_seed = seed
    solver.parameters.relative_gap_limit = 0.01
    solver.parameters.num_search_workers = 8
    solver.parameters.randomize_search = True
    solver.parameters.log_search_progress = False
    return solver


def get_model_size(model: cp_model.CpModel) -> Tuple[int, int]:
    proto = model.Proto()
    num_constraints = len(proto.constraints)
    num_bool_vars = len(proto.variables)
    return num_constraints, num_bool_vars


def run_phase1(model, state) -> SolverResult:
    # Phase 1A: Check feasibility
    logger.info("🚀 Phase 1A: checking feasibility...")
    # 1. Tell the model to minimize penalty sum
    model.Maximize(sum(r.flag for r in state.hard_rules.values()))

    # debug: print model size
    num_constraints, num_bool_vars = get_model_size(model)
    logger.info(f"→ #constraints_p1 = {num_constraints},  #bool_vars = {num_bool_vars}")

    solver = configure_solver()
    status = solver.Solve(model)

    status = solver.Solve(model)
    total_hards = len(state.hard_rules)
    logger.info(total_hards)
    satisfied_hards = int(solver.ObjectiveValue())
    logger.info(satisfied_hards)
    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
    if satisfied_hards != total_hards or status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        dropped = [r.message for r in state.hard_rules.values() if solver.Value(r.flag) == 0]
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

    # # save "best" solution found
    # cached_values = {}
    # for n in nurse_names:
    #     for d in range(num_days):
    #         for s in range(shift_types):
    #             cached_values[(n, d, s)] = solver.Value(work[n, d, s])

    # cached_total_prefs_met = 0
    # for n in nurse_names:
    #     for d in range(num_days):
    #         picked = [s for s in range(shift_types) if cached_values[(n, d, s)]]
    #         pref = prefs_by_nurse[n].get(d)
    #         if pref is not None and len(picked) == 1 and pref in picked:
    #             cached_total_prefs_met += 1

    cached = {
        (n, d, s): solver.Value(state.work[n, d, s])
        for n in state.nurse_names
        for d in range(state.num_days)
        for s in range(state.shift_types)
    }

    fairness_gap = solver.Value(state.gap_pct) if state.gap_pct is not None else None

    # cached_gap = solver.Value(state.gap_pct) if state.gap_pct is not None else "N/A"
    # high1 = solver.ObjectiveValue()
    # best_penalty = solver.ObjectiveValue() + solver.Value(sum(state.low_priority_penalty))
    # logger.info(f"▶️ Phase 1 complete: best total penalty = {best_penalty}; best fairness gap = {cached_gap}")

    return SolverResult(solver, status, cached, high_penalty, low_penalty, fairness_gap)


def run_phase2(model, state, p1: SolverResult) -> SolverResult:
    # Phase 2: Maximize preferences
    logger.info("🚀 Phase 2: maximizing preferences…")

    model.Add(sum(r.flag for r in state.hard_rules.values()) == len(state.hard_rules))
    model.Add(sum(state.high_priority_penalty) <= round(p1.high_penalty))
    if p1.fairness_gap is not None:
        model.Add(state.gap_pct <= round(p1.fairness_gap))
        # model.AddLinearConstraint(gap_pct, 0, T)

    # Switch objective to preferences
    # preference_obj = sum(total_satisfied[n] for n in nurse_names)
    # model.Maximize(preference_obj)
    model.Minimize(sum(state.low_priority_penalty))

    # debug: print model size
    num_constraints, num_bool_vars = get_model_size(model)
    logger.info(f"→ #constraints_p2 = {num_constraints},  #bool_vars = {num_bool_vars}")

    # 4. Re-solve with cached values from Phase 1 as hints
    solver = configure_solver()
    for (n, d, s), val in p1.cached_values.items():
        model.AddHint(state.work[n, d, s], val)

    status = solver.Solve(model)
    high_penalty = solver.Value(sum(state.high_priority_penalty))
    low_penalty = solver.ObjectiveValue()
    fairness_gap = solver.Value(state.gap_pct) if state.gap_pct is not None else None

    logger.info(f"⏱ Solve time: {solver.WallTime():.2f} seconds")
    logger.info(f"High Priority Penalty Phase 2: {high_penalty}")
    logger.info(f"Low Priority Penalty Phase 2: {low_penalty}")

    cached = {
        (n, d, s): solver.Value(state.work[n, d, s])
        for n in state.nurse_names
        for d in range(state.num_days)
        for s in range(state.shift_types)
    }

    return SolverResult(solver, status, cached, high_penalty, low_penalty, fairness_gap)

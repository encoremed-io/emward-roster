import pandas as pd
from typing import Tuple
from ortools.sat.python import cp_model
from core.state import ScheduleState
from .solver import SolverResult, run_phase1, run_phase2
from .extractor import extract_schedule_and_summary, get_total_prefs_met
import logging

logger = logging.getLogger(__name__)

def solve_schedule(model: cp_model.CpModel, state: ScheduleState, og_nurse_order: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    # Run Phase 1
    p1 = run_phase1(model, state)
    best_result = p1
    
    # Only run phase 2 if low priority penalty exists, which means shifts preferences exist
    if state.low_priority_penalty:
        p2: SolverResult = run_phase2(model, state, p1)
    
        if p2.status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            logger.info("⚠️ Phase 2 failed: using fallback Phase 1 solution.")
            logger.info(f"Solver Phase 2 status: {p2.solver.StatusName(p2.status)}")
        else:
            logger.info(f"▶️ Phase 2 complete")
            best_result = p2
    
    else:
        logger.info("⏭️ Skipping Phase 2: No shift preferences provided.")

    logger.info("✅ Done!")
    logger.info(f"📊 Total penalties = {best_result.high_penalty + (best_result.low_penalty or 0)}")
    logger.info(f"🔍 Total preferences met = {get_total_prefs_met(state, best_result)}")
    if best_result.fairness_gap is not None:
        logger.info(f"📈 Fairness gap (max % - min %) = {best_result.fairness_gap}")

    schedule_df, summary_df, violations, metrics = extract_schedule_and_summary(state, best_result, og_nurse_order)
    logger.info("📁 Schedule and summary generated.")
    return schedule_df, summary_df, violations, metrics

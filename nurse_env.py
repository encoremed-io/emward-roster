# nurse_env.py

import gymnasium as gym
import numpy as np
import pandas as pd
import json
from gymnasium import spaces, Env
from gymnasium.utils import seeding
from datetime import date as dt_date
from collections import defaultdict

with open('config/constants.json', 'r') as f:
    constants = json.load(f)

SHIFT_LABELS = constants["SHIFT_LABELS"]
SHIFT_HOURS = constants["SHIFT_HOURS"]
AVG_HOURS = constants["AVG_HOURS"]
DAYS_PER_WEEK = constants["DAYS_PER_WEEK"]
MIN_ACCEPTABLE_WEEKLY_HOURS = constants["MIN_ACCEPTABLE_WEEKLY_HOURS"]
PREFERRED_WEEKLY_HOURS = constants["PREFERRED_WEEKLY_HOURS"]
PREF_HOURS_PENALTY = constants["PREF_HOURS_PENALTY"]
AM_COVERAGE_MIN_PERCENT = constants["AM_COVERAGE_MIN_PERCENT"]
AM_COVERAGE_PENALTIES = constants["AM_COVERAGE_PENALTIES"]
PREF_MISS_PENALTY = constants["PREF_MISS_PENALTY"]
FAIRNESS_GAP_PENALTY = constants["FAIRNESS_GAP_PENALTY"]
FAIRNESS_GAP_THRESHOLD = constants["FAIRNESS_GAP_THRESHOLD"]

def compute_total_penalty(assignment: np.ndarray,
                          profiles_df: pd.DataFrame,
                          preferences_df: pd.DataFrame,
                          start_date: pd.Timestamp | dt_date,
                          fixed_assignments,
                          active_days: int) -> int:
    """
    Compute the total penalty for a given assignment.
    (Same logic as in your standalone function.)
    """

    if isinstance(start_date, pd.Timestamp):
        date_start : dt_date = start_date.date()
    else:
        date_start = start_date

    # Prepare name/index mappings
    nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
    name_to_idx = {n: i for i, n in enumerate(nurse_names)}
    shift_str_to_idx = {'AM': 0, 'PM': 1, 'NIGHT': 2}

    el_days = defaultdict(set)
    for (nurse, d), label in fixed_assignments.items():
        if label.upper() == "EL":
            el_days[nurse].add(d)

    num_days = active_days

    # Build preferences and MC-day sets
    shift_prefs = {n: {} for n in nurse_names}
    mc_days = {n: set() for n in nurse_names}
    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        for label, val in row.items():
            if isinstance(label, pd.Timestamp):
                d = label.date()
            elif isinstance(label, dt_date):
                d = label
            else:
                d = pd.to_datetime(str(label)).date()

            d = (d - date_start).days
            if pd.notna(val) and 0 <= d < num_days:
                v = str(val).strip().upper()
                if v == 'MC':
                    mc_days[nm].add(d)
                elif v in shift_str_to_idx:
                    shift_prefs[nm][d] = shift_str_to_idx[v]

    total_penalty = 0
    N, D, S = assignment.shape

    # 1) Weekly hours soft penalty
    for i, nurse in enumerate(nurse_names):
        for w in range((num_days + DAYS_PER_WEEK - 1) // DAYS_PER_WEEK):
            days = list(range(w*DAYS_PER_WEEK, min((w+1)*DAYS_PER_WEEK, num_days)))
            hours = sum(assignment[i, d, s] * SHIFT_HOURS[s]
                        for d in days for s in range(S))
            mc_cnt = sum(1 for d in days if d in mc_days[nurse])
            el_cnt = sum(1 for d in days if d in el_days[nurse])
            adj = (mc_cnt + el_cnt) * int(AVG_HOURS)
            eff_pref = max(0, PREFERRED_WEEKLY_HOURS - adj)
            if hours < eff_pref:
                total_penalty += PREF_HOURS_PENALTY

    # 2) AM coverage penalty
    for d in range(num_days):
        total_shifts = assignment[:, d, :].sum()
        if total_shifts == 0:
            continue
        am_count = assignment[:, d, 0].sum()
        pct = 100 * am_count / total_shifts
        if pct < AM_COVERAGE_MIN_PERCENT - 20:
            total_penalty += AM_COVERAGE_PENALTIES[2]
        elif pct < AM_COVERAGE_MIN_PERCENT - 10:
            total_penalty += AM_COVERAGE_PENALTIES[1]
        elif pct < AM_COVERAGE_MIN_PERCENT:
            total_penalty += AM_COVERAGE_PENALTIES[0]

    # 3) Preference-miss and fairness gap penalties
    pct_sat = []
    for i, nurse in enumerate(nurse_names):
        prefs = shift_prefs[nurse]
        met = 0
        for d, s in prefs.items():
            if assignment[i, d, s] == 1:
                met += 1
            else:
                total_penalty += PREF_MISS_PENALTY
        if prefs:
            pct_sat.append(100 * met / len(prefs))
    if pct_sat:
        gap = max(pct_sat) - min(pct_sat)
        if gap >= FAIRNESS_GAP_THRESHOLD:
            over_gap = gap - FAIRNESS_GAP_THRESHOLD
            total_penalty += over_gap * FAIRNESS_GAP_PENALTY

    return int(total_penalty)


def compute_penalty_per_day(assignment: np.ndarray,
                            profiles_df: pd.DataFrame,
                            preferences_df: pd.DataFrame,
                            start_date: pd.Timestamp | dt_date,
                            day_idx: int) -> int:
    """
    Compute penalty only for the given day_idx.
    Includes:
    - AM coverage penalty for that day
    - Preference miss penalty for that day
    Note: Weekly hours and fairness gap penalties require weekly or full schedule info,
    so they are omitted here (or can be added with some approximation).
    """

    if isinstance(start_date, pd.Timestamp):
        date_start : dt_date = start_date.date()
    else:
        date_start = start_date

    nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
    shift_str_to_idx = {'AM': 0, 'PM': 1, 'NIGHT': 2}

    N, D, S = assignment.shape
    if day_idx < 0 or day_idx >= D:
        raise ValueError("day_idx out of range")

    # Build preferences for nurses for this day only
    shift_prefs = {n: {} for n in nurse_names}
    for nurse, row in preferences_df.iterrows():
        nm = str(nurse).strip().upper()
        for label, val in row.items():
            if isinstance(label, pd.Timestamp):
                d = label.date()
            elif isinstance(label, dt_date):
                d = label
            else:
                d = pd.to_datetime(str(label)).date()
            d = (d - date_start).days
            if d == day_idx and pd.notna(val):
                v = str(val).strip().upper()
                if v in shift_str_to_idx:
                    shift_prefs[nm][d] = shift_str_to_idx[v]

    total_penalty = 0

    # 1) AM coverage penalty for this day only
    total_shifts = assignment[:, day_idx, :].sum()
    if total_shifts > 0:
        am_count = assignment[:, day_idx, 0].sum()
        pct = 100 * am_count / total_shifts
        if pct < AM_COVERAGE_MIN_PERCENT - 20:
            total_penalty += AM_COVERAGE_PENALTIES[2]
        elif pct < AM_COVERAGE_MIN_PERCENT - 10:
            total_penalty += AM_COVERAGE_PENALTIES[1]
        elif pct < AM_COVERAGE_MIN_PERCENT:
            total_penalty += AM_COVERAGE_PENALTIES[0]

    # 2) Preference-miss penalty for this day only
    for i, nurse in enumerate(nurse_names):
        prefs = shift_prefs[nurse]
        # prefs only for this day (day_idx), so check if nurse has preference this day
        if day_idx in prefs:
            preferred_shift = prefs[day_idx]
            if assignment[i, day_idx, preferred_shift] != 1:
                total_penalty += PREF_MISS_PENALTY

    return int(total_penalty)


class NurseRosteringEnv(Env):
    """Gym environment for nurse rostering."""
    metadata = {'render.modes': []}

    def __init__(self,
                 profiles_df: pd.DataFrame,
                 preferences_df: pd.DataFrame,
                 start_date: pd.Timestamp,
                 active_days: int = 0):
        
        super().__init__()

        if len(profiles_df) == 0:
            raise ValueError("profiles_df cannot be empty")
        if len(preferences_df.columns) == 0:
            raise ValueError("preferences_df must contain at least one day")
        
        # Set days parameters
        # If active_days not provided, use all columns in prefs
        self.active_days = active_days
        self.fixed_assignments = {}

        # # Pad preferences_df up to max_days columns with zeros
        # prefs = preferences_df.copy()
        # if prefs.shape[1] < max_days:
        #     last_date = pd.to_datetime(prefs.columns[-1])
        #     for i in range(prefs.shape[1], max_days):
        #         new_date = (last_date + pd.Timedelta(days=(i - prefs.shape[1] + 1))).date()
        #         prefs[new_date] = 0  # neutral preference value
        # # If more cols than max, slice
        # prefs = prefs.iloc[:, :max_days]
        
        self.profiles_df = profiles_df
        self.preferences_df = preferences_df
        self.start_date = start_date
        self.np_random, _ = seeding.np_random(None)

        self.nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
        self.N = len(self.nurse_names)
        self.D = active_days
        self.S = 3  # AM, PM, Night

        # One-hot assignment tensor: shape (N, D, S)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.N * self.D * self.S,),
            dtype=np.int8
        )
        self.action_space = spaces.Discrete(self.N * self.D * self.S)

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        super().reset(seed=seed)
        
        # Re-initialize environment state
        self.assignment = np.zeros((self.N, self.D, self.S), dtype=np.int8)
        self.step_count = 0
        self.valid_count = 0
        self.total_reward = 0
        self.done = False
    
        return self._get_obs(), {}  # observation and empty info dict
    

    def _get_obs(self):
        # full = self.assignment.flatten()
        # # Mask out positions beyond active_days
        # feat_per_day = (self.N * self.S)
        # cutoff = feat_per_day * self.active_days
        # masked = np.zeros_like(full)
        # masked[:cutoff] = full[:cutoff]
        # return masked
        return self.assignment.flatten()


    def step(self, action: int):
        n = action // (self.D * self.S)
        rem = action % (self.D * self.S)
        d = rem // self.S
        s = rem % self.S

        penalty_reduction = 0.0
        valid_assign_bonus = 0.0
        pref_match_bonus = 0.0
        am_coverage_bonus = 0.0
        weekly_hours_bonus = 0.0
        final_penalty_term = 0.0
        reward = 0.0

        self.step_count += 1

        # # skip masked days
        # if d >= self.active_days:
        #     obs = self._get_obs()
        #     info = {
        #         "penalty_reduction": penalty_reduction,
        #         "valid_assign":      valid_assign_bonus,
        #         "preference":        pref_match_bonus,
        #         "am_coverage":       am_coverage_bonus,
        #         "weekly_hours":      weekly_hours_bonus,
        #         "final_penalty":     final_penalty_term
        #     }
        #     return obs, reward, done, False, info

        # 1) Illegal move mask: if nurse already has a shift that day, zero reward and skip
        if self.assignment[n, d].any():
            # no change to assignment, small -1 reward instead of huge negative
            reward = -1.0
            valid_assign_bonus -= 1.0
            print(f"[ILLEGAL MOVE] nurse={n}, day={d}")  # for debugging
        else:
            self.valid_count += 1

            # 2) Compute old penalty for this day
            old_penalty = compute_penalty_per_day(
                self.assignment,
                self.profiles_df,
                self.preferences_df,
                self.start_date,
                d
            )

            # 3) Apply the action
            self.assignment[n, d, s] = 1

            # 4) Compute new penalty for this day
            new_penalty = compute_penalty_per_day(
                self.assignment,
                self.profiles_df,
                self.preferences_df,
                self.start_date,
                d
            )

            # 5) Base reward = reduction in per-day penalty
            penalty_reduction = float(old_penalty - new_penalty)

            # ── REWARD SHAPING ──
            # +1 for any valid assignment
            valid_assign_bonus += 1.0

            # +5 if this assignment matches the nurse’s preference for that day
            # (assumes SHIFT_LABELS = ['AM','PM','Night'] is defined at class scope)
            nurse_name = self.nurse_names[n]
            day_date = (self.start_date + pd.to_timedelta(d, unit="D")).date()
            pref_val = self.preferences_df.at[nurse_name, day_date]
            if isinstance(pref_val, str) and pref_val.strip().upper() == SHIFT_LABELS[s].upper():
                pref_match_bonus += 5

            # ── DAILY AM‐COVERAGE BONUS ──
            # whenever you’ve taken N actions, you’ve just finished assigning everyone
            # for one day, so step_count % N == 0
            if self.valid_count % self.N == 0:
                # which day did we just finish?
                day_finished = (self.valid_count // self.N) - 1

                # compute AM coverage for that day
                am_count   = int(self.assignment[:, day_finished, 0].sum())
                total_any  = int(self.assignment[:, day_finished, :].sum())
                pct_am     = 100 * am_count / total_any if total_any > 0 else 0

                am_coverage_bonus += am_count

                # if you hit your AM‐coverage target, give +5
                if pct_am >= AM_COVERAGE_MIN_PERCENT:
                    am_coverage_bonus += 5.0 

                # ── DAILY PARTIAL WEEKLY‐HOURS BONUS ──
                # Compute total hours for days [0..day_finished]
                week_idx = day_finished // DAYS_PER_WEEK
                if week_idx == 0:  # only first week
                    week_hours_so_far = 0
                    for dd in range(0, day_finished + 1):
                        for shift_idx, hrs in enumerate(SHIFT_HOURS):
                            week_hours_so_far += int(
                                self.assignment[:, dd, shift_idx].sum()
                            ) * hrs

                    avg_hours_so_far = week_hours_so_far / self.N
                    # Give up to +2 points per day if avg_hours_so_far → 40
                    partial_bonus = (avg_hours_so_far / PREFERRED_WEEKLY_HOURS) * 2.0
                    weekly_hours_bonus += partial_bonus

                # ── FULL WEEKLY‐HOURS TARGET ON DAY 6 ──
                if (day_finished + 1) % DAYS_PER_WEEK == 0:
                    # Sum 7 days exactly
                    start_d = week_idx * DAYS_PER_WEEK
                    end_d   = start_d + DAYS_PER_WEEK
                    full_week_hours = 0
                    for dd in range(start_d, end_d):
                        for shift_idx, hrs in enumerate(SHIFT_HOURS):
                            full_week_hours += int(
                                self.assignment[:, dd, shift_idx].sum()
                            ) * hrs

                    avg_nurse_hours = full_week_hours / self.N
                    # If we hit 40 (PREFERRED_WEEKLY_HOURS), +10 bonus
                    if avg_nurse_hours >= PREFERRED_WEEKLY_HOURS:
                        weekly_hours_bonus += 10.0

            raw_reward = (
                penalty_reduction
                + valid_assign_bonus
                + pref_match_bonus
                + am_coverage_bonus
                + weekly_hours_bonus
            )
            reward = raw_reward

        # 6) End-of-episode final penalty
        done = (self.step_count >= self.N * self.D)
        if done:
            final_penalty = compute_total_penalty(
                self.assignment,
                self.profiles_df,
                self.preferences_df,
                self.start_date,
                self.fixed_assignments,
                self.active_days
            )
            final_penalty_term = float(final_penalty / self.active_days)
            survival_bonus = 5.0
            # subtract the long-term penalty once at the end
            reward = reward - (final_penalty_term / 100) + survival_bonus
        
        # 7) Clip per‐step reward to [-10, +10]
        reward = max(-10, min(10, reward))

        # 8) Build the info dict so you can log each component later
        info = {
            "penalty_reduction": penalty_reduction,
            "valid_assign":      valid_assign_bonus,
            "preference":        pref_match_bonus,
            "am_coverage":       am_coverage_bonus,
            "weekly_hours":      weekly_hours_bonus,
            "final_penalty":     final_penalty_term
        }

        # 8) Return observation, shaped reward, done flag, and info
        return self._get_obs(), reward, done, False, info


    def render(self, mode='human'):
        # Print a simple readable schedule: nurses x days, with shift labels or '-'
        for n, nurse in enumerate(self.nurse_names):
            line = f"{nurse:10s}: "
            for d in range(self.D):
                assigned_shift = np.where(self.assignment[n, d, :] == 1)[0]
                if len(assigned_shift) == 1:
                    line += SHIFT_LABELS[assigned_shift[0]][0] + " "
                else:
                    line += "- "
            print(line)

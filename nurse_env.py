# nurse_env.py

import gymnasium as gym
import numpy as np
import pandas as pd
import json
from gymnasium import spaces, Env
from gymnasium.utils import seeding
from datetime import date as dt_date

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

def compute_total_penalty(assignment: np.ndarray,
                          profiles_df: pd.DataFrame,
                          preferences_df: pd.DataFrame,
                          start_date: pd.Timestamp | dt_date) -> int:
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

    num_days = assignment.shape[1]

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
            adj = mc_cnt * AVG_HOURS
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
        total_penalty += gap * FAIRNESS_GAP_PENALTY

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
                 num_days: int):
        
        super().__init__()

        if len(profiles_df) == 0:
            raise ValueError("profiles_df cannot be empty")
        if len(preferences_df.columns) == 0:
            raise ValueError("preferences_df must contain at least one day")
        
        self.profiles_df = profiles_df
        self.preferences_df = preferences_df
        self.start_date = start_date
        self.num_days = num_days
        self.np_random, _ = seeding.np_random(None)

        self.nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
        self.N = len(self.nurse_names)
        self.D = num_days
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
        self.total_reward = 0
        self.done = False

        obs = self.assignment.flatten()

        if obs.ndim == 0:
            obs = np.array([], dtype=np.int8)
    
        return obs, {}  # observation and empty info dict


    def step(self, action: int):
        n = action // (self.D * self.S)
        rem = action % (self.D * self.S)
        d = rem // self.S
        s = rem % self.S

        # Default info and done flag
        info = {}
        self.step_count += 1
        done = (self.step_count >= self.N * self.D)

        # 1) Illegal move mask: if nurse already has a shift that day, zero reward and skip
        if self.assignment[n, d].any():
            # no change to assignment, small zero reward instead of huge negative
            reward = 0
        else:
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
            reward = old_penalty - new_penalty

            # ── REWARD SHAPING ──
            # +1 for any valid assignment
            reward += 1

            # +5 if this assignment matches the nurse’s preference for that day
            # (assumes SHIFT_LABELS = ['AM','PM','Night'] is defined at class scope)
            nurse_name = self.nurse_names[n]
            day_date = (self.start_date + pd.to_timedelta(d, unit="D")).date()
            pref_val = self.preferences_df.at[nurse_name, day_date]
            if isinstance(pref_val, str) and pref_val.strip().upper() == SHIFT_LABELS[s].upper():
                reward += 5

        # 6) End-of-episode final penalty
        if done:
            final_penalty = compute_total_penalty(
                self.assignment,
                self.profiles_df,
                self.preferences_df,
                self.start_date
            )
            info['final_penalty'] = final_penalty
            # subtract the long-term penalty once at the end
            reward -= final_penalty
        
        # 7) Clip per‐step reward to [-10, +10]
        reward = max(-10, min(10, reward))

        # 8) Return flattened observation, shaped reward, done flag, and info
        return self.assignment.flatten(), reward, done, False, info


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

# nurse_env.py

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces, Env
from datetime import date as dt_date

def compute_total_penalty(assignment: np.ndarray,
                          profiles_df: pd.DataFrame,
                          preferences_df: pd.DataFrame,
                          start_date: pd.Timestamp | dt_date) -> int:
    """
    Compute the total penalty for a given assignment.
    (Same logic as in your standalone function.)
    """
    # === Constants ===
    SHIFT_HOURS = [7, 7, 10]
    AVG_HOURS = 7
    DAYS_PER_WEEK = 7

    MIN_ACCEPTABLE_WEEKLY_HOURS = 30
    PREFERRED_WEEKLY_HOURS = 40
    PREF_HOURS_PENALTY = 1000

    AM_COVERAGE_MIN_PERCENT = 60
    AM_COVERAGE_PENALTIES = [1000, 5000, 10000]

    PREF_MISS_PENALTY = 10
    FAIRNESS_GAP_PENALTY = 5

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


class NurseRosteringEnv(Env):
    """Gym environment for nurse rostering."""
    metadata = {'render.modes': []}

    def __init__(self,
                 profiles_df: pd.DataFrame,
                 preferences_df: pd.DataFrame,
                 start_date: pd.Timestamp,
                 num_days: int):
        super().__init__()
        self.profiles_df = profiles_df
        self.preferences_df = preferences_df
        self.start_date = start_date
        self.num_days = num_days

        self.nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
        self.N = len(self.nurse_names)
        self.D = num_days
        self.S = 3  # AM, PM, Night

        # One-hot assignment tensor: shape (N, D, S)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.N, self.D, self.S),
            dtype=np.int8
        )
        self.action_space = spaces.Discrete(self.N * self.D * self.S)

        self.reset()

    def reset(self):
        self.assignment = np.zeros((self.N, self.D, self.S), dtype=np.int8)
        self.step_count = 0
        return self.assignment.copy()

    def step(self, action: int):
        n = action // (self.D * self.S)
        rem = action % (self.D * self.S)
        d = rem // self.S
        s = rem % self.S

        # If nurse already has a shift that day, big penalty
        if self.assignment[n, d].any():
            reward = -10_000
        else:
            self.assignment[n, d, s] = 1
            reward = 0

        self.step_count += 1
        done = (self.step_count >= self.N * self.D)

        if done:
            # subtract the total penalty as our final reward
            penalty = compute_total_penalty(
                self.assignment,
                self.profiles_df,
                self.preferences_df,
                self.start_date
            )
            reward -= penalty

        return self.assignment.copy(), reward, done, {}

    def render(self, mode='human'):
        # you can implement a simple print-out here if you like
        pass

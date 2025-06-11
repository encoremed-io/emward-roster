import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces, Env
from gymnasium.utils import seeding

from utils.pen_calc import *

HUGE_VIOLATION_PENALTY = 2000

class NurseRosteringEnv(Env):
    """Gym environment for nurse rostering."""
    metadata = {'render.modes': []}

    def __init__(
        self,
        profiles_df: pd.DataFrame,
        preferences_df: pd.DataFrame,
        start_date: pd.Timestamp,
        active_days: int = 0,
        phase: int = 2,
        hp_baseline=None
    ):
        super().__init__()

        assert phase in (1, 2), "phase must be 1 or 2"
        self.phase = phase
        self.hp_baseline = hp_baseline

        if profiles_df.empty:
            raise ValueError("profiles_df cannot be empty")
        if preferences_df.columns.empty:
            raise ValueError("preferences_df must contain at least one day")

        self.profiles_df = profiles_df
        self.preferences_df = preferences_df
        self.start_date = start_date
        self.active_days = active_days
        self.fixed_assignments = {}

        self.np_random, _ = seeding.np_random(None)
        self.nurse_names = [str(n).strip().upper() for n in profiles_df['Name']]
        self.N = len(self.nurse_names)
        self.D = active_days
        self.S = len(SHIFT_LABELS)

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

        self.assignment = np.zeros((self.N, self.D, self.S), dtype=np.int8)
        self.step_count = 0
        self.total_reward = 0
        self.done = False

        # compute initial HP penalty
        self.cum_hp = compute_high_priority_penalty(
            self.assignment,
            self.profiles_df,
            self.preferences_df,
            self.start_date,
            self.fixed_assignments,
            self.active_days
        )
        return self._get_obs(), {}

    def _get_obs(self):
        return self.assignment.flatten()

    def step(self, action: int):
        # decode action index
        n = action // (self.D * self.S)
        rem = action % (self.D * self.S)
        d = rem // self.S
        s = rem % self.S

        if self.done:
            raise RuntimeError("Step called after environment is done")

        self.step_count += 1

        # illegal move?
        if self.assignment[n, d].any():
            return self._get_obs(), -1.0, False, False, {"illegal": 1}

        # pre‐action HP penalty
        hp_before = compute_high_priority_penalty(
            self.assignment, self.profiles_df, self.preferences_df,
            self.start_date, self.fixed_assignments, self.active_days
        )

        # apply assignment
        self.assignment[n, d, s] = 1

        # post‐action HP penalty
        hp_after = compute_high_priority_penalty(
            self.assignment, self.profiles_df, self.preferences_df,
            self.start_date, self.fixed_assignments, self.active_days
        )
        self.cum_hp = hp_after
        hp_delta = hp_before - hp_after

        # low‐priority penalty (aggregate)
        lp_pen = compute_low_priority_penalty(
            self.assignment, self.profiles_df, self.preferences_df,
            self.start_date, self.active_days
        )
        lp_estimate = -lp_pen

        # prepare breakdown inputs
        shift_prefs, mc_days = get_shift_prefs_and_mc_days(
            self.preferences_df, self.profiles_df, self.start_date, self.active_days
        )
        el_days = get_el_days(self.fixed_assignments)
        senior_set = get_senior_set(self.profiles_df)

        # ── High-priority components ───────────────────────────────
        hp_am_cov        = hp_am_coverage(self.assignment, self.active_days)
        hp_weekly        = hp_weekly_hours(
            self.assignment, self.nurse_names, mc_days, el_days, self.active_days
        )

        # staffing shortages
        nurse_shortage = 0
        senior_shortage = 0
        for day in range(self.D):
            for shift in range(self.S):
                cnt = int(self.assignment[:, day, shift].sum())
                if cnt < MIN_NURSES_PER_SHIFT:
                    nurse_shortage += (MIN_NURSES_PER_SHIFT - cnt)
                if cnt > 0:
                    sc = sum(
                        int(self.assignment[i, day, shift] == 1 and self.nurse_names[i] in senior_set)
                        for i in range(self.N)
                    )
                    if sc < MIN_SENIORS_PER_SHIFT:
                        senior_shortage += 1

        hp_nurses_per_shift  = nurse_shortage * HARD_CONSTRAINT_PENALTY
        hp_seniors_per_shift = senior_shortage * HARD_CONSTRAINT_PENALTY

        # ── Low-priority components ────────────────────────────────
        pref_misses = 0
        sats = []
        for i, nm in enumerate(self.nurse_names):
            prefs = shift_prefs.get(nm, {})
            if not prefs:
                continue
            met = 0
            for day, ss in prefs.items():
                if self.assignment[i, day, ss] == 1:
                    met += 1
                else:
                    pref_misses += 1
            sats.append(100 * met / len(prefs))

        lp_preference = pref_misses * PREF_MISS_PENALTY

        gap = max(sats) - min(sats) if sats else 0
        lp_fairness = ((gap - FAIRNESS_GAP_THRESHOLD) * FAIRNESS_GAP_PENALTY
                       if sats and gap >= FAIRNESS_GAP_THRESHOLD else 0)

        # ── Reward & termination ───────────────────────────────────
        if self.phase == 1:
            reward = hp_delta
            violation = 0
        else:
            BIG = 1000.0
            reward = hp_delta * BIG + lp_estimate
            violation = 0
            if self.hp_baseline is not None and self.cum_hp > self.hp_baseline:
                violation = self.cum_hp - self.hp_baseline
                reward -= violation * HUGE_VIOLATION_PENALTY

        self.done = (self.step_count >= self.N * self.D)
        truncated = False
        reward = float(np.clip(reward, -10, 10))

        info = {
            # high priority
            "hp_am_coverage":       hp_am_cov,
            "hp_weekly_hours":      hp_weekly,
            "hp_nurses_per_shift":  hp_nurses_per_shift,
            "hp_seniors_per_shift": hp_seniors_per_shift,
            # low priority
            "lp_preference":        -lp_preference,
            "lp_fairness":          -lp_fairness,
            # aggregates
            "hp_delta":             hp_delta,
            "lp_estimate":          lp_estimate,
            "illegal":              0,
        }
        if self.phase == 2 and violation > 0:
            info["hp_violation"] = violation

        return self._get_obs(), reward, self.done, truncated, info


    def render(self, mode='human'):
        for i, nurse in enumerate(self.nurse_names):
            line = f"{nurse:10s}: "
            for d in range(self.D):
                shifts = np.where(self.assignment[i, d] == 1)[0]
                line += (SHIFT_LABELS[shifts[0]][0] if len(shifts) == 1 else "-") + " "
            print(line)
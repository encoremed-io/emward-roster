import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces, Env
from gymnasium.utils import seeding

from utils.pen_calc import *
from utils.shift_utils import *

HUGE_VIOLATION_PENALTY = 10000

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
        self.best_phase1_result = None
        self.best_phase2_result = None

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
        # only AM, PM, Night
        self.S = len(SHIFT_LABELS)
        # add one extra action representing an implicit Rest (no assignment)
        self.A = self.S + 1

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.N * self.D * self.S,),
            dtype=np.int8
        )
        # actions 0..S-1 = shifts, S = rest (no shift)
        self.action_space = spaces.Discrete(self.A)

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        super().reset(seed=seed)

        # assignment[n,d,s] flags only for real shifts
        self.assignment = np.zeros((self.N, self.D, self.S), dtype=np.int8)
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        self.assigned_flag = np.zeros((self.N, self.D), dtype=bool)

        # parse all prefs & MC days exactly once per episode
        self.shift_prefs, self.mc_days = (
            get_shift_preferences(self.preferences_df, self.profiles_df, self.start_date, self.active_days, SHIFT_LABELS),
            get_mc_days(self.preferences_df, self.profiles_df, self.start_date, self.active_days)
        )

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

    def _current_nd(self):
        idx = self.step_count
        # # assign shifts for 1 nurse for all days first, before moving on to next nurse
        # n = idx // self.D
        # d = idx % self.D
        # assign shifts for all nurses for 1 day first, before moving on to next day
        d = idx // self.N
        n = idx % self.N
        return n, d

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Step called after environment is done")

        n, d = self._current_nd()
        s = action

        shift_prefs, mc_days = self.shift_prefs, self.mc_days
        nm = self.nurse_names[n]

        # ——— ILLEGAL: already assigned ———
        if self.assigned_flag[n, d]:
            self.assigned_flag[n, d] = True
            self.step_count += 1
            self.done = (self.step_count >= self.N * self.D)
            # print(f"[ILLEGAL] nurse={n}, day={d}, Shift already exists")
            return self._get_obs(), -HUGE_VIOLATION_PENALTY, self.done, False, {
                "illegal": 1,
                "reason": "already assigned",
            }

        # ——— ILLEGAL: MC violation ———
        if d in mc_days.get(nm, set()):
            self.assigned_flag[n, d] = True
            self.step_count += 1
            self.done = (self.step_count >= self.N * self.D)
            # print(f"[MC] nurse={n}, day={d}, MC violation")
            return self._get_obs(), 0, self.done, False, {
                "illegal": 1,
                "reason": "MC violation",
            }

        # ——— LEGAL assignment ———
        hp_before = self.cum_hp

        if s < self.S:
            self.assignment[n, d, s] = 1
        #     print(f"[ASSIGNED] nurse={n}, day={d}, shift={SHIFT_LABELS[s]}")
        # else:
        #     print(f"[ASSIGNED] nurse={n}, day={d}, REST (implicit)")

        self.assigned_flag[n, d] = True

        hp_after = compute_high_priority_penalty(
            self.assignment, self.profiles_df, self.preferences_df,
            self.start_date, self.fixed_assignments, self.active_days
        )
        self.cum_hp = hp_after
        # hp_delta = hp_before - hp_after
        hp_delta = (hp_before - hp_after) / (hp_before + 1e-8)

        lp_pen      = compute_low_priority_penalty(
            self.assignment, self.profiles_df, self.preferences_df,
            self.start_date, self.active_days
        )
        lp_estimate = -lp_pen

        el_days    = get_el_days(self.fixed_assignments, self.nurse_names)
        senior_set = get_senior_set(self.profiles_df)

        # HP components
        hp_am_cov         = hp_am_coverage(self.assignment, self.active_days)
        hp_am_senior_cov = hp_am_senior_coverage(
            self.assignment, self.nurse_names, senior_set, self.active_days
        )
        hp_weekly = hp_weekly_hours(
            self.assignment, self.nurse_names, mc_days, el_days, self.active_days
        )
        hp_nurses_per_shift = hp_nurses_staffing_level(self.assignment, self.active_days)
        hp_seniors_per_shift = hp_senior_staffing_level(
            self.assignment, self.nurse_names, senior_set, self.active_days
        )

        # LP: preference & fairness
        pref_misses = 0
        sats        = []
        for i, nm in enumerate(self.nurse_names):
            prefs = shift_prefs.get(nm, {})
            met   = sum(self.assignment[i, day, ss]
                        for day, ss in prefs.items())
            pref_misses += int(len(prefs) - met)
            if prefs:
                sats.append(100.0 * met / len(prefs))

        lp_preference = int(pref_misses) * PREF_MISS_PENALTY
        gap           = max(sats) - min(sats) if sats else 0
        lp_fairness   = ((gap - FAIRNESS_GAP_THRESHOLD) *
                         FAIRNESS_GAP_PENALTY
                         if gap >= FAIRNESS_GAP_THRESHOLD else 0)
        
        # ——— Advance and return for the legal case ———
        self.step_count += 1
        self.done = (self.step_count >= self.N * self.D)

        # Compute reward
        violation = 0
        if self.phase == 1:
            reward = hp_delta
            if self.done:
                if self.best_phase1_result is None or self.cum_hp < self.best_phase1_result:
                    reward += 1000
                    self.best_phase1_result = self.cum_hp
                else:
                    reward -= np.clip(self.cum_hp - self.best_phase1_result, 0, 5000)

        else:
            reward = hp_delta * 10.0 + lp_estimate
            if self.done:
                if self.hp_baseline is not None and self.cum_hp > self.hp_baseline:
                    violation = self.cum_hp - self.hp_baseline
                    reward -= violation * HUGE_VIOLATION_PENALTY

                if self.best_phase2_result is None or self.cum_hp < self.best_phase2_result:
                    reward += 1000  # Bonus for better schedule
                    self.best_phase2_result = self.cum_hp
                else:
                    reward -= np.clip(self.cum_hp - self.best_phase2_result, 0, 5000) 

        info = {
            "illegal":               0,
            "hp_am_coverage":        -hp_am_cov,
            "hp_am_senior_coverage": -hp_am_senior_cov,
            "hp_weekly_hours":       -hp_weekly,
            "hp_nurses_per_shift":   -hp_nurses_per_shift,
            "hp_seniors_per_shift":  -hp_seniors_per_shift,
            "lp_preference":         -lp_preference,
            "lp_fairness":           -lp_fairness,
            "hp_delta":              hp_delta,
            "lp_estimate":           lp_estimate,
        }
        if self.done:
            info["cum_hp"] = self.cum_hp
            if self.phase == 2 and violation > 0:
                info["hp_violation"] = violation

        return self._get_obs(), reward, self.done, False, info


    def render(self, mode='human'):
        for i, nurse in enumerate(self.nurse_names):
            line = f"{nurse:10s}: "
            for d in range(self.D):
                shifts = np.where(self.assignment[i, d] == 1)[0]
                if shifts.size == 1:
                    token = SHIFT_LABELS[shifts[0]]
                else:
                    token = "-"  # either rest or unassigned
                line += f"{token:5s} "
            print(line)
        print("\nLegend: AM, PM, Night; blank = Rest/MC/unassigned")

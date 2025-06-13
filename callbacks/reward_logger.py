from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardComponentLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._keys = [
            "illegal",
            "hp_am_coverage",
            "hp_am_senior_coverage",
            "hp_weekly_hours",
            "hp_nurses_per_shift",
            "hp_seniors_per_shift",
            "lp_preference",
            "lp_fairness",
            "hp_delta",
            "lp_estimate",
        ]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if not infos:
            return True

        # Step-level logging
        for key in self._keys:
            vals = [info.get(key, 0.0) for info in infos]
            mean_val = float(np.mean(vals))
            self.logger.record(f"reward/{key}", mean_val)

        # Episode-end logging
        for info, done in zip(infos, dones):
            if done:
                if "cum_hp" in info:
                    self.logger.record("episode/cum_hp", info["cum_hp"])
                if "hp_violation" in info:
                    self.logger.record("episode/hp_violation", info["hp_violation"])

        return True

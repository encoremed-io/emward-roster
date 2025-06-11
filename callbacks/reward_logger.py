from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardComponentLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._keys = [
            # high priority
            "hp_am_coverage",
            "hp_weekly_hours",
            "hp_nurses_per_shift",
            "hp_seniors_per_shift",
            # low priority
            "lp_preference",
            "lp_fairness",
            # aggregates (optional)
            "hp_delta",
            "lp_estimate",
        ]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for key in self._keys:
            vals = [info.get(key, 0.0) for info in infos]
            mean_val = float(np.mean(vals))
            self.logger.record(f"reward/{key}", mean_val)
        return True


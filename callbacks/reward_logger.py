from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardComponentLogger(BaseCallback):
    """
    Custom callback for SB3 that logs each reward component from info dict to TensorBoard.
    We assume your step() returns info keys:
      - "penalty_reduction"
      - "valid_assign"
      - "preference"
      - "am_coverage"
      - "weekly_hours"
      - "final_penalty"
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Which keys in info we expect
        self._keys = [
            "penalty_reduction",
            "valid_assign",
            "preference",
            "am_coverage",
            # "weekly_hours",
            "final_penalty"
        ]

    def _on_step(self) -> bool:
        # stable-baselines3 makes a list of infos from each env in self.locals["infos"]
        infos = self.locals.get("infos", [])

        if len(infos) == 0:
            return True

        # For each key, collect its values across all parallel envs in this step
        for key in self._keys:
            # Gather all values for this key; default to 0.0 if missing
            vals = [info.get(key, 0.0) for info in infos]
            # Compute mean across environments, since SB3 runs n_envs in parallel
            mean_val = float(np.mean(vals))
            # Log it under “reward/<component>”
            self.logger.record(f"reward/{key}", mean_val)

        return True

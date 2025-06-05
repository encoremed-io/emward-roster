# import os
# import pandas as pd
# import numpy as np
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines3.common.callbacks import EvalCallback

# from build_model import load_nurse_profiles, load_shift_preferences
# from nurse_env import NurseRosteringEnv

# def make_env_factory(num_days):
#     """
#     Returns a function that when called builds a Monitor-wrapped NurseRosteringEnv
#     for the given horizon (num_days), slicing your real preference sheet.
#     """
#     def _make_env():
#         # 1) Load full sheets
#         profiles = load_nurse_profiles("data/nurse_profiles.xlsx")
#         prefs    = load_shift_preferences("data/nurse_preferences.xlsx")

#         # 2) Randomly select a `num_days` slice from available date columns
#         all_dates = prefs.columns.tolist()
#         max_start = len(all_dates) - num_days
#         if max_start < 0:
#             raise ValueError(f"Preferences sheet only has {len(all_dates)} days, but you asked for {num_days}")

#         start_idx = np.random.randint(0, max_start + 1)
#         selected_dates = all_dates[start_idx : start_idx + num_days]
#         prefs = prefs[selected_dates]
#         if prefs.shape[1] < num_days:
#             raise ValueError(f"Preferences sheet only has {prefs.shape[1]} days, but you asked for {num_days}")

#         # 3) Derive start_date from that slice
#         start_date = pd.to_datetime(prefs.columns[0])

#         # 4) Build the env
#         env = NurseRosteringEnv(
#             profiles_df=profiles,
#             preferences_df=prefs,
#             start_date=start_date,
#             max_days=28,
#             active_days=num_days
#         )
#         return Monitor(env, filename=None)
#     return _make_env

# if __name__ == "__main__":
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("eval_logs", exist_ok=True)

#     n_envs = 8
#     horizons = [7, 14, 28]
#     timesteps_map = {7: 200_000, 14: 600_000, 28: 800_000}
#     prev_model = None

#     for num_days in horizons:
#         print(f"\n=== TRAINING on {num_days}-day windows ===")

#         # 1) Make the factory and spawn 8 Subproc workers
#         make_env = make_env_factory(num_days)
#         vec_env  = SubprocVecEnv([make_env] * n_envs)

#         # 2) Eval env must be same VecEnv type to avoid warnings
#         eval_vec_env = SubprocVecEnv([make_env] * n_envs)

#         # 3) Set up EvalCallback
#         eval_callback = EvalCallback(
#             eval_vec_env,
#             best_model_save_path=f"models/ppo_nurse_{num_days}d/",
#             log_path=f"eval_logs/{num_days}d/",
#             eval_freq=10_000,
#             n_eval_episodes=10,
#             deterministic=True,
#         )

#         # 4) Create PPO if no prev model
#         if prev_model is None:
#             model = PPO(
#                 policy="MlpPolicy",
#                 env=vec_env,
#                 verbose=1,
#                 tensorboard_log="./tb_logs",
#                 seed=42,
#                 learning_rate=0.001,
#                 ent_coef=0.005,
#                 gamma=0.99,
#                 n_steps=2048,
#                 batch_size=512,
#                 n_epochs=10,
#                 clip_range=0.2,
#                 policy_kwargs=dict(net_arch=[256, 256]) 
#             )
#         else:
#             model = PPO.load(
#                 prev_model,
#                 env=vec_env,
#                 tensorboard_log="./tb_logs",
#                 seed=42,
#                 learning_rate=0.001,
#                 ent_coef=0.005,
#                 gamma=0.99,
#                 n_steps=2048,
#                 batch_size=512,
#                 n_epochs=10,
#                 clip_range=0.2,
#                 policy_kwargs=dict(net_arch=[256, 256])
#             )

#         model.learn(
#             total_timesteps=timesteps_map[num_days],
#             tb_log_name=f"PPO_{num_days}d",
#             callback=eval_callback
#         )

#         # 5) Save best-found and final policy
#         prev_model = f"models/ppo_nurse_{num_days}d/best_model"
#         model.save(prev_model)
#         print(f"✅ Saved policy for {num_days}-day horizon")


import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from utils.loader import *
from nurse_env import NurseRosteringEnv
from callbacks.reward_logger import RewardComponentLogger

# ------------------------
# Schedule helper funcs
# ------------------------

np.random.seed(42)
profiles_full = load_nurse_profiles("data/nurse_profiles.xlsx")
preferences_full = load_shift_preferences("data/nurse_preferences.xlsx")

class EntropyDecayCallback(BaseCallback):
    """
    Linearly decay ent_coef from start_coef to end_coef over total_timesteps
    """
    def __init__(self, start_coef: float, end_coef: float, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total_timesteps)
        new_coef = self.start_coef + frac * (self.end_coef - self.start_coef)
        setattr(self.model, "ent_coef", new_coef)
        return True

# ------------------------
# Environment factories
# ------------------------

def make_train_env_factory(num_days):
    def _make_env():
        profiles = profiles_full.copy()
        prefs = preferences_full.copy()
        all_dates = prefs.columns.tolist()
        max_start = len(all_dates) - num_days
        start_idx = np.random.randint(0, max_start + 1)
        selected = all_dates[start_idx : start_idx + num_days]
        prefs = prefs[selected]
        start_date = pd.to_datetime(prefs.columns[0])
        env = NurseRosteringEnv(
            profiles_df=profiles,
            preferences_df=prefs,
            start_date=start_date,
            active_days=num_days
        )
        return Monitor(env)
    return _make_env


def compute_eval_starts(total_days: int, window: int, n_windows: int = 8):
    max_start = total_days - window
    if max_start <= 0:
        return [0]
    k = min(n_windows, max_start + 1)
    return [int(round(i * max_start / (k - 1))) for i in range(k)]


def make_eval_env_factory(num_days):
    prefs_full = preferences_full.copy()
    total_days = len(prefs_full.columns)
    starts = compute_eval_starts(total_days, num_days, n_windows=8)

    def _make_env():
        profiles = profiles_full.copy()
        prefs    = preferences_full.copy()
        all_dates = prefs.columns.tolist()

        # Pop the front element, append to back → cycle in shuffled order
        idx = starts.pop(0)
        starts.append(idx)

        selected = all_dates[idx : idx + num_days]
        prefs = prefs[selected]
        start_date = pd.to_datetime(selected[0])

        env = NurseRosteringEnv(
            profiles_df=profiles,
            preferences_df=prefs,
            start_date=start_date,
            active_days=num_days
        )
        return Monitor(env)
    return _make_env


def lr_with_optional_warmup(start_lr: float, end_lr: float, total_steps: int, warmup_steps: int = 0):
    if warmup_steps == 0:
        def pure_linear(progress_remaining: float) -> float:
            return end_lr + progress_remaining * (start_lr - end_lr)
        return pure_linear

    warmup_frac = warmup_steps / total_steps

    def _schedule(progress_remaining: float) -> float:
        if progress_remaining > (1.0 - warmup_frac):
            return end_lr
        else:
            eff_progress = progress_remaining / (1.0 - warmup_frac)
            return end_lr + (eff_progress * (start_lr - end_lr))
    
    return _schedule

# ------------------------
# Training loop
# ------------------------

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("eval_logs", exist_ok=True)

    n_envs = 8
    # horizons = [7, 14, 28]
    timesteps_map = {7: 400_000, 14: 600_000, 28: 800_000}
    prev_model_path = None  # Will hold the file path of the best model

    horizons = [7]
    for num_days in horizons:
        print(f"\n=== TRAINING on {num_days}-day windows ===")

        # 1) Build train/environment
        train_factory = make_train_env_factory(num_days)
        vec_env = SubprocVecEnv([train_factory] * n_envs)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # 2) Build fixed eval environment
        eval_factory = make_eval_env_factory(num_days)
        eval_vec_env = SubprocVecEnv([eval_factory] * n_envs)
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False

        # 3) Set up EvalCallback
        eval_cb = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"models/ppo_nurse_{num_days}d/",
            log_path=f"eval_logs/{num_days}d/",
            eval_freq=10_000,
            n_eval_episodes=40,
            deterministic=True
        )

        # 4) Adaptive hyperparameters
        match num_days:
            case 7:
                start_lr, end_lr, start_coef, end_coef, n_steps, batch_size, n_epochs = 3e-4, 1e-5, 1e-2, 1e-3, 4096, 512, 10
                warmup_steps = 0
            case 14:
                start_lr, end_lr, start_coef, end_coef, n_steps, batch_size, n_epochs = 3e-4, 1e-5, 7e-3, 1e-4, 4096, 1024, 3
                warmup_steps = 50_000
            case _:  # 28 days
                start_lr, end_lr, start_coef, end_coef, n_steps, batch_size, n_epochs = 3e-4, 1e-5, 1e-2, 1e-4, 4096, 1024, 3
                warmup_steps = 50_000

        total_steps = timesteps_map[num_days]
        schedule_fn = lr_with_optional_warmup(start_lr, end_lr, total_steps, warmup_steps)

        # 5) Instantiate a fresh EntropyDecayCallback for THIS horizon
        ent_decay_cb = EntropyDecayCallback(
            start_coef=start_coef,
            end_coef=end_coef,
            total_timesteps=timesteps_map[num_days],
            verbose=0
        )

        reward_logger = RewardComponentLogger(verbose=0)

        # 6) Combine callbacks
        cb_list = CallbackList([eval_cb, ent_decay_cb, reward_logger])

        # 7) Create or load the PPO model
        if prev_model_path is None:
            # 7-day horizon: instantiate from scratch
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                tensorboard_log="./tb_logs",
                seed=42,
                learning_rate=schedule_fn,
                ent_coef=start_coef,
                gamma=0.99,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256])
            )
        else:
            # Subsequent horizons: load a brand-new PPO instance with new hyperparameters
            model = PPO.load(
                prev_model_path,     # load weights from previous best
                env=vec_env,         # attach the new VecEnv
                tensorboard_log="./tb_logs",
                seed=42,
                learning_rate=schedule_fn,
                ent_coef=start_coef,
                gamma=0.99,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256])
            )
            # No need to call set_parameters(), because PPO.load(...) already built a new buffer 
            # (of size `n_steps`) and loaded all weights/optimizer state.

        # ── 8) TWO-PHASE TRAINING: warm-up (if requested) + main schedule ──
        print(f"  • Training: {total_steps} steps (warmup={warmup_steps})")
        model.learn(
            total_timesteps=total_steps,
            tb_log_name=f"PPO_{num_days}d",
            callback=cb_list
        )

        # 9) Save best & final models to disk
        best_path = f"models/ppo_nurse_{num_days}d/best_model"
        model.save(best_path)
        prev_model_path = best_path   # -> used in the next loop iteration

        print(f"✅ Saved policy for {num_days}-day horizon")

        vec_env.close()
        eval_vec_env.close()


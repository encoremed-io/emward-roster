import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from build_model import load_nurse_profiles, load_shift_preferences
from nurse_env import NurseRosteringEnv

def make_env_factory(num_days):
    """
    Returns a function that when called builds a Monitor-wrapped NurseRosteringEnv
    for the given horizon (num_days), slicing your real preference sheet.
    """
    def _make_env():
        # 1) Load full sheets
        profiles = load_nurse_profiles("data/nurse_profiles.xlsx")
        prefs    = load_shift_preferences("data/nurse_preferences.xlsx")

        # 2) Randomly select a `num_days` slice from available date columns
        all_dates = prefs.columns.tolist()
        max_start = len(all_dates) - num_days
        if max_start < 0:
            raise ValueError(f"Preferences sheet only has {len(all_dates)} days, but you asked for {num_days}")

        start_idx = np.random.randint(0, max_start + 1)
        selected_dates = all_dates[start_idx : start_idx + num_days]
        prefs = prefs[selected_dates]
        if prefs.shape[1] < num_days:
            raise ValueError(f"Preferences sheet only has {prefs.shape[1]} days, but you asked for {num_days}")

        # 3) Derive start_date from that slice
        start_date = pd.to_datetime(prefs.columns[0])

        # 4) Build the env
        env = NurseRosteringEnv(
            profiles_df=profiles,
            preferences_df=prefs,
            start_date=start_date,
            max_days=28,
            active_days=num_days
        )
        return Monitor(env, filename=None)
    return _make_env

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("eval_logs", exist_ok=True)

    n_envs = 8
    horizons = [7, 14, 28]
    timesteps_map = {7: 200_000, 14: 600_000, 28: 800_000}
    prev_model = None

    for num_days in horizons:
        print(f"\n=== TRAINING on {num_days}-day windows ===")

        # 1) Make the factory and spawn 8 Subproc workers
        make_env = make_env_factory(num_days)
        vec_env  = SubprocVecEnv([make_env] * n_envs)

        # 2) Eval env must be same VecEnv type to avoid warnings
        eval_vec_env = SubprocVecEnv([make_env] * n_envs)

        # 3) Set up EvalCallback
        eval_callback = EvalCallback(
            eval_vec_env,
            best_model_save_path=f"models/ppo_nurse_{num_days}d/",
            log_path=f"eval_logs/{num_days}d/",
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True,
        )

        # 4) Create PPO if no prev model
        if prev_model is None:
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                tensorboard_log="./tb_logs",
                seed=42,
                learning_rate=0.001,
                ent_coef=0.005,
                gamma=0.99,
                n_steps=2048,
                batch_size=512,
                n_epochs=10,
                clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]) 
            )
        else:
            model = PPO.load(
                prev_model,
                env=vec_env,
                tensorboard_log="./tb_logs",
                seed=42,
                learning_rate=0.001,
                ent_coef=0.005,
                gamma=0.99,
                n_steps=2048,
                batch_size=512,
                n_epochs=10,
                clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256])
            )

        model.learn(
            total_timesteps=timesteps_map[num_days],
            tb_log_name=f"PPO_{num_days}d",
            callback=eval_callback
        )

        # 5) Save best-found and final policy
        prev_model = f"models/ppo_nurse_{num_days}d/best_model"
        model.save(prev_model)
        print(f"✅ Saved policy for {num_days}-day horizon")



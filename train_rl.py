import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from build_model import load_nurse_profiles, load_shift_preferences
from nurse_env import NurseRosteringEnv
import gymnasium as gym

def make_env():
    """
    Factory to create a fresh NurseRosteringEnv wrapped in a Monitor.
    """
    # 1) Load data
    profiles_df    = load_nurse_profiles("data/nurse_profiles.xlsx")
    preferences_df = load_shift_preferences("data/nurse_preferences.xlsx")

    if profiles_df.empty:
        raise ValueError("No nurse profiles loaded during warm start")
    if preferences_df.empty:
        raise ValueError("No shift preferences loaded during warm start")
    
    # 2) Pull out scheduling window
    start_date = preferences_df.columns[0]
    num_days   = len(preferences_df.columns)
    
    # 3) Instantiate and wrap
    env = NurseRosteringEnv(
        profiles_df=profiles_df,
        preferences_df=preferences_df,
        start_date=pd.to_datetime(start_date),
        num_days=num_days
    )
    return Monitor(env, filename=None)  # records rewards/lengths for SB3

if __name__ == "__main__":
    # Create a vectorized env (even a single copy is fine for PPO)
    vec_env = DummyVecEnv([make_env])
    
    # Instantiate the PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tb_logs",  # optional: for TensorBoard
        seed=42
    )
    
    # Train for 200k timesteps
    model.learn(total_timesteps=200_000, tb_log_name="PPO_nurse")
    
    # Save the trained policy
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_nurse_roster")
    print("✅ Saved RL policy to ppo_nurse_roster.zip")

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

from utils.loader import load_nurse_profiles, load_shift_preferences
from nurse_env import NurseRosteringEnv
from callbacks.reward_logger import RewardComponentLogger

# ───────────────
# Hyper-helpers
# ───────────────

SEED = 42

np.random.seed(SEED)
profiles_full    = load_nurse_profiles("data/nurse_profiles.xlsx")
preferences_full = load_shift_preferences("data/nurse_preferences.xlsx")

class EntropyDecayCallback(BaseCallback):
    """Linearly decay ent_coef from start_coef to end_coef over total_timesteps."""
    def __init__(self, start_coef, end_coef, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start_coef      = start_coef
        self.end_coef        = end_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        frac     = min(1.0, self.num_timesteps / self.total_timesteps)
        new_coef = self.start_coef + frac * (self.end_coef - self.start_coef)
        setattr(self.model, "ent_coef", new_coef)
        return True

def lr_with_optional_warmup(start_lr, end_lr, total_steps, warmup_steps=0):
    if warmup_steps == 0:
        return lambda prog: end_lr + prog * (start_lr - end_lr)
    warmup_frac = warmup_steps / total_steps
    def schedule(prog):
        # prog = progress_remaining
        if prog > (1.0 - warmup_frac):
            return end_lr
        eff = prog / (1.0 - warmup_frac)
        return end_lr + eff * (start_lr - end_lr)
    return schedule

# ───────────────
# Env factories
# ───────────────

def make_env_factory(num_days, phase, hp_baseline=None):
    def _make_env():
        profiles = profiles_full.copy()
        prefs    = preferences_full.copy()
        dates    = prefs.columns.tolist()
        start_idx = np.random.randint(0, len(dates) - num_days + 1)
        window    = dates[start_idx : start_idx + num_days]
        prefs     = prefs[window]
        start_dt  = pd.to_datetime(window[0])
        env = NurseRosteringEnv(
            profiles_df    = profiles,
            preferences_df = prefs,
            start_date     = start_dt,
            active_days    = num_days,
            phase          = phase,
            hp_baseline    = hp_baseline
        )
        return Monitor(env)
    return _make_env

def make_eval_env_factory(num_days, phase, hp_baseline=None):
    """Factory that returns a deterministic eval env builder."""
    def _make_env():
        dates    = preferences_full.columns.tolist()
        window   = dates[:num_days]
        prefs    = preferences_full[window]
        start_dt = pd.to_datetime(window[0])
        env = NurseRosteringEnv(
            profiles_df    = profiles_full.copy(),
            preferences_df = prefs,
            start_date     = start_dt,
            active_days    = num_days,
            phase          = phase,
            hp_baseline    = hp_baseline
        )
        return Monitor(env)
    return _make_env

def evaluate_hp(model, num_days, n_episodes=10):
    """Run a phase-1 model to get its average final HP."""
    # 4 parallel eval envs
    env = SubprocVecEnv([make_eval_env_factory(num_days, phase=1)] * 4)
    # normalize observations but keep raw rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env.training   = False
    env.norm_reward = False

    total_hp = 0.0
    for _ in range(n_episodes):
        obs   = env.reset()          # only obs is returned
        done  = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])    # finish when the first sub-env ends

        # fetch the first sub-env's cum_hp
        total_hp += env.env_method('cum_hp')[0]

    env.close()
    return total_hp / n_episodes

# ───────────────
# Training
# ───────────────

if __name__ == "__main__":
    os.makedirs("models",    exist_ok=True)
    os.makedirs("eval_logs", exist_ok=True)

    n_envs       = 8
    horizons     = [7]           # extend to [7,14,28] as needed
    timesteps_map= {7:400_000, 14:600_000, 28:800_000}

    prev_phase2_model = None

    for num_days in horizons:
        print(f"\n→ PHASE 1: Minimize HP penalties on {num_days}-day windows")

        # 1a) Create training VecEnv (phase=1)
        ph1_factory = make_env_factory(num_days, phase=1, hp_baseline=None)
        ph1_vec     = SubprocVecEnv([ph1_factory]*n_envs)
        ph1_vec     = VecNormalize(ph1_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env_ph1 = SubprocVecEnv([make_eval_env_factory(num_days, 1)] * n_envs)
        eval_env_ph1 = VecNormalize(eval_env_ph1, norm_obs=True, norm_reward=False)

        # 1b) Hyper-parameters per horizon
        match num_days:
            case 7:
                start_lr, end_lr, start_coef, end_coef = 3e-4, 1e-5, 1e-2, 1e-3
                n_steps, batch_size, n_epochs = 4096, 512, 10
                warmup_steps = 0
            case 14:
                start_lr, end_lr, start_coef, end_coef = 3e-4, 1e-5, 7e-3, 1e-4
                n_steps, batch_size, n_epochs = 4096, 1024, 3
                warmup_steps = 50_000
            case 28:
                start_lr, end_lr, start_coef, end_coef = 3e-4, 1e-5, 1e-2, 1e-4
                n_steps, batch_size, n_epochs = 4096, 1024, 3
                warmup_steps = 50_000

        # 1c) Callbacks
        eval_cb_ph1    = EvalCallback(
            eval_env_ph1,
            best_model_save_path=f"models/ppo_nurse_{num_days}d/phase1/",
            log_path=f"eval_logs/phase1_{num_days}d/",
            eval_freq=10_000,
            n_eval_episodes=40,
            deterministic=True
        )
        reward_logger  = RewardComponentLogger()
        ent_decay_cb   = EntropyDecayCallback(start_coef, end_coef, timesteps_map[num_days])
        cb_ph1         = CallbackList([eval_cb_ph1, ent_decay_cb, reward_logger])

        # 1d) Build & train Phase 1 PPO
        if prev_phase2_model is None:
            model_ph1 = PPO(
                "MlpPolicy", 
                ph1_vec, 
                verbose=1, 
                seed=SEED,
                learning_rate=lr_with_optional_warmup(start_lr, end_lr, timesteps_map[num_days], warmup_steps),
                ent_coef=start_coef,
                gamma=0.99,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                tensorboard_log="./tb_logs"
            )
        else:
            model_ph1 = PPO.load(
                prev_phase2_model,
                env=ph1_vec, 
                seed=SEED,
                learning_rate=lr_with_optional_warmup(start_lr, end_lr, timesteps_map[num_days], warmup_steps),
                ent_coef=start_coef,
                gamma=0.99,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                tensorboard_log="./tb_logs"
            )

        model_ph1.learn(total_timesteps=timesteps_map[num_days], callback=cb_ph1)
        # model_ph1.save(f"models/ppo_nurse_{num_days}d/phase1_final_model")
        ph1_vec.close()

        # 1e) Evaluate Phase 1’s HP baseline
        hp_baseline = evaluate_hp(model_ph1, num_days, n_episodes=20)
        print(f"→ Phase 1 HP baseline: {hp_baseline:.1f}")

        # ───────────────
        # PHASE 2: Lock in HP ≤ baseline, then minimize LP
        print(f"\n→ PHASE 2: Optimize LP while preserving HP ≤ {hp_baseline:.1f}")

        # 2a) Create training VecEnv (phase=2), keep raw rewards
        ph2_factory = make_env_factory(num_days, phase=2, hp_baseline=hp_baseline)
        ph2_vec     = SubprocVecEnv([ph2_factory]*n_envs)
        ph2_vec     = VecNormalize(ph2_vec, norm_obs=True, norm_reward=False)
        eval_env_ph2 = SubprocVecEnv([make_eval_env_factory(num_days, 2, hp_baseline)] * n_envs)
        eval_env_ph2 = VecNormalize(eval_env_ph2, norm_obs=True, norm_reward=False)

        # 2b) Callbacks
        eval_cb_ph2   = EvalCallback(
            eval_env_ph2,
            best_model_save_path=f"models/ppo_nurse_{num_days}d/phase2/",
            log_path=f"eval_logs/phase2_{num_days}d/",
            eval_freq=10_000,
            n_eval_episodes=20,
            deterministic=True
        )
        reward_logger = RewardComponentLogger()
        ent_decay_cb  = EntropyDecayCallback(start_coef, end_coef, timesteps_map[num_days])
        cb_ph2        = CallbackList([eval_cb_ph2, ent_decay_cb, reward_logger])

        # 2c) Load Phase1 weights into Phase2
        model_ph2 = PPO.load(
            f"models/ppo_nurse_{num_days}d/phase1_best_model.zip",
            env=ph2_vec,
            seed=SEED,
            learning_rate=lr_with_optional_warmup(start_lr, end_lr, timesteps_map[num_days], warmup_steps),
            ent_coef=start_coef,
            gamma=0.99,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            tensorboard_log="./tb_logs"
        )
        model_ph2.learn(total_timesteps=timesteps_map[num_days], callback=cb_ph2)
        # model_ph2.save(f"models/ppo_nurse_{num_days}d/phase2_final_model")
        ph2_vec.close()

        prev_phase2_model = f"models/ppo_nurse_{num_days}d/phase2/best_model.zip"
        print(f"✅ Saved policy for {num_days}-day horizon")

    print("\n✅ All horizons (and both phases) completed.")

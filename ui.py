# ui.py
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import date
from build_model import (
    load_nurse_profiles,
    load_shift_preferences,
    validate_nurse_data,
    build_schedule_model,
)
from nurse_env import NurseRosteringEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import traceback
import logging

def download_excel(df, filename):
    buffer = BytesIO()
    df.to_excel(buffer, index=not df.index.equals(pd.RangeIndex(len(df))), engine="xlsxwriter")
    buffer.seek(0)
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


logging.basicConfig(filename="ui_error.log", level=logging.ERROR)
st.set_page_config(page_title="Nurse Roster Scheduler", layout="wide")
st.title("🩺 Nurse Roster Scheduler")

st.markdown(
    """
    Upload your nurse profiles and preferences, choose a start date and horizon,
    then generate a roster. This uses CP-SAT under the hood (with optional RL warm-start).
    """
)

# ---- Sidebar inputs ----
st.sidebar.header("Inputs")

profiles_file = st.sidebar.file_uploader(
    "Upload nurse_profiles.xlsx", type=["xlsx"], accept_multiple_files=False
)
prefs_file = st.sidebar.file_uploader(
    "Upload nurse_preferences.xlsx", type=["xlsx"], accept_multiple_files=False
)

start_date = st.sidebar.date_input(
    "Schedule start date", value=date.today()
)
num_days = st.sidebar.slider(
    "Number of days", min_value=7, max_value=28, value=14
)

# === Step 3: Warm‐start toggle ===
use_rl = st.sidebar.checkbox(
    "Warm‐start with RL policy",
    value=True,
    help="If checked, will load your PPO model and generate a draft assignment to seed the CP solver."
)

# ---- Main ----
if st.sidebar.button("Generate Schedule"):
    if not profiles_file or not prefs_file:
        st.error("Please upload both the profiles and preferences Excel files.")
    else:
        try:
            # Load profiles
            # If it's an UploadedFile, read into a DataFrame directly,
            # then apply the cleaning steps from load_nurse_profiles.
            if isinstance(profiles_file, BytesIO) or hasattr(profiles_file, "read"):
                df_profiles = pd.read_excel(profiles_file)
                df_profiles['Name'] = df_profiles['Name'].str.strip().str.upper()
            else:
                df_profiles = load_nurse_profiles(profiles_file)

            # Load preferences
            if isinstance(prefs_file, BytesIO) or hasattr(prefs_file, "read"):
                df_prefs = pd.read_excel(prefs_file)
                # replicate load_shift_preferences logic
                df_prefs.rename(columns={df_prefs.columns[0]: 'Name'}, inplace=True)
                df_prefs.set_index('Name', inplace=True)
                cleaned = []
                for col in df_prefs.columns:
                    dt = pd.to_datetime(str(col).strip().split()[-1], format="%Y-%m-%d").date()
                    cleaned.append(dt)
                df_prefs.columns = cleaned
                df_prefs.index = df_prefs.index.str.strip().str.upper()
            else:
                df_prefs = load_shift_preferences(prefs_file)

            # Validate
            validate_nurse_data(df_profiles, df_prefs)

            # === Generate RL warm‐start if requested ===
            rl_assignment = None
            if use_rl:
                try:
                    env = NurseRosteringEnv(
                        profiles_df=df_profiles,
                        preferences_df=df_prefs,
                        start_date=pd.to_datetime(start_date),
                        num_days=num_days
                    )
                    vec_env = DummyVecEnv([lambda: env])
                    model = PPO.load("models/ppo_nurse_roster", env=vec_env)
                except (FileNotFoundError, OSError):
                    st.warning(
                        "⚠️ RL model file `ppo_nurse_roster.zip` not found. "
                        "Falling back to pure CP scheduling."
                    )
                    model = None

                if model is not None:
                    # roll out the policy to build a draft assignment
                    env = NurseRosteringEnv(
                        profiles_df=df_profiles,
                        preferences_df=df_prefs,
                        start_date=pd.to_datetime(start_date),
                        num_days=num_days
                    )
                    obs, _ = env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        if isinstance(action, np.ndarray):
                            action = int(action)
                        obs, _, done, _, _ = env.step(action)
                    rl_assignment = obs.tolist()
                    st.sidebar.success("🔁 Warm-start draft generated via RL")

            # Build schedule
            sched_df, summary_df = build_schedule_model(
                df_profiles,
                df_prefs,
                pd.to_datetime(start_date),
                num_days,
                rl_assignment=rl_assignment
            )

            # Display results
            st.subheader("📅 Generated Schedule")
            st.dataframe(sched_df, use_container_width=True)

            st.subheader("📊 Summary Metrics")
            st.dataframe(summary_df, use_container_width=True)

            # Download buttons
            st.markdown("**Download results:**")
            download_excel(sched_df, "nurse_schedule.xlsx")
            download_excel(summary_df, "nurse_summary.xlsx")

        # except Exception as e:
        #     st.error(f"Error: {e}")
        
        except Exception as e:
            tb = traceback.format_exc()
            st.error(f"Error: {e}")
            st.text_area("Full Traceback", tb, height=300)
            logging.error(tb)
else:
    st.info("Configure inputs in the sidebar and click Generate Schedule.")

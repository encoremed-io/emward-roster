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
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode
from nurse_env import NurseRosteringEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json
import numpy as np
import traceback
import logging

with open('config/constants.json', 'r') as f:
    constants = json.load(f)

SHIFT_LABELS = constants["SHIFT_LABELS"]

logging.basicConfig(filename="ui_error.log", level=logging.ERROR)

st.set_page_config(page_title="Nurse Roster Scheduler", layout="wide")
st.title("🩺 Nurse Roster Scheduler")
st.markdown("""
Upload your nurse profiles and preferences, choose a start date and horizon,
then generate a roster. This uses CP-SAT under the hood (with optional RL warm-start).
""")

def download_excel(df, filename):
    buffer = BytesIO()
    df.to_excel(buffer,
                index=not df.index.equals(pd.RangeIndex(len(df))),
                engine="xlsxwriter")
    buffer.seek(0)
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def show_editable_schedule():
    st.subheader("📅 Schedule (mark ‘EL’ for emergency leave)")
    sched_df = st.session_state.sched_df
    if sched_df is None or sched_df.empty:
        st.write("No schedule data to show.")
        return
    
    disp = sched_df.reset_index().rename(columns={'index': 'Nurse'})
    # st.write("disp preview:", disp.head())
    gb = GridOptionsBuilder.from_dataframe(disp)
    for c in disp.columns:
        if c != "Nurse":
            gb.configure_column(c, editable=True)
    grid = AgGrid(
        disp,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=True,
        height=300,
        
        key="editable_schedule_grid"
    )
    edited = pd.DataFrame(grid["data"]).set_index("Nurse")

    # Collect new EL overrides
    new_overrides = {}
    for nurse in edited.index:
        for col in edited.columns:
            if edited.at[nurse, col] == "EL" and st.session_state.sched_df.at[nurse, col] != "EL":
                day_idx = (pd.to_datetime(col).date() 
                           - st.session_state.start_date.date()).days
                new_overrides[(nurse, day_idx)] = "EL"

    st.session_state.pending_overrides = new_overrides
    st.sidebar.write(f"Pending EL overrides: {len(new_overrides)}")

# Sidebar inputs
st.sidebar.header("Inputs")
profiles_file = st.sidebar.file_uploader("Upload nurse_profiles.xlsx", type=["xlsx"])
prefs_file = st.sidebar.file_uploader("Upload nurse_preferences.xlsx", type=["xlsx"])
start_date = st.sidebar.date_input("Schedule start date", value=date.today())
num_days = st.sidebar.slider("Number of days", 7, 28, 14)
use_rl = st.sidebar.checkbox("Warm-start with RL policy", value=True)

# Store core state
for key, default in {
    "fixed": {},
    "rl_assignment": None,
    "sched_df": None,
    "summary_df": None,
    "df_profiles": None,
    "df_prefs": None,
    "start_date": pd.to_datetime(start_date),
    "num_days": num_days,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Generate Schedule
if st.sidebar.button("Generate Schedule"):
    st.session_state.fixed.clear()
    st.session_state.rl_assignment = None
    st.session_state.start_date = pd.to_datetime(start_date)
    st.session_state.num_days = num_days

    if not profiles_file or not prefs_file:
        st.error("Please upload both the profiles and preferences Excel files.")
        st.stop()

    try:
        # Load profiles
        df_profiles = pd.read_excel(profiles_file)
        df_profiles['Name'] = df_profiles['Name'].str.strip().str.upper()

        # Load preferences
        df_prefs = pd.read_excel(prefs_file)
        df_prefs.rename(columns={df_prefs.columns[0]: 'Name'}, inplace=True)
        df_prefs.set_index('Name', inplace=True)
        df_prefs.columns = [
            pd.to_datetime(str(c).strip().split()[-1], format="%Y-%m-%d").date()
            for c in df_prefs.columns
        ]
        df_prefs.index = df_prefs.index.str.strip().str.upper()

        missing, extra = validate_nurse_data(df_profiles, df_prefs)
        if missing or extra:
            msg = "⚠️ Mismatch:\n"
            if missing: msg += f"Missing in prefs: {missing}\n"
            if extra: msg += f"Extra in prefs: {extra}"
            st.error(msg)
            st.stop()

        st.session_state.df_profiles = df_profiles
        st.session_state.df_prefs = df_prefs

        # RL warm start
        if use_rl:
            env = NurseRosteringEnv(df_profiles, df_prefs, pd.to_datetime(start_date), num_days)
            vec = DummyVecEnv([lambda: env])
            for h in (7, 14, 28):
                if h >= num_days:
                    try:
                        model = PPO.load(f"models/ppo_nurse_{h}d.zip", env=vec)
                        obs, _ = env.reset()
                        done = False
                        while not done:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, done, _, _ = env.step(int(action))
                        st.session_state.rl_assignment = obs.tolist()
                        st.sidebar.success(f"🔁 Warm-start from {h}-day model")
                    except:
                        pass
                    break

        sched, summ = build_schedule_model(
            df_profiles, df_prefs,
            pd.to_datetime(start_date), num_days,
            rl_assignment=st.session_state.rl_assignment,
            fixed_assignments=st.session_state.fixed
        )
        st.session_state.sched_df = sched
        st.session_state.summary_df = summ
        st.session_state.original_sched_df = sched.copy()

    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Error: {e}")
        st.text_area("Traceback", tb, height=200)
        st.stop()

if "show_schedule_expanded" not in st.session_state:
    st.session_state.show_schedule_expanded = False

if st.session_state.sched_df is not None:
    # Use a radio to simulate expander toggle
    if st.session_state.sched_df is not None:
        choice = st.radio(
            "Editable Schedule",
            options=["Hide", "Show"],
            index=1 if st.session_state.show_schedule_expanded else 0,
            horizontal=True
        )
        st.session_state.show_schedule_expanded = (choice == "Show")

    if st.session_state.show_schedule_expanded:
        show_editable_schedule()

    if st.button("🔁 Regenerate with Emergency Leave"):
        orig = st.session_state.original_sched_df
        overrides = st.session_state.pending_overrides or {}
        if not overrides:
            st.warning("No EL overrides to apply.")
            st.stop()

        # find earliest EL day across all overrides
        w0 = min(day_idx for (_, day_idx) in overrides.keys())

        # build a full fixed_assignments dict
        fixed = {}
        orig = st.session_state.original_sched_df

        # 1) freeze days < w0, but only valid single shifts / EL / MC
        for nurse, row in orig.iterrows():
            for i, col in enumerate(orig.columns):
                if i >= w0:
                    break
                val = str(row[col]).strip()
                # only freeze true shifts or EL; skip "MC", "Rest", "AM/PM*", etc.
                if val in SHIFT_LABELS or val.upper() == "EL":
                    fixed[(nurse, i)] = val
                # skip "Rest" or "AM/PM*" etc.

        # 2) add all of your pending EL overrides
        fixed.update(overrides)

        # store it back so next time you keep on growing it
        st.session_state.fixed = fixed

        # now re-solve only the tail
        sched2, summ2 = build_schedule_model(
            st.session_state.df_profiles,
            st.session_state.df_prefs,
            st.session_state.start_date,
            st.session_state.num_days,
            rl_assignment=st.session_state.rl_assignment,
            fixed_assignments=fixed
        )
        st.session_state.sched_df   = sched2
        st.session_state.summary_df = summ2

        st.success(f"Re-solved from day {w0} onward.")
        st.session_state.show_schedule_expanded = False
        st.rerun()

    st.subheader("📅 Final Schedule")
    st.dataframe(st.session_state.sched_df, use_container_width=True)

    st.subheader("📊 Final Summary Metrics")
    st.dataframe(st.session_state.summary_df, use_container_width=True)

    st.markdown("**Download results:**")
    download_excel(st.session_state.sched_df, "nurse_schedule.xlsx")
    download_excel(st.session_state.summary_df, "nurse_summary.xlsx")

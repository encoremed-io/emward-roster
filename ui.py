# ui.py
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
from io import BytesIO
from datetime import date
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode
from nurse_env import NurseRosteringEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import Counter, defaultdict
import traceback
import logging
from build_model import build_schedule_model
from utils.loader import *
from utils.validate import *
from utils.constants import MAX_MC_DAYS_PER_WEEK, DAYS_PER_WEEK

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
    if sched_df.empty:
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
    new_el = {}
    new_mc = {}
    for nurse in edited.index:
        for col in edited.columns:
            val = edited.at[nurse, col].strip().upper()
            old = st.session_state.sched_df.at[nurse, col].strip().upper()
            if val == "EL" and old != "EL":
                day_idx = (pd.to_datetime(col).date() 
                           - st.session_state.start_date.date()).days
                new_el[(nurse, day_idx)] = "EL"
            if val == "MC" and old != "MC":
                day_idx = (pd.to_datetime(col).date() 
                           - st.session_state.start_date.date()).days
                new_mc[(nurse, day_idx)] = "MC"

    st.session_state.pending_el = new_el
    st.session_state.pending_mc = new_mc
    st.sidebar.write(f"Pending EL overrides: {len(new_el)}")
    st.sidebar.write(f"Total EL declarations: {len(st.session_state.all_el_overrides)}")
    st.sidebar.write(f"Pending MC overrides: {len(new_mc)}")
    st.sidebar.write(f"Total MC declarations: {len(st.session_state.all_mc_overrides)}")
    # all_mc_overrides and all_el_overrides only updated after validation

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
    "all_el_overrides": {},
    "all_mc_overrides": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Generate Schedule
if st.sidebar.button("Generate Schedule"):
    st.session_state.fixed.clear()
    st.session_state.rl_assignment = None
    st.session_state.start_date = pd.to_datetime(start_date)
    st.session_state.num_days = num_days
    st.session_state.all_el_overrides = {}
    st.session_state.all_mc_overrides = {}

    if not profiles_file or not prefs_file:
        st.error("Please upload both the profiles and preferences Excel files.")
        st.stop()

    try:
        # Load profiles
        df_profiles = load_nurse_profiles(profiles_file)
        df_prefs = load_shift_preferences(prefs_file)

        missing, extra = validate_nurse_data(df_profiles, df_prefs)
        if missing or extra:
            msg = "⚠️ Mismatch between nurse profiles and preferences:\n\n"
            if missing: msg += f"Not found in preferences: {sorted(missing)}\n"
            if extra: msg += f"Not found in profiles: {sorted(extra)}"
            st.error(msg)
            st.stop()

        st.session_state.df_profiles = df_profiles
        st.session_state.df_prefs = df_prefs

        # RL warm start
        if use_rl:
            for h in (7, 14, 28):
                if h >= num_days and os.path.exists(f"models/ppo_nurse_{h}d/phase1/best_model.zip") and os.path.exists(f"models/ppo_nurse_{h}d/phase2/best_model.zip"):
                    # 1) Phase 1 rollout to minimize high-priority penalties
                    env1 = NurseRosteringEnv(
                        st.session_state.df_profiles,
                        st.session_state.df_prefs,
                        st.session_state.start_date,
                        active_days=st.session_state.num_days,
                        phase=1,
                        hp_baseline=None
                    )
                    vec1 = DummyVecEnv([lambda: env1])
                    model1 = PPO.load(f"models/ppo_nurse_{h}d/phase1/best_model.zip", env=vec1)
                    obs = env1.reset()[0]            # unpack (obs,info)
                    done = False
                    while not done:
                        action, _ = model1.predict(obs, deterministic=True)
                        obs, _, done, _, _ = env1.step(int(action))
                    # at the end of rollout, env1.cum_hp holds the minimized HP baseline
                    hp_baseline = env1.cum_hp

                    # 2) Phase 2 rollout to honor HP ≤ baseline, then optimize LP
                    env2 = NurseRosteringEnv(
                        st.session_state.df_profiles,
                        st.session_state.df_prefs,
                        st.session_state.start_date,
                        active_days=st.session_state.num_days,
                        phase=2,
                        hp_baseline=hp_baseline
                    )
                    vec2 = DummyVecEnv([lambda: env2])
                    model2 = PPO.load(f"models/ppo_nurse_{h}d/phase2/best_model.zip", env=vec2)
                    obs = env2.reset()[0]
                    done = False
                    while not done:
                        action, _ = model2.predict(obs, deterministic=True)
                        obs, _, done, _, _ = env2.step(int(action))

                    # 3) extract the final assignment as flat list for CP-SAT warm start
                    #    (obs is a flattened vector of 0/1 assignments)
                    st.session_state.rl_assignment = obs.tolist()
                    st.sidebar.success(f"🔁 RL warm-start from {h}-day model")
                    break
                else:
                    st.sidebar.error(f"⚠️ No matching RL model found!")
                    st.info("🔁 Using fallback without warm start ...")
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

        st.session_state.show_schedule_expanded = False
        st.session_state["editable_toggle"] = "Hide"
        st.rerun()

    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Error: {e}")
        st.text_area("Traceback", tb, height=200)
        st.stop()

if "show_schedule_expanded" not in st.session_state:
    st.session_state.show_schedule_expanded = False

if st.session_state.sched_df is not None:
    # Use a radio to simulate expander toggle
    choice = st.radio(
        "Editable Schedule",
        options=["Hide", "Show"],
        index=1 if st.session_state.show_schedule_expanded else 0,
        horizontal=True,
        key="editable_toggle"
    )
    st.session_state.show_schedule_expanded = (choice == "Show")

    if st.session_state.show_schedule_expanded:
        show_editable_schedule()

        if st.button("🔁 Regenerate with All Overrides"):
            el_overrides = st.session_state.get("all_el_overrides") or {}
            mc_overrides = st.session_state.get("all_mc_overrides") or {}
            pending_el = st.session_state.get("pending_el") or {}
            pending_mc = st.session_state.get("pending_mc") or {}

            # check mc override validity
            if pending_mc:

                # 1) collect every MC day: initial schedule, old overrides, new overrides
                orig = st.session_state.original_sched_df
                init = {(n,i) for n,r in orig.iterrows() 
                        for i,col in enumerate(orig.columns) 
                        if str(r[col]).strip().upper()=="MC"}
                all_mc = init | set(mc_overrides) | set(pending_mc)

                # 2) group days by nurse
                nd = defaultdict(set)
                for n,d in all_mc: 
                    nd[n].add(d)

                # 3) validate
                errs = []
                for n, days in nd.items():
                    # weekly counts
                    for wk, cnt in Counter(d//DAYS_PER_WEEK for d in days).items():
                        if cnt > MAX_MC_DAYS_PER_WEEK:
                            errs.append(f"{n}: {cnt} MCs in week {wk+1} (max {MAX_MC_DAYS_PER_WEEK})")
                    # consecutive runs
                    seq = sorted(days)
                    if any(seq[i+2] - seq[i] < 3 for i in range(len(seq)-2)):
                        errs.append(f"{n}: >2 consecutive MC days")

                if errs:
                    st.session_state.pending_mc.clear()
                    st.error("❌ Invalid MC override:\n" + "\n".join(errs))
                    st.info("Click 'hide' and 'show' to try again")
                    st.stop()

            if not pending_el and not pending_mc:
                st.info("No overrides to apply.")

            else:
                # merge pending into cumulative
                st.session_state.all_el_overrides.update(st.session_state.pending_el)
                st.session_state.all_mc_overrides.update(st.session_state.pending_mc)
                el_overrides = st.session_state.all_el_overrides
                mc_overrides = st.session_state.all_mc_overrides

                # find earliest override day across pending overrides
                all_days = [d for (_,d) in pending_el.keys()] + [d for (_,d) in pending_mc.keys()]
                w0 = min(all_days)
                # st.write(w0)

                # clear pending
                st.session_state.pending_el.clear()
                st.session_state.pending_mc.clear()

                # build fixed assignments from most updated schedule
                fixed = {}
                orig = st.session_state.sched_df.copy()

                # freeze all values for days < w0
                for nurse, row in orig.iterrows():
                    for i, col in enumerate(orig.columns):
                        if i >= w0:
                            break
                        val = str(row[col]).strip()
                        fixed[(nurse, i)] = val

                # add all of your pending overrides, then store it back to keep growing next time
                fixed.update(el_overrides)
                fixed.update(mc_overrides)
                st.session_state.fixed = fixed

                # re-solve starting from earliest EL day (included)
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

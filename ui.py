# ui.py
"""Streamlit UI for Nurse Roster Scheduler."""
import sys
import os

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from datetime import date, time, datetime
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode
from collections import Counter, defaultdict
import traceback
import logging
from scheduler.builder import build_schedule_model
from utils.loader import *
from utils.validate import *
from utils.constants import *
from utils.download import download_excel
from utils.shift_utils import shift_duration_minutes
from exceptions.custom_errors import *
from schemas.schedule.generate import StaffAllocations
import requests

LOG_PATH = os.path.join(os.path.dirname(__file__), "ui_error.log")

try:
    with open(LOG_PATH, "w") as f:
        pass
except FileNotFoundError:
    os.makedirs(os.path.dirname(LOG_PATH))
    with open(LOG_PATH, "w") as f:
        pass

# logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),  # Optional: show logs in terminal
    ],
    force=True,  # Ensure this config takes effect even if others were set
)

logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
logging.getLogger("watchdog").setLevel(logging.WARNING)

CUSTOM_ERRORS = (
    NoFeasibleSolutionError,
    InvalidMCError,
    InvalidALError,
    ConsecutiveMCError,
    ConsecutiveALError,
    InputMismatchError,
    InvalidPreviousScheduleError,
    InvalidPrioritySettingError,
    FileReadingError,
    FileContentError,
)

st.set_page_config(page_title="Nurse Roster Scheduler", layout="wide")
st.title("🩺 Nurse Roster Scheduler")
st.markdown(
    """
Upload your nurse profiles and preferences, choose a start and end date, enter schedule parameters,
then generate a roster.\n 
This uses CP-SAT under the hood.
"""
)


# def show_editable_schedule():
#     logging.info("Showing editable schedule")
#     """
#     Show the schedule in an editable grid. The user can type "EL" for emergency leave or "MC" for medical leave.
#     Invalid inputs are detected and an error message is shown. New EL/MC overrides are collected and stored in the session state.
#     """
#     st.subheader("📅 Editable Schedule")
#     st.caption("Type 'EL' for emergency leave or 'MC' for medical leave")
#     sched_df = st.session_state.sched_df
#     if sched_df.empty:
#         st.write("No schedule data to show.")
#         return

#     disp = sched_df.reset_index().rename(columns={"index": "Nurse"})
#     disp = disp.map(lambda x: ", ".join(x) if isinstance(x, list) else x)

#     gb = GridOptionsBuilder.from_dataframe(disp)
#     for c in disp.columns:
#         if c != "Nurse":
#             gb.configure_column(c, editable=True)

#     grid = AgGrid(
#         disp,
#         gridOptions=gb.build(),
#         update_mode=GridUpdateMode.MODEL_CHANGED,
#         fit_columns_on_grid_load=True,
#         height=300,
#         key="editable_schedule_grid",
#     )
#     edited = pd.DataFrame(grid["data"]).set_index("Nurse")
#     st.write("Edited result", edited.head())
#     invalid = []  # error handling

#     # Collect new EL/MC overrides
#     new_el = {}
#     new_mc = {}

#     for nurse in edited.index:
#         for col in edited.columns:
#             val = edited.at[nurse, col]
#             # print("$$$$$$:\n", val)
#             # print("Error! Exiting.")
#             # sys.exit()

#             # # Normalize old value
#             # old_val = st.session_state.sched_df.at[nurse, col]
#             # if isinstance(old_val, list):
#             #     old = old_val[0].strip().upper() if old_val else ""
#             # else:
#             #     old = str(old_val).strip().upper()

#             # # Normalize new value
#             # if isinstance(val, list):
#             #     val = val[0] if val else None  # get first item or None
#             # if val is None or (not isinstance(val, (list, tuple)) and pd.isna(val)):
#             #     val = None
#             # else:
#             #     val = str(val).strip().upper()

#             def normalize_cell(v):
#                 if isinstance(v, list):
#                     v = v[0] if v else None
#                 if v is None or (not isinstance(v, (list, tuple)) and pd.isna(v)):
#                     return None
#                 return str(v).strip().upper()

#             old_val = st.session_state.sched_df.at[nurse, col]
#             old = normalize_cell(old_val)
#             val = normalize_cell(val)

#             # Skip if no change
#             if val == old:
#                 continue

#             # Allow only EL or MC
#             if val not in {"EL", "MC"}:
#                 st.warning(
#                     f"⚠️ Invalid input for {nurse} on {col}: '{val}' (only 'EL' or 'MC' are allowed)."
#                 )
#                 st.stop()

#             # Handle EL and MC override
#             day_idx = (pd.to_datetime(col).date() - st.session_state.start_date).days
#             if val == "EL":
#                 new_el[(nurse, day_idx)] = "EL"
#             elif val == "MC":
#                 new_mc[(nurse, day_idx)] = "MC"

#     if invalid:
#         msg = [
#             "⚠️ Some inputs are empty or invalid (not 'EL', 'MC', or the original shift). Please correct them:\n"
#         ]
#         for nurse, col, val, old in invalid:
#             msg.append(f"     • {nurse} on {col}: '{val}' (Original shift: '{old}')\n")
#         st.warning("\n".join(msg))
#         st.stop()

#     st.session_state.pending_el = new_el
#     st.session_state.pending_mc = new_mc

#     if not invalid:
#         st.session_state.sched_df = edited.copy()

#     st.sidebar.write(f"Pending EL overrides: {len(new_el)}")
#     st.sidebar.write(f"Total EL declarations: {len(st.session_state.all_el_overrides)}")
#     st.sidebar.write(f"Pending MC overrides: {len(new_mc)}")
#     st.sidebar.write(f"Total MC declarations: {len(st.session_state.all_mc_overrides)}")
#     # all_mc_overrides and all_el_overrides only updated after validation


# Show the updated schedule
def show_editable_schedule():
    logging.info("Showing editable schedule")
    st.subheader("📅 Editable Schedule")
    st.caption("Type 'EL' for emergency leave or 'MC' for medical leave")

    sched_df = st.session_state.sched_df
    if sched_df.empty:
        st.write("No schedule data to show.")
        return

    # 1. Display-safe version for grid (stringified values)
    disp = sched_df.reset_index().rename(columns={"index": "Nurse"})
    disp = disp.map(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x) if pd.notna(x) else ""
    )

    gb = GridOptionsBuilder.from_dataframe(disp)
    for c in disp.columns:
        if c != "Nurse":
            gb.configure_column(c, editable=True)

    # 2. Show grid
    grid = AgGrid(
        disp,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        height=300,
        key="editable_schedule_grid",
    )

    # 3. Get edited grid and compare to displayed version
    edited = pd.DataFrame(grid["data"]).set_index("Nurse")
    original_disp = disp.set_index("Nurse")

    def normalize(val):
        return str(val).strip().upper() if pd.notna(val) else ""

    # 4. Track overrides
    new_el = {}
    new_mc = {}

    for nurse in edited.index:
        for col in edited.columns:
            old = normalize(original_disp.at[nurse, col])
            new = normalize(edited.at[nurse, col])

            # ✅ Only validate truly edited cells
            if new == old:
                continue

            if new not in {"EL", "MC"}:
                st.warning(
                    f"⚠️ Invalid input for {nurse} on {col}: '{new}' (only 'EL' or 'MC' allowed)."
                )
                st.stop()

            day_idx = (pd.to_datetime(col).date() - st.session_state.start_date).days
            if new == "EL":
                new_el[(nurse, day_idx)] = "EL"
            elif new == "MC":
                new_mc[(nurse, day_idx)] = "MC"

    # ✅ Save new overrides and edited state
    st.session_state.pending_el = new_el
    st.session_state.pending_mc = new_mc
    st.session_state.sched_df = edited.copy()

    st.sidebar.write(f"Pending EL overrides: {len(new_el)}")
    st.sidebar.write(f"Pending MC overrides: {len(new_mc)}")

    # trigger swap suggestions
    if new_el or new_mc:
        fetch_swap_suggestions(new_el, new_mc)


# Show the swap suggestions format
def fetch_swap_suggestions(new_el, new_mc):
    target_nurse_ids = list(
        set(n for n, _ in list(new_el.keys()) + list(new_mc.keys()))
    )
    print("wootie:\n", target_nurse_ids)
    print("Error! Exiting.")
    sys.exit()
    target_shifts = []
    # for nurse in target_nurse_ids:
    #     shifts = []

    #     for (n, day), val in new_el.items():
    #         if n == nurse:
    #             shifts.append({
    #                 "date": (st.session_state.start_date + timedelta(days=day)).strftime("%Y-%m-%d"),
    #                 "shiftTypeId": ["AM"]  # or "PM" / "Night" depending on your logic
    #             })

    #     for (n, day), val in new_mc.items():
    #         if n == nurse:
    #             shifts.append({
    #                 "date": (st.session_state.start_date + timedelta(days=day)).strftime("%Y-%m-%d"),
    #                 "shiftTypeId": ["AM"]  # same here
    #             })

    #     target_shifts.append({
    #         "nurseId": nurse,
    #         "targetShift": shifts
    #     })

    # # Prepare request payload
    # payload = {
    #     "targetNurseId": target_shifts,
    #     "settings": st.session_state.schedule_settings,
    #     "roster": st.session_state.roster_data,
    # }

    # try:
    #     res = requests.post("http://localhost:8000/swap/suggestions", json=payload)
    #     res.raise_for_status()
    #     data = res.json()
    #     st.session_state.swap_suggestions = data["results"]
    #     st.success("Swap suggestions fetched!")
    # except Exception as e:
    #     st.error(f"❌ Failed to fetch suggestions: {e}")


# Sidebar inputs
st.sidebar.header("Inputs")
profile_input_mode = st.sidebar.radio(
    "Nurse Profiles Input Method", options=["Upload File", "Manual Entry"], index=0
)

if profile_input_mode == "Upload File":
    profiles_file = st.sidebar.file_uploader("Upload Nurse Profiles", type=["xlsx"])
    if not profiles_file:
        st.info("Please upload the nurse profiles Excel file.")
        # st.stop()
    try:
        df_profiles = load_nurse_profiles(profiles_file)
    except CUSTOM_ERRORS as e:
        st.error(str(e))
        st.stop()
    senior_count = junior_count = 0
else:
    profiles_file = None
    # Generate DataFrame for seniors and juniors
    senior_count = int(st.sidebar.number_input("Number of Senior Nurses", min_value=1))
    junior_count = int(st.sidebar.number_input("Number of Junior Nurses", min_value=1))
    senior_names = [f"S{str(i).zfill(2)}" for i in range(senior_count)]
    junior_names = [f"J{str(i).zfill(2)}" for i in range(junior_count)]
    names = senior_names + junior_names
    titles = ["Senior"] * senior_count + ["Junior"] * junior_count
    years_exp = [3] * senior_count + [
        0
    ] * junior_count  # Example: seniors have ≥3 years, juniors 0
    df_profiles = pd.DataFrame(
        {
            "Name": names,
            "Title": titles,
            "YearsExperience": years_exp,
            "DoubleShift": [False] * (senior_count + junior_count),
        }
    )

# print("df_profiles:\n", df_profiles)
# print("Error! Exiting.")
# sys.exit()
# prefs_file = st.sidebar.file_uploader(
#     "Upload Nurse Preferences (Optional)",
#     type=["xlsx"],
#     help=("• Preferences file only applies to specified dates in file."),
# )
prefs_file: Union[str, Path, bytes, IO, None] = None

# training_shifts_file = st.sidebar.file_uploader(
#     "Upload Training Shifts (Optional)",
#     type=["xlsx"],
#     help=(
#         "• If provided, these shifts will be excluded for the nurse for the schedule on stated days.\n\n"
#         "• Training Shifts file only appied to specified dates in file."
#     ),
# )
training_shifts_file: Union[str, Path, bytes, IO, None] = None

# prev_sched_file = st.sidebar.file_uploader(
#     "Upload Previous Schedule (Optional)",
#     type=["xlsx"],
#     help=(
#         "• Upload a previous schedule to ensure continuity.\n\n"
#         "• Previous Schedule end date must be before current schedule start date."
#     ),
# )
prev_sched_file: Union[str, Path, bytes, IO, None] = None

start_date = pd.to_datetime(
    st.sidebar.date_input("Schedule start date", value=date.today(), key="start_date")
)
end_date = pd.to_datetime(
    st.sidebar.date_input("Schedule end date", value=date.today(), key="end_date")
)
# num_days = st.sidebar.slider("Number of days", 7, 28, 14)
num_days = (end_date - start_date).days + 1

# --- Add dynamic shift durations --- #
st.sidebar.markdown("### Shift Durations")
st.sidebar.subheader("☀️ AM Shift")  # 7AM - 2PM
am_start = st.sidebar.time_input("AM Start", time(7, 0), key="am_start")
am_end = st.sidebar.time_input("AM End", time(14, 0), key="am_end")

st.sidebar.subheader("🌇 PM Shift")  # 2PM - 9PM
pm_start = st.sidebar.time_input("PM Start", time(14, 0), key="pm_start")
pm_end = st.sidebar.time_input("PM End", time(21, 0), key="pm_end")

st.sidebar.subheader("🌙 Night Shift")  # 9PM - 7AM
night_start = st.sidebar.time_input("Night Start", time(21, 0), key="night_start")
night_end = st.sidebar.time_input("Night End", time(7, 0), key="night_end")

shift_durations = [
    shift_duration_minutes(am_start, am_end),
    shift_duration_minutes(pm_start, pm_end),
    shift_duration_minutes(night_start, night_end),
]

# --- Add dynamic scheduling parameters here ---
st.sidebar.markdown("### Schedule Parameters")
min_nurses_per_shift = st.sidebar.number_input(
    "Minimum nurses per shift",
    min_value=1,
    value=MIN_NURSES_PER_SHIFT,
    key="min_nurses_per_shift",
)
min_seniors_per_shift = st.sidebar.number_input(
    "Minimum seniors per shift",
    min_value=0,
    value=MIN_SENIORS_PER_SHIFT,
    key="min_seniors_per_shift",
)

with st.sidebar.expander("Staff Allocation", expanded=True):
    senior_staff_allocation = st.radio(
        "Senior Staff",
        options=["Yes", "No"],
        horizontal=True,
        index=1,
        key="senior_staff_allocation",
    )

    senior_staff_percentage = st.slider(
        "Percentage of Senior Nurses",
        min_value=50,
        max_value=100,
        help="Aim for at least this % of nurses working shift.",
        key="senior_staff_percentage",
        disabled=(senior_staff_allocation == "No"),
    )

    senior_staff_allocation_refinement = st.radio(
        "Refine Senior Staff Allocation",
        options=["Yes", "No"],
        horizontal=True,
        index=1,
        key="senior_staff_allocation_refinement",
        disabled=(senior_staff_allocation == "No"),
    )

    senior_staff_allocation_refinement_value = st.number_input(
        "Refinement Value %",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Adjust the percentage of senior nurses allocated to shifts.",
        key="senior_staff_allocation_refinement_value",
        disabled=(
            senior_staff_allocation_refinement == "No"
            or senior_staff_allocation == "No"
        ),
    )

st.sidebar.subheader("Working Hours")
min_acceptable_weekly_hours = st.sidebar.number_input(
    "Min weekly hours",
    min_value=1,
    value=MIN_ACCEPTABLE_WEEKLY_HOURS,
    help="The minimum number of hours a nurse must be scheduled for each week. MC and EL days reduce this cap.",
    key="min_acceptable_weekly_hours",
    # disabled=pref_weekly_hours_hard,
)

max_weekly_hours = st.sidebar.number_input(
    "Max weekly hours",
    min_value=1,
    value=MAX_WEEKLY_HOURS,
    help="The maximum number of hours a nurse can be scheduled to work in a week. MC and EL days reduce this cap.",
    key="max_weekly_hours",
)
preferred_weekly_hours = st.sidebar.number_input(
    "Preferred weekly hours",
    min_value=1,
    value=PREFERRED_WEEKLY_HOURS,
    help="The ideal number of hours a nurse should work per week. The model tries to meet this, but may assign less if needed. MC and EL days reduce this cap.",
    key="preferred_weekly_hours",
)
# pref_weekly_hours_hard = st.sidebar.checkbox(
#     "Preferred weekly hours is a hard constraint",
#     value=False,
#     help="If checked, the preferred weekly hours is enforced strictly.",
#     key="pref_weekly_hours_hard",
# )

# if pref_weekly_hours_hard:
#     st.caption("Preferred weekly hours would be a hard minimum.")
#     min_acceptable_weekly_hours = preferred_weekly_hours  # Set to preferred weekly  hours if hard constraint is enabled

min_weekly_rest = st.sidebar.number_input(
    "Rest day eligible",
    min_value=1,
    value=MIN_WEEKLY_REST,
    help="The minimum number of days a nurse must rest per week.",
    key="min_weekly_rest",
)
# with st.sidebar.expander("AM Coverage Constraint", expanded=True):
#     activate_am_cov = st.checkbox(
#         "Activate AM Coverage",
#         value=False,
#         help="Activates the minimum AM coverage constraint.",
#         key="activate_am_cov",
#     )

#     am_coverage_min_percent = st.slider(
#         "AM coverage min percent",
#         min_value=34,
#         max_value=100,
#         value=AM_COVERAGE_MIN_PERCENT,
#         help="Aim for at least this % of nurses working AM shift.",
#         key="am_coverage_min_percent",
#         disabled=not activate_am_cov,
#     )

#     am_coverage_min_hard = st.checkbox(
#         "Minimum AM coverage is a hard constraint",
#         value=False,
#         help=(
#             "• If checked, the system strictly applies the AM nurse percentage.\n\n"
#             "• If unchecked, it lowers the target gradually using the step value, "
#             "but always ensures AM shifts are not outnumbered by PM or Night shifts."
#         ),
#         key="am_coverage_min_hard",
#         disabled=not activate_am_cov,
#     )

#     am_coverage_relax_step = st.number_input(
#         "AM relax step",
#         min_value=1,
#         max_value=66,
#         value=AM_COVERAGE_RELAX_STEP,
#         help="If minimum AM coverage is not met, gradually relax by this % of AM shifts.",
#         disabled=not activate_am_cov or am_coverage_min_hard,
#         key="am_coverage_relax_step",
#     )

#     if activate_am_cov and am_coverage_min_hard:
#         am_coverage_relax_step = 0
#         if am_coverage_min_hard:
#             st.caption("AM relax step will be ignored.")

# am_senior_min_percent = st.sidebar.slider(
#     "AM senior min percent",
#     min_value=50,
#     max_value=100,
#     value=AM_SENIOR_MIN_PERCENT,
#     help="Aim for at least this % of nurses in AM being seniors.",
#     key="am_senior_min_percent",
# )
# am_senior_min_hard = st.sidebar.checkbox(
#     "Minimum AM senior coverage is a hard constraint",
#     value=False,
#     help=(
#         "• If checked, the system strictly applies the senior percentage for AM shifts.\n\n"
#         "• If unchecked, it lowers the target gradually using the step value, but always ensures seniors are not outnumbered by juniors in AM shifts."
#     ),
#     key="am_senior_min_hard",
# )
# am_senior_relax_step = st.sidebar.number_input(
#     "AM senior relax step",
#     min_value=1,
#     max_value=50,
#     value=AM_SENIOR_RELAX_STEP,
#     help="If minimum AM senior coverage is not met, gradually relax by this % of senior nurses for AM shifts.",
#     disabled=am_senior_min_hard,
#     key="am_senior_relax_step",
# )
# if am_senior_min_hard:
#     st.caption("AM senior relax step will be ignored.")
#     am_senior_relax_step = 0

st.sidebar.subheader("Others")
weekend_rest = st.sidebar.checkbox(
    "Enforce alternating weekend rest",
    value=True,
    help="If checked, nurses who work on a weekend must rest the same day next weekend.",
    key="weekend_rest",
)
back_to_back_shift = st.sidebar.checkbox(
    "Allow back-to-back shifts",
    value=False,
    help="If checked, nurses may be scheduled for consecutive shifts (e.g., Night followed by AM).",
    key="back_to_back_shift",
)
allow_double_shift = st.sidebar.checkbox(
    "Allow double shifts",
    value=False,
    help="If checked, nurses may be scheduled for two consecutive shifts (e.g., AM followed by PM).",
    key="allow_double_shift",
)
# use_sliding_window = st.sidebar.checkbox(
#     "Use sliding window for weekly hours",
#     value=False,
#     help="If checked, the maximum weekly hours is enforced over any consecutive 7-day window, not just calendar weeks. This provides stricter control over nurse workload.",
#     key="use_sliding_window",
# )
shift_balance = st.sidebar.checkbox(
    "Balance shift assignments",
    value=False,
    help="If checked, the system will attempt to balance shift assignments among nurses. May cause longer solve times.",
    key="shift_balance",
)
if shift_balance:
    priority_setting = st.sidebar.select_slider(
        "Priority setting",
        options=[
            "Fairness",
            "Fairness-leaning",
            "50/50",
            "Preference-leaning",
            "Preference",
        ],
        value="50/50",
        key="priority_setting",
        help=(
            "Select priority for solving:\n\n"
            "• Fairness: Prioritise fairness of shift distributions.\n\n"
            "• Preference: Prioritise number of preferences met.\n\n"
        ),
    )

    st.sidebar.info(
        "⚠️ Your selected priority guides the solver, but results may vary depending on inputs and constraints.\n\n"
        "For example, 'Preference' may not always lead to more preferences met than '50/50'."
    )
else:
    priority_setting = "50/50"

# Validate input parameters
errors = validate_input_params(
    df_profiles,
    num_days,
    min_nurses_per_shift,
    min_seniors_per_shift,
    max_weekly_hours,
    preferred_weekly_hours,
    min_acceptable_weekly_hours,
)
if errors:
    st.error("\n".join(errors))
    st.stop()

# st.sidebar.markdown("---")
# st.sidebar.subheader("Add preferences manually")

# if "manual_prefs" not in st.session_state:
#     st.session_state.manual_prefs = []

# nurses = sorted(df_profiles["Name"].str.strip().str.upper().tolist())

# sel_nurse = st.sidebar.selectbox(
#     "Select Nurse for Preferences", options=nurses, index=None, key="sel_nurse"
# )

# sel_date = st.sidebar.date_input(
#     "Select Date for Preferences",
#     value=None,
#     min_value=start_date,
#     max_value=end_date,
#     key="sel_date",
# )

# sel_pref = st.sidebar.selectbox(
#     "Select Preference",
#     options=[
#         lbl for lbl in SHIFT_LABELS + NO_WORK_LABELS if lbl not in ["EL", "REST", "TR"]
#     ],
#     index=None,
#     key="sel_pref",
# )

# if st.sidebar.button("Add Preference"):
#     if not sel_nurse or not sel_date or not sel_pref:
#         st.sidebar.error("Please select a nurse, date, and preference.")
#         st.stop()

#     duplicates = any(
#         p["Nurse"] == sel_nurse and p["Date"] == pd.to_datetime(sel_date)
#         for p in st.session_state.manual_prefs
#     )

#     if duplicates:
#         st.sidebar.warning(f"{sel_nurse} already has a preference for {sel_date}.")
#     else:
#         now = datetime.now()
#         st.session_state.manual_prefs.append(
#             {
#                 "Nurse": sel_nurse,
#                 "Date": pd.to_datetime(sel_date),
#                 "Preference": sel_pref,
#                 "Timestamp": now,
#                 "Source": "Manual",
#             }
#         )
#         st.sidebar.success(
#             f"Added preference for {sel_nurse} on {sel_date}: {sel_pref} at {now:%H:%M:%S}."
#         )

# if st.session_state.manual_prefs:
#     st.sidebar.markdown("**Current Manual Preferences**")
#     for idx, pref in enumerate(st.session_state.manual_prefs, 1):
#         col1, col2, _ = st.sidebar.columns([7, 1, 1])
#         col1.markdown(
#             f"{idx}. {pref['Nurse']} on {pref['Date'].date()}: {pref['Preference']}"
#         )
#         if col2.button(label="❌", key=f"remove_{idx}", help="Remove this preference"):
#             st.session_state.manual_prefs.pop(idx - 1)
#             st.rerun()
# else:
#     st.sidebar.markdown("No manual preferences found.")

# Store core state
for key, default in {
    "fixed": {},
    "sched_df": None,
    "summary_df": None,
    "df_profiles": None,
    "df_prefs": None,
    "df_train_shifts": None,
    "df_prev_sched": None,
    "start_date": pd.to_datetime(start_date),
    "num_days": num_days,
    "shift_durations": shift_durations,
    "min_nurses_per_shift": min_nurses_per_shift,
    "min_seniors_per_shift": min_seniors_per_shift,
    "senior_staff_allocation": senior_staff_allocation,
    "senior_staff_percentage": senior_staff_percentage,
    "senior_staff_allocation_refinement": senior_staff_allocation_refinement,
    "senior_staff_allocation_refinement_value": senior_staff_allocation_refinement_value,
    "staff_allocation": StaffAllocations(
        seniorStaffAllocation=(senior_staff_allocation == "Yes"),
        seniorStaffPercentage=senior_staff_percentage,
        seniorStaffAllocationRefinement=(senior_staff_allocation_refinement == "Yes"),
        seniorStaffAllocationRefinementValue=senior_staff_allocation_refinement_value,
    ),
    "min_acceptable_weekly_hours": min_acceptable_weekly_hours,
    "max_weekly_hours": max_weekly_hours,
    "preferred_weekly_hours": preferred_weekly_hours,
    "min_weekly_rest": min_weekly_rest,
    "weekend_rest": weekend_rest,
    "back_to_back_shift": back_to_back_shift,
    "allow_double_shift": allow_double_shift,
    "shift_balance": shift_balance,
    "priority_setting": priority_setting,
    # "use_sliding_window": use_sliding_window,
    "pref_weekly_hours_hard": False,
    # "activate_am_cov": activate_am_cov,
    # "am_coverage_min_percent": am_coverage_min_percent,
    # "am_coverage_min_hard": am_coverage_min_hard,
    # "am_coverage_relax_step": am_coverage_relax_step,
    # "am_senior_min_percent": am_senior_min_percent,
    # "am_senior_min_hard": am_senior_min_hard,
    # "am_senior_relax_step": am_senior_relax_step,
    "all_el_overrides": {},
    "all_mc_overrides": {},
    "all_prefs_meta": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

logging.info(
    "Senior Staff Allocation: %-5s | Refinement Allowed: %-5s",
    senior_staff_allocation,
    senior_staff_allocation_refinement,
)

if "missing_prefs" not in st.session_state:
    st.session_state.missing_prefs = None
if "missing_train_shifts" not in st.session_state:
    st.session_state.missing_train_shifts = None

# Generate Schedule
if st.sidebar.button("Generate Schedule", type="primary"):
    st.session_state.fixed.clear()
    st.session_state.num_days = num_days
    st.session_state.all_el_overrides = {}
    st.session_state.all_mc_overrides = {}
    file_prefs_ts = datetime.now()
    st.session_state["file_prefs_ts"] = file_prefs_ts
    logging.info("Generate Schedule : %s", st.session_state)

    if profile_input_mode == "Upload File" and not profiles_file:
        st.error("Please upload a valid profiles excel file.")
        st.stop()

    try:
        # Load and validate shift preferences and training shifts data
        if prefs_file:
            df_prefs = load_shift_preferences(prefs_file)
            try:
                st.session_state.missing_prefs = validate_data(
                    df_profiles, df_prefs, "profiles", "preferences", False
                )
            except InputMismatchError as e:
                st.error(str(e))
                st.stop()
            logging.info(st.session_state.missing_prefs)

            for nurse in df_prefs.index:
                for col in df_prefs.columns:
                    v = df_prefs.at[nurse, col]
                    if pd.notna(v) and v != "":
                        df_prefs.at[nurse, col] = (
                            str(v).strip().upper(),
                            file_prefs_ts,
                        )

        else:
            df_prefs = pd.DataFrame(index=df_profiles["Name"])
            st.session_state.missing_prefs = None

        for p in st.session_state.get("manual_prefs", []):
            n = p["Nurse"].strip().upper()
            d = pd.to_datetime(p["Date"])
            pref = p["Preference"].strip().upper()

            if d not in df_prefs.columns:
                df_prefs[d] = ""
            df_prefs.at[n, d] = (pref, p["Timestamp"])

        # 1) build file‑prefs metadata
        file_ts = st.session_state["file_prefs_ts"]
        file_prefs = []
        for nurse in df_prefs.index:
            for col in df_prefs.columns:
                raw = df_prefs.at[nurse, col]
                if pd.notna(raw) and raw != "":
                    if isinstance(raw, tuple) and len(raw) == 2:
                        val, ts = raw
                    else:
                        val, ts = raw, file_prefs_ts

                    if ts != file_prefs_ts:
                        continue

                    val = str(val).strip().upper()
                    file_prefs.append(
                        {
                            "Nurse": nurse,
                            "Date": col,
                            "Preference": val,
                            "Timestamp": ts,
                            "Source": "File",
                        }
                    )

        # all_prefs_meta = st.session_state.get("file_prefs_meta", []) + st.session_state.manual_prefs
        # st.session_state["all_prefs_meta"] = all_prefs_meta

        # 2) merge with manual prefs and sort by Timestamp
        manual = st.session_state.get("manual_prefs", [])
        all_prefs_sorted = sorted(file_prefs + manual, key=lambda r: r["Timestamp"])

        st.sidebar.write("🔀 Preference application order (oldest → newest):")
        for i, rec in enumerate(all_prefs_sorted, 1):
            dt = rec["Date"].strftime("%Y-%m-%d")  # <-- added line
            ts = rec["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            st.sidebar.write(
                f"{i}. [{rec['Source']}] {rec['Nurse']} on {dt}: "
                f"{rec['Preference']} (at {ts})"
            )

        # 3) store the merged+sorted metadata if needed elsewhere
        st.session_state["all_prefs_meta"] = all_prefs_sorted

        # Training Shifts
        if training_shifts_file:
            df_train_shifts = load_training_shifts(training_shifts_file)
            try:
                st.session_state.missing_train_shifts = validate_data(
                    df_profiles, df_train_shifts, "profiles", "training shifts", False
                )
            except InputMismatchError as e:
                st.error(str(e))
                st.stop()
        else:
            df_train_shifts = pd.DataFrame(index=df_profiles["Name"])
            st.session_state.missing_train_shifts = None

        # Previous schedule
        if prev_sched_file:
            df_prev_sched = load_prev_schedule(prev_sched_file)
            try:
                validate_data(
                    df_profiles, df_prev_sched, "profiles", "previous schedule", False
                )
            except InputMismatchError as e:
                st.error(str(e))
                st.stop()
        else:
            df_prev_sched = pd.DataFrame(index=df_profiles["Name"])

        # Build initial schedule
        st.session_state.df_profiles = df_profiles
        st.session_state.df_prefs = df_prefs
        st.session_state.df_train_shifts = df_train_shifts
        st.session_state.df_prev_sched = df_prev_sched

        # print("profiles:\n", df_profiles)
        # print("Error1! Exiting.")
        # sys.exit()

        sched, summ, violations, metrics = build_schedule_model(
            df_profiles,
            df_prefs,
            df_train_shifts,
            df_prev_sched,
            pd.to_datetime(start_date),
            num_days,
            shift_durations,
            min_nurses_per_shift=st.session_state.min_nurses_per_shift,
            min_seniors_per_shift=st.session_state.min_seniors_per_shift,
            max_weekly_hours=st.session_state.max_weekly_hours,
            preferred_weekly_hours=st.session_state.preferred_weekly_hours,
            min_acceptable_weekly_hours=st.session_state.min_acceptable_weekly_hours,
            min_weekly_rest=st.session_state.min_weekly_rest,
            pref_weekly_hours_hard=st.session_state.pref_weekly_hours_hard,
            # activate_am_cov=st.session_state.activate_am_cov,
            # am_coverage_min_percent=st.session_state.am_coverage_min_percent,
            # am_coverage_min_hard=st.session_state.am_coverage_min_hard,
            # am_coverage_relax_step=st.session_state.am_coverage_relax_step,
            # am_senior_min_percent=st.session_state.am_senior_min_percent,
            # am_senior_min_hard=st.session_state.am_senior_min_hard,
            # am_senior_relax_step=st.session_state.am_senior_relax_step,
            # use_sliding_window=st.session_state.use_sliding_window,
            weekend_rest=st.session_state.weekend_rest,
            back_to_back_shift=st.session_state.back_to_back_shift,
            staff_allocation=StaffAllocations(
                seniorStaffAllocation=(
                    st.session_state.senior_staff_allocation == "No"
                ),
                seniorStaffPercentage=st.session_state.senior_staff_percentage,
                seniorStaffAllocationRefinement=(
                    st.session_state.senior_staff_allocation_refinement == "No"
                ),
                seniorStaffAllocationRefinementValue=st.session_state.senior_staff_allocation_refinement_value,
            ),
            shift_balance=st.session_state.shift_balance,
            priority_setting=st.session_state.priority_setting,
            fixed_assignments=st.session_state.fixed,
            allow_double_shift=st.session_state.allow_double_shift,
        )

        # print("schedule:\n", sched)
        # print("Error! Exiting.")
        # sys.exit()

        st.session_state.sched_df = sched
        st.session_state.summary_df = summ
        st.session_state.violations = violations
        st.session_state.metrics = metrics
        st.session_state.original_sched_df = sched.copy()

        st.session_state.show_schedule_expanded = False
        st.session_state["editable_toggle"] = "Hide"
        st.rerun()

    except CUSTOM_ERRORS as e:
        st.error(str(e))
        st.stop()
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
        key="editable_toggle",
        help="Show or hide the editable schedule to indicate MC and EL overrides.",
    )
    st.session_state.show_schedule_expanded = choice == "Show"

    if st.session_state.show_schedule_expanded:
        show_editable_schedule()

        # Regenerate schedule with all overrides
        if st.button("🔁 Regenerate with All Overrides"):
            el_overrides = st.session_state.get("all_el_overrides") or {}
            mc_overrides = st.session_state.get("all_mc_overrides") or {}
            pending_el = st.session_state.get("pending_el") or {}
            pending_mc = st.session_state.get("pending_mc") or {}

            # check mc override validity
            if pending_mc:

                # 1) collect every MC day: initial schedule, old overrides, new overrides
                orig = st.session_state.original_sched_df
                init = {
                    (n, i)
                    for n, r in orig.iterrows()
                    for i, col in enumerate(orig.columns)
                    if str(r[col]).strip().upper() == "MC"
                }
                all_mc = init | set(mc_overrides) | set(pending_mc)

                # 2) group days by nurse
                nd = defaultdict(set)
                for n, d in all_mc:
                    nd[n].add(d)

                # 3) validate
                errs = []
                for n, days in nd.items():
                    # weekly counts
                    for wk, cnt in Counter(d // DAYS_PER_WEEK for d in days).items():
                        if cnt > MAX_MC_DAYS_PER_WEEK:
                            errs.append(
                                f"{n}: {cnt} MCs in week {wk+1} (max {MAX_MC_DAYS_PER_WEEK})"
                            )
                    # consecutive runs
                    seq = sorted(days)
                    if any(seq[i + 2] - seq[i] < 3 for i in range(len(seq) - 2)):
                        errs.append(f"{n}: >2 consecutive MC days")

                if errs:
                    st.session_state.pending_mc.clear()
                    st.error("❌ Invalid MC override:\n" + "\n".join(errs))
                    st.info("Click 'hide' and 'show' to try again")
                    st.stop()

            try:
                if not pending_el and not pending_mc:
                    st.info("No overrides to apply.")

                else:
                    # merge pending into cumulative
                    st.session_state.all_el_overrides.update(
                        st.session_state.pending_el
                    )
                    st.session_state.all_mc_overrides.update(
                        st.session_state.pending_mc
                    )
                    el_overrides = st.session_state.all_el_overrides
                    mc_overrides = st.session_state.all_mc_overrides

                    # find earliest override day across pending overrides
                    all_days = [d for (_, d) in pending_el.keys()] + [
                        d for (_, d) in pending_mc.keys()
                    ]
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

                    # re-solve starting from earliest MC/EL day (included)
                    sched2, summ2, violations2, metrics2 = build_schedule_model(
                        st.session_state.df_profiles,
                        st.session_state.df_prefs,
                        st.session_state.df_train_shifts,
                        st.session_state.df_prev_sched,
                        st.session_state.start_date,
                        st.session_state.num_days,
                        st.session_state.shift_durations,
                        st.session_state.min_nurses_per_shift,
                        st.session_state.min_seniors_per_shift,
                        st.session_state.max_weekly_hours,
                        st.session_state.preferred_weekly_hours,
                        st.session_state.pref_weekly_hours_hard,
                        st.session_state.min_acceptable_weekly_hours,
                        st.session_state.min_weekly_rest,
                        # st.session_state.activate_am_cov,
                        # st.session_state.am_coverage_min_percent,
                        # st.session_state.am_coverage_min_hard,
                        # st.session_state.am_coverage_relax_step,
                        # st.session_state.am_senior_min_percent,
                        # st.session_state.am_senior_min_hard,
                        # st.session_state.am_senior_relax_step,
                        st.session_state.weekend_rest,
                        st.session_state.back_to_back_shift,
                        st.session_state.use_sliding_window,
                        st.session_state.shift_balance,
                        st.session_state.priority_setting,
                        st.session_state.fixed,
                        st.session_state.allow_double_shift,
                        st.session_state.staff_allocation,
                    )
                    st.session_state.sched_df = sched2
                    st.session_state.summary_df = summ2
                    st.session_state.violations = violations2
                    st.session_state.metrics = metrics2

                    st.success(f"Re-solved from day {w0} onward.")
                    st.session_state.show_schedule_expanded = False
                    st.rerun()
            except CUSTOM_ERRORS as e:
                st.error(str(e))
                st.stop()
            except Exception as e:
                tb = traceback.format_exc()
                st.error(f"Error: {e}")
                st.text_area("Traceback", tb, height=200)
                st.stop()

    if st.session_state.missing_prefs:
        fixed_assignments_exist = bool(st.session_state.fixed)

        if fixed_assignments_exist:
            fixed_nurses = {n for (n, _) in st.session_state.fixed.keys()}

            lines = st.session_state.missing_prefs.strip().splitlines()
            if len(lines) >= 3 and "•" in lines[1]:
                nurse_line = lines[1].strip()
                nurses = [
                    n.strip() for n in nurse_line[1:].split(",")
                ]  # remove • and split

                # Filter nurses not in fixed
                nurses_to_warn = [n for n in nurses if n not in fixed_nurses]

                if nurses_to_warn:
                    msg = [lines[0]]
                    msg.append(f"     • {', '.join(sorted(nurses_to_warn))}")
                    msg.append(lines[2])
                    warning_msg = "\n".join(msg)
                    st.warning(warning_msg)
                    logging.info(warning_msg)
                # else: skip warning (all fixed)
        else:
            # First generation, show full warning
            st.warning(st.session_state.missing_prefs)
            logging.info(st.session_state.missing_prefs)

    if st.session_state.missing_train_shifts:
        st.warning(st.session_state.missing_train_shifts)
        logging.info(st.session_state.missing_train_shifts)

    # Show schedule
    st.subheader("📅 Final Schedule")
    st.dataframe(st.session_state.sched_df, use_container_width=True)

    # Show summary
    st.subheader("📊 Final Summary Metrics")
    st.dataframe(st.session_state.summary_df, use_container_width=True)

    # Show relevent violations, if any
    violations = st.session_state.get("violations", {})
    if violations:
        st.subheader("⚠️ Violations Summary")
        st.caption(
            "These are soft constraint violations that the model tried to minimize."
        )

        for category, items in violations.items():
            count = (
                len(items)
                if hasattr(items, "__len__") and not isinstance(items, str)
                else items
            )
            st.markdown(f"🔸 **{category}**: {count} case{'s' if count != 1 else ''}")

            # List items if they exist
            if isinstance(items, list) and items:
                with st.expander(f"Details for {category}"):
                    for item in sorted(items):
                        st.markdown(f"- {item}")
            elif isinstance(items, str) and items != "N/A":
                st.markdown(f"- {items}")

    # Show relevent metrics, if any
    metrics = st.session_state.get("metrics", {})
    if metrics:
        st.subheader("📈 Metrics Summary")
        st.caption(
            "These are indicators of preferences satisfaction and fairness of the schedule."
        )
        for category, items in metrics.items():
            match category:
                case "Preference Met":
                    st.markdown(f"🔸 **{category}**: {items} preferences met")
                case "Preference Unmet":
                    total_unmet = sum(
                        s["Prefs_Unmet"]
                        for s in st.session_state.summary_df.to_dict(orient="records")
                    )
                    st.markdown(
                        f"🔸 **{category}**: {total_unmet} unmet preferences across {len(items)} nurse{'s' if len(items)!=1 else ''}"
                    )
                case "Fairness Gap":
                    if items == "N/A":
                        st.markdown(f"🔸 **{category}**: N/A")
                    else:
                        value = (
                            f"{items}%"
                            if isinstance(items, (int, float))
                            else str(items)
                        )
                        st.markdown(f"🔸 **{category}**: {value} gap")
                        st.caption(
                            "🛈 Shows the percentage difference of preferences met between the most and least satisfied nurse. Smaller gap = fairer distribution of meeting preferences."
                        )
            if isinstance(items, list) and items:
                with st.expander(f"Details for {category}"):
                    for item in sorted(items):
                        st.markdown(f"- {item}")
            elif isinstance(items, str) and items != "N/A":
                st.markdown(f"- {items}")

    st.markdown("---")
    st.subheader("📥 Download results")
    download_excel(st.session_state.sched_df, "nurse_schedule.xlsx")
    download_excel(st.session_state.summary_df, "nurse_summary.xlsx")

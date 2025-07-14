# 🩺 Roster Scheduler

**An intelligent nurse roster scheduler** powered by Google OR-Tools and Streamlit. It helps hospitals and planners generate optimized, fair, and constraint-aware schedules based on individual preferences and coverage requirements.

**Key features:**

-   **Full Schedule Generation**: Based on nurse profiles, preferences, and training shifts.
-   **Summary Dashboard**: Shows total preferences met and any soft constraint violations.
-   **Interactive Edit Mode**: Assign MC/EL shifts manually and then regenerate the roster.

---

## 🚀 Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/JKniaaa/Roster-Scheduler.git
cd roster-scheduler
```

or

```bash
git clone git@github.com:JKniaaa/Roster-Scheduler.git
cd roster-scheduler
```

---

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Linux/macOS
.venv\Scripts\activate         # On Windows
```

---

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Set the Python path for local imports

Create a .env file in the project root (same level as requirements.txt) with the following content:

```bash
PYTHONPATH=src
```

This sets the PYTHONPATH environment variable to the src/ directory, allowing Python to resolve internal imports properly.

---

### 5. Run the Application

#### Option 1: Using Invoke

Make sure `invoke` is installed:

```bash
pip install invoke
```

Then, run the app using:

```bash
invoke front
```

This runs `streamlit run ui.py` via the tasks.py file.

#### Option 2: Manually with Streamlit

```bash
streamlit run ui.py
```

---

## 📁 Project Structure

The project is structured as follows:

```text
roster-scheduler/
│
├── .venv/
├── .env
├── config/            → Constants and path configuration
├── data/              → Nurse input Excel files
├── jupyter/           → Development notebooks
├── legacy/            → Deprecated modules (e.g., old scheduler logic)
├── src/               → Main Python source code
│   ├── core/          → Scheduling state and constraint manager
│   ├── exceptions/    → Custom exception classes
│   ├── scheduler/     → Model setup, builder, runner, and solver logic
|       └── rules/     → Constraint rule functions
│   └── utils/         → Utility and helper functions
│
├── ui.py              → Streamlit web interface
├── tasks.py           → Invoke tasks
├── requirements.txt   → Project dependencies
└── README.md
```

## 📊 Input Files

The project expects the following input files:

-   `nurse_profiles.xlsx`: Contains nurse profiles, including `Names`, `Titles`, and `Years of Experience`.
-   `nurse_preferences.xlsx`: **(Optional)** Contains nurse preferences for each `date`, including shift preferences and leave requirements.
-   `training_shifts.xlsx`: **(Optional)** Contains training shifts for each nurse for each `date`.

Templates for each file can be found in the `data/` directory.

## 🙏 Acknowledgements

-   [Google OR-Tools](https://developers.google.com/optimization)
-   [Streamlit](https://streamlit.io/)
-   [Pandas](https://pandas.pydata.org/)

## 🏢 Internship Project

Developed by **Goh Jun Keat** under the internship program at **Encore Med Sdn Bhd**.  
Encore Med Sdn Bhd may incorporate this scheduler into its future products.  
© 2025 Encore Med Sdn Bhd. All rights reserved.

## 👤 Author

[**Goh Jun Keat**](https://github.com/JKniaaa) @ 2025

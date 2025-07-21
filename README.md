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

### 5. Build & Run the Application

#### Build the application

```bash
docker-compose up --build
```

#### Run the application (api)

```bash
docker-compose up api
```

#### Run the application (ui)

```bash
docker-compose up ui
```

This runs `streamlit run ui.py` via the tasks.py file.


## 📁 Project Structure

The project is structured as follows:

```text
roster-scheduler/
│
├── .venv/
├── .env
├── api/               → Api paths
├── config/            → Constants and path configuration
├── data/              → Nurse input Excel files
├── exceptions/        → Custom error exceptions
├── jupyter/           → Development notebooks
├── legacy/            → Deprecated modules (e.g., old scheduler logic)
├── models/            → Training models & logics
├── core/              → Scheduling state and constraint manager
├── exceptions/        → Custom exception classes
├── scheduler/         → Model setup, builder, runner, and solver logic
    └── rules/         → Constraint rule functions
├── utils/             → Utility and helper functions
├── main.py            → Application entry file
├── Dockerfile         → Dockerfile
├── docker-compose.yml → Docker composer file
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

## 👤 Author

[**Goh Jun Keat**](https://github.com/JKniaaa) @ 2025

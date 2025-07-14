""" Invoke tasks. """
import os
import sys
import io
from invoke.tasks import task
import subprocess

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

@task
def hello(c):
    print("hello")


@task
def install(c):
    c.run("pip install -r requirements.txt")


# @task
# def back(c):
#     c.run("uvicorn app:app --reload --host 127.0.0.1 --port 8001")


@task
def front(c):
    c.run("streamlit run ui.py", env={"PYTHONUTF8": "1"})


# @task
# def all(c):
#     # Launch backend
#     subprocess.Popen(
#         ["cmd", "/k", "uvicorn app:app --reload --host 127.0.0.1 --port 8001"],
#         creationflags=subprocess.CREATE_NEW_CONSOLE
#     )

#     # Launch frontend with PYTHONUTF8=1 set properly
#     env = os.environ.copy()
#     env["PYTHONUTF8"] = "1"
#     subprocess.Popen(
#         ["cmd", "/k", "streamlit run ui.py"],
#         creationflags=subprocess.CREATE_NEW_CONSOLE,
#         env=env
#     )


# @task
# def train(c):
#     c.run("python train_rl.py", env={"PYTHONUTF8": "1"})


# @task
# def train_logs(c):
#     c.run("tensorboard --logdir tb_logs")


@task
def clean(c):
    """
    Cross-platform clean task to remove all __pycache__ folders and .pyc files.
    """
    if os.name == 'nt':  # Windows
        # Remove all .pyc files
        c.run("for /R %f in (*.pyc) do del /F /Q \"%f\"", warn=True)
        # Remove all __pycache__ directories recursively
        c.run('for /d /r %d in (__pycache__) do @if exist "%d" rmdir /s /q "%d"', warn=True)
    else:  # Unix/Linux/macOS
        c.run("find . -type f -name '*.pyc' -delete", warn=True)
        c.run("find . -type d -name '__pycache__' -exec rm -r {} +", warn=True)
        
import os
import sys
import io
from invoke.tasks import task

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')

@task
def hello(c):
    print("hello")


@task
def install(c):
    c.run("pip install -r requirements.txt")


@task
def back(c):
    c.run("uvicorn app:app --reload --host 127.0.0.1 --port 8001")


@task
def front(c):
    c.run("streamlit run ui.py", env={"PYTHONUTF8": "1"})


@task
def all(c):
    back(c)
    front(c)


@task
def train(c):
    c.run("python train_rl.py", env={"PYTHONUTF8": "1"})


@task
def train_logs(c):
    c.run("tensorboard --logdir tb_logs")


@task
def clean(c):
    if os.name == 'nt':  # Windows
        c.run("del /s /q *.pyc", warn=True)
        c.run("rmdir /s /q __pycache__", warn=True)
    else:  # Unix/Linux/macOS
        c.run("rm -rf __pycache__ *.pyc", warn=True)
        
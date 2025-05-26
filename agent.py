# agent.py

import requests
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_openai import OpenAI
from langchain_community.llms import FakeListLLM
from pydantic import BaseModel

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY in .env file"

class ScheduleInput(BaseModel):
    profiles_path: str
    prefs_path: str
    start_date: str
    num_days: int
    rl_assignment: list


def call_schedule_api(
    profiles_path: str,
    prefs_path: str,
    start_date: str,
    num_days: int,
    rl_assignment: Optional[List[List[List[int]]]] = None
) -> Dict[str, Any]:
    payload = {
        "profiles_path": profiles_path,
        "prefs_path":    prefs_path,
        "start_date":    start_date,
        "num_days":      num_days,
        "rl_assignment": rl_assignment or []
    }
    resp = requests.post("http://127.0.0.1:8001/schedule", json=payload)
    resp.raise_for_status()
    return resp.json()


nurse_scheduler = Tool.from_function(
    name="nurse_scheduler",
    func=call_schedule_api,
    description=(
        "Use this tool to generate a nurse schedule. "
        "Requires 'profiles_path', 'prefs_path', 'start_date' (YYYY-MM-DD), 'num_days', and 'rl_assignment'."
    ),
    args_schema=ScheduleInput
)

tools = [nurse_scheduler]

# llm = OpenAI(temperature=0)
llm = FakeListLLM(responses=[
    """Thought: I should generate a schedule.
        Action: nurse_scheduler
        Action Input: {
            "profiles_path": "data/nurse_profiles.xlsx",
            "prefs_path": "data/nurse_preferences.xlsx",
            "start_date": "2025-05-26",
            "num_days": 14,
            "rl_assignment": []
        }"""
    ])

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

if __name__ == "__main__":
    prompt = (
        "Generate a 14-day nurse roster using 'data/nurse_profiles.xlsx' "
        "and 'data/nurse_preferences.xlsx', warm-start with RL if available, "
        "and return the schedule and total penalty."
    )

    result = agent.invoke({"input": prompt})
    print("\n=== Agent Response ===\n")
    print(result)

"""ADHD Task Initiation Coaching Evaluation Environment.

Evaluates ADHD coaching responses by scoring tool calling and response quality.
V2: Multiple scenarios, state tracking, rubric-based scoring.
"""

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import ADHDAction, ADHDObservation
from reward import score_rubric


# ADHD task initiation scenarios
ADHD_SCENARIOS = [
    "I can't start writing the email to my manager",
    "I've been staring at this blank document for 30 minutes",
    "I need to make a phone call but I keep putting it off",
    "I'm stuck on starting this presentation",
    "I've been avoiding this report all day",
    "I don't know how to begin this project proposal",
    "I keep switching tabs instead of starting my work",
    "I'm overwhelmed by this task list",
    "I can't focus on writing this code review",
    "I've been procrastinating on this assignment for hours",
]

# Non-ADHD scenarios: (prompt, expected_tool or None)
NON_ADHD_SCENARIOS = [
    ("What's the weather like today?", "web_search_tool"),
    ("What is the latest revenue for IBM?", "web_search_tool"),
    ("What is the capital of France?", "web_search_tool"),
    ("Write me a poem about cats", None),
    ("Translate this sentence to Spanish", None),
]


def generate_user_state() -> dict:
    """Generate randomized user state (the 'knobs')."""
    hour = random.randint(6, 22)
    minute = random.randint(0, 59)
    return {
        "time_of_day": f"{hour:02d}:{minute:02d}",
        "position_in_chair": random.choice(["normal", "slouching", "standing"]),
        "minutes_since_last_stood": random.randint(0, 240),
    }


class ADHDEnvironment(Environment):
    """ADHD Task Initiation Coaching Evaluation Environment.

    Evaluates coaching responses for ADHD task initiation paralysis.
    Innovation: state tracking + tool calling evaluation.

    V2: Multiple scenarios, state tracking, rubric-based scoring.
    - 10 ADHD scenarios + 5 non-ADHD scenarios
    - 3 state variables (time_of_day, position_in_chair, minutes_since_last_stood)
    - Rubric with tool calling + state awareness scoring

    Single-turn: reset() -> step() -> done=True
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_scenario: str = ""
        self.current_user_state: dict = {}
        self.is_adhd_scenario: bool = True
        self.expected_tool: Optional[str] = None

    def reset(self) -> ADHDObservation:
        """Generate new episode with randomized scenario and user state."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_user_state = generate_user_state()

        # Pick ADHD 80% / non-ADHD 20%
        if random.random() < 0.8:
            self.current_scenario = random.choice(ADHD_SCENARIOS)
            self.is_adhd_scenario = True
            self.expected_tool = "adhd_coach_tool"
        else:
            scenario_tuple = random.choice(NON_ADHD_SCENARIOS)
            self.current_scenario = scenario_tuple[0]
            self.is_adhd_scenario = False
            self.expected_tool = scenario_tuple[1]

        return ADHDObservation(
            scenario=self.current_scenario,
            state=self.current_user_state,
            done=False,
            reward=0.0,
            scoring={
                "version": "v2",
                "available_tools": [
                    "adhd_coach_tool",
                    "web_search_tool",
                ],
            },
        )

    def step(self, action: ADHDAction) -> ADHDObservation:  # type: ignore[override]
        """Score a coaching response.

        Single-turn: returns done=True after scoring.
        """
        self._state.step_count += 1

        scoring = score_rubric(
            action,
            self.current_scenario,
            self.current_user_state,
            self.is_adhd_scenario,
            self.expected_tool,
        )
        scoring["action"] = {
            "tool_calls": action.tool_calls,
            "message": action.message,
        }

        return ADHDObservation(
            scenario=self.current_scenario,
            state=self.current_user_state,
            done=True,
            reward=scoring["total_score"],
            scoring=scoring,
        )

    @property
    def state(self) -> State:
        return self._state

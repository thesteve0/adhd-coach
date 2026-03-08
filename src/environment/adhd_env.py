"""ADHD Task Initiation Coaching Evaluation Environment.

Minimal V1: Single hardcoded scenario, tool calling evaluation only.
State tracking and scenario variety will be added in the afternoon.
"""

import asyncio
from openenv.core.client_types import StepResult
from typing import Optional

from .models import ADHDAction, ADHDObservation
from .reward import create_rubric_v1, explain_score_v1


class ADHDEnvironment:
    """ADHD Task Initiation Coaching Evaluation Environment

    V1: Minimal implementation
    - Single hardcoded scenario
    - Tool calling evaluation only
    - No state tracking yet

    Single-turn: reset() → step() → done=True
    """

    # Hardcoded scenario for V1
    DEFAULT_SCENARIO = "I can't start writing the email to my manager"

    def __init__(self, seed: Optional[int] = None):
        """Initialize environment

        Args:
            seed: Random seed (not used in V1, will use for state generation later)
        """

        self.seed = seed
        self.rubric = create_rubric_v1()

        # Current episode state
        self.current_scenario: Optional[str] = None

    def reset(self) -> StepResult:
        """Generate new episode

        V1: Returns hardcoded scenario with empty state.

        Returns:
            StepResult with observation
        """

        # Use hardcoded scenario for V1
        self.current_scenario = self.DEFAULT_SCENARIO

        # Create observation (no state yet)
        observation = ADHDObservation(
            scenario=self.current_scenario,
            state={},  # Empty for V1, will add state tracking later
            done=False,
            reward=0.0,
            metadata={
                "version": "v1",
                "note": "State tracking will be added in afternoon iteration",
            }
        )

        return StepResult(
            observation=observation,
            reward=0.0,
            done=False
        )

    async def step(self, action: ADHDAction) -> StepResult:
        """Score the coaching response (single-turn)

        Args:
            action: ADHDAction with tool_calls and message

        Returns:
            StepResult with score and explanation

        Raises:
            ValueError: If reset() hasn't been called
        """

        if self.current_scenario is None:
            raise ValueError("Must call reset() before step()")

        # Score using V1 rubric (async)
        state = {}  # Empty state for V1
        score = await self.rubric(action, state, self.current_scenario)

        # Generate explanation
        metadata = explain_score_v1(action, score)

        # Add action details to metadata
        metadata["action"] = {
            "tool_calls": action.tool_calls,
            "message": action.message,
        }

        # Create observation
        observation = ADHDObservation(
            scenario=self.current_scenario,
            state=state,
            done=True,  # Single-turn
            reward=score,
            metadata=metadata
        )

        return StepResult(
            observation=observation,
            reward=score,
            done=True
        )

    def get_info(self) -> dict:
        """Return environment metadata

        Returns:
            Environment info dict
        """

        return {
            "environment": "ADHD Task Initiation Coaching",
            "version": "0.1.0-v1",
            "stage": "minimal",
            "features": {
                "tool_calling_evaluation": True,
                "state_tracking": False,  # Will add in afternoon
                "scenario_variety": False,  # Will add in afternoon
                "multi_criteria_rubric": False,  # Will add in evening
            },
            "available_tools": [
                "adhd_task_initiation_coach",
                "set_timer",
                "break_down_task",
            ],
            "scoring_criteria": ["tool_calling"],
        }

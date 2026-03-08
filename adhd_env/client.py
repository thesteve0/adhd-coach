"""ADHD Environment Client.

Connects to an ADHD coaching evaluation environment server via WebSocket.
"""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import ADHDAction, ADHDObservation


class ADHDEnv(EnvClient[ADHDAction, ADHDObservation]):
    """Client for the ADHD Task Initiation Coaching Environment.

    Example:
        >>> with ADHDEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.scenario)
        ...
        ...     result = client.step(ADHDAction(
        ...         tool_calls=["adhd_task_initiation_coach"],
        ...         message="Open email and type just the recipient name."
        ...     ))
        ...     print(f"Reward: {result.reward}")
    """

    def _step_payload(self, action: ADHDAction) -> Dict:
        """Convert ADHDAction to JSON payload."""
        return {
            "tool_calls": action.tool_calls,
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ADHDObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = ADHDObservation(
            scenario=obs_data.get("scenario", ""),
            state=obs_data.get("state", {}),
            scoring=obs_data.get("scoring", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

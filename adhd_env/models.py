"""Data models for the ADHD Task Initiation Coaching Environment."""

from pydantic import Field
from typing import List, Dict, Any

from openenv.core.env_server.types import Action, Observation


class ADHDAction(Action):
    """Action: Tool calls + coaching response to evaluate.

    Models submit tool_calls (which tools they'd invoke) and a message
    (the coaching response text) for scoring.
    """

    tool_calls: List[str] = Field(
        default_factory=list,
        description="Tools called by the model (e.g., ['adhd_task_initiation_coach'])",
    )
    message: str = Field(
        default="",
        description="The coaching response text",
    )


class ADHDObservation(Observation):
    """Observation: ADHD scenario + user state.

    Returned from reset() with the scenario and state.
    Returned from step() with the scored reward and scoring details.
    Note: done, reward, metadata are inherited from Observation base class.
    Note: OpenEnv's serialize_observation excludes 'metadata' from HTTP responses,
    so we use a custom 'scoring' field for scoring details.
    """

    scenario: str = Field(
        default="",
        description="The task initiation scenario (user utterance)",
    )
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description="User state tracking (sitting time, energy, etc.)",
    )
    scoring: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scoring breakdown and explanation (visible in HTTP responses)",
    )

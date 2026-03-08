"""Data models for ADHD coaching environment.

Minimal version: ADHDAction and ADHDObservation only.
State tracking will be added in the afternoon.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ADHDAction(BaseModel):
    """Action: Tool calls + coaching response to evaluate"""

    tool_calls: List[str] = Field(
        default_factory=list,
        description="Tools called by the model (e.g., ['adhd_task_initiation_coach'])"
    )
    message: str = Field(
        description="The coaching response text"
    )


class ADHDObservation(BaseModel):
    """Observation returned to client"""

    scenario: str = Field(
        description="The task initiation scenario (user utterance)"
    )
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description="User state (empty for now, will add state tracking later)"
    )
    done: bool = Field(
        default=False,
        description="Episode termination flag"
    )
    reward: float = Field(
        default=0.0,
        description="Reward score (0.0-1.0)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scoring explanation and auxiliary data"
    )

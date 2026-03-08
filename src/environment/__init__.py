"""ADHD Task Initiation Coaching Evaluation Environment

OpenEnv environment that evaluates ADHD coaching quality through:
- Tool calling evaluation (V1)
- State tracking (coming in afternoon)
- Composable rubric scoring (coming in evening)
"""

from .models import ADHDAction, ADHDObservation
from .adhd_env import ADHDEnvironment

__version__ = "0.1.0-v1"

__all__ = [
    "ADHDEnvironment",
    "ADHDAction",
    "ADHDObservation",
]

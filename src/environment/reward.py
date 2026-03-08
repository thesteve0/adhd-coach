"""Reward function using verifiers Rubric framework.

V1: Tool calling evaluation only (100% weight).
Will add more criteria later based on manual testing.
"""

from typing import Dict, Any
from .models import ADHDAction


# ===== TOOL VALIDATION =====

VALID_TOOLS = [
    "adhd_task_initiation_coach",
    "set_timer",
    "break_down_task",
]


def validate_tool_calls(tool_calls: list[str]) -> Dict[str, Any]:
    """Validate tool calls and return metadata"""

    return {
        "all_valid": all(tool in VALID_TOOLS for tool in tool_calls),
        "valid_tools": [tool for tool in tool_calls if tool in VALID_TOOLS],
        "invalid_tools": [tool for tool in tool_calls if tool not in VALID_TOOLS],
        "called_count": len(tool_calls),
        "called_primary_tool": "adhd_task_initiation_coach" in tool_calls,
    }


# ===== SCORING CRITERION =====

async def score_tool_calling(action: ADHDAction, state: Dict[str, Any], scenario: str) -> float:
    """V1 Criterion: Did the model call the appropriate tool?

    This is the PRIMARY innovation - evaluating tool calling, not just text.

    Args:
        action: The action taken (tool_calls + message)
        state: User state (currently empty, will add later)
        scenario: The scenario text

    Returns:
        1.0 - Called adhd_task_initiation_coach
        0.5 - Called a tool, but not the primary one
        0.0 - No tools called
    """

    if not action.tool_calls:
        return 0.0

    validation = validate_tool_calls(action.tool_calls)

    # Best: called the primary tool
    if validation["called_primary_tool"]:
        return 1.0

    # Acceptable: called a valid tool (just not the primary one)
    if validation["all_valid"] and validation["called_count"] > 0:
        return 0.5

    # Bad: called tools but they were invalid
    return 0.0


# ===== RUBRIC FACTORY =====

def create_rubric_v1():
    """Create V1 rubric: Tool calling only

    Note: We're NOT using verifiers.Rubric for V1 to keep it simple.
    We'll integrate verifiers.Rubric in V2 after we validate the approach.
    """

    async def score(action: ADHDAction, state: Dict[str, Any], scenario: str) -> float:
        """Score an action using V1 criterion"""
        return await score_tool_calling(action, state, scenario)

    return score


def explain_score_v1(action: ADHDAction, score: float) -> Dict[str, Any]:
    """Generate explanation for V1 scoring

    Args:
        action: The action that was scored
        score: The computed score

    Returns:
        Metadata dict explaining the score
    """

    validation = validate_tool_calls(action.tool_calls)

    return {
        "version": "v1",
        "total_score": score,
        "criteria": {
            "tool_calling": {
                "score": score,
                "weight": 1.0,
                "tools_called": action.tool_calls,
                "validation": validation,
                "explanation": (
                    "Called primary tool (adhd_task_initiation_coach)" if validation["called_primary_tool"]
                    else "No tools called" if not action.tool_calls
                    else f"Called tool but not primary: {action.tool_calls}"
                ),
            }
        },
    }

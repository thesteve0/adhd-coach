"""Reward scoring for ADHD coaching environment.

V2: Rubric-based scoring with tool calling + state awareness.
- Tool calling: 40% weight - penalizes wrong-domain tools
- State awareness: 30% weight - rewards state-responsive coaching
- ADHD relevance: 30% weight - rewards directive, low-cognitive-load responses
"""

from typing import Dict, Any, Optional
from models import ADHDAction


# ADHD-domain tools
ADHD_TOOLS = {"adhd_coach_tool"}


def score_tool_calling(
    action: ADHDAction,
    is_adhd_scenario: bool,
    expected_tool: Optional[str] = None,
) -> float:
    """Score tool selection based on scenario type.

    ADHD scenario:
        1.0  - called adhd_coach_tool
        0.0  - no tools called
       -0.5  - called a non-ADHD tool (wrong domain)

    Non-ADHD scenario:
       -0.5  - called adhd_coach_tool (wrong domain)
        0.7  - called the expected non-ADHD tool
        0.5  - no tools called (neutral)
        0.5  - called some other non-ADHD tool (neutral)
    """
    called = set(action.tool_calls)

    if is_adhd_scenario:
        if "adhd_coach_tool" in called:
            return 1.0
        if not called:
            return 0.0
        # Called non-ADHD tool on ADHD scenario
        return -0.5
    else:
        # Non-ADHD scenario
        if "adhd_coach_tool" in called:
            return -0.5
        if expected_tool and expected_tool in called:
            return 0.7
        # No tool or some other non-ADHD tool - neutral
        return 0.5


def score_state_awareness(action: ADHDAction, user_state: dict) -> float:
    """Score whether response accounts for user state.

    1.0 - mentions movement/stretching when sitting 60+ min or slouching
    1.0 - suggests simpler tasks when evening (hour >= 20)
    0.5 - generic response (default, neutral)
    """
    msg = action.message.lower()
    score = 0.5  # default neutral

    minutes_sitting = user_state.get("minutes_since_last_stood", 0)
    position = user_state.get("position_in_chair", "normal")
    time_str = user_state.get("time_of_day", "12:00")
    hour = int(time_str.split(":")[0])

    movement_keywords = [
        "stand", "stretch", "walk", "move", "get up", "posture",
        "take a break", "step away", "physical",
    ]

    # Reward movement suggestions when sitting too long or slouching
    if minutes_sitting >= 60 or position == "slouching":
        if any(kw in msg for kw in movement_keywords):
            score = 1.0

    # Reward simpler task suggestions in the evening
    evening_keywords = [
        "simple", "small", "easy", "quick", "short", "wind down",
        "rest", "tomorrow", "lighter",
    ]
    if hour >= 20:
        if any(kw in msg for kw in evening_keywords):
            score = 1.0

    return score


def score_adhd_relevance(action: ADHDAction, is_adhd_scenario: bool) -> float:
    """Score ADHD-specific response quality.

    For ADHD scenarios: rewards concise responses and reflective questions.
    For non-ADHD: returns neutral 0.5.
    """
    if not is_adhd_scenario:
        return 0.5

    msg = action.message.strip()
    if not msg:
        return 0.0

    score = 0.5  # baseline
    msg_lower = msg.lower()

    # Reward reflective/clarifying questions that prompt self-reflection
    if "?" in msg:
        question_words = ("what", "how")
        reflective_words = ("specific", "detail", "details", "feeling", "think", "reflect", "explain")
        if any(qw in msg_lower for qw in question_words) and any(rw in msg_lower for rw in reflective_words):
            score += 0.15

    # Reward concise responses (under 100 words = lower cognitive load)
    word_count = len(msg.split())
    if 5 <= word_count <= 50:
        score += 0.25
    elif word_count > 100:
        score -= 0.25  # too long, high cognitive load

    return max(0.0, min(1.0, score))


def score_rubric(
    action: ADHDAction,
    scenario: str,
    user_state: dict,
    is_adhd_scenario: bool,
    expected_tool: Optional[str] = None,
) -> Dict[str, Any]:
    """Combined rubric score with per-criterion breakdown.

    Weights: tool_calling 40% + state_awareness 30% + adhd_relevance 30%
    Total clamped to 0.0-1.0.
    """
    tool_score = score_tool_calling(action, is_adhd_scenario, expected_tool)
    state_score = score_state_awareness(action, user_state)
    relevance_score = score_adhd_relevance(action, is_adhd_scenario)

    raw_total = (tool_score * 0.4) + (state_score * 0.3) + (relevance_score * 0.3)
    total = max(0.0, min(1.0, raw_total))

    return {
        "version": "v2",
        "total_score": round(total, 3),
        "criteria": {
            "tool_calling": {
                "score": tool_score,
                "weight": 0.4,
                "is_adhd_scenario": is_adhd_scenario,
                "expected_tool": expected_tool,
                "tools_called": action.tool_calls,
            },
            "state_awareness": {
                "score": state_score,
                "weight": 0.3,
                "user_state": user_state,
            },
            "adhd_relevance": {
                "score": relevance_score,
                "weight": 0.3,
            },
        },
    }

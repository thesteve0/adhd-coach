"""Manual environment testing with hardcoded interactions.

This is the PRIMARY TEST per judge feedback:
"Stage 1 is manual interaction. See if the environment exposes
the right 'knobs' and gives reasonable scores."
"""

import asyncio
from src.environment import ADHDEnvironment, ADHDAction


async def test_manual_interactions():
    """Test environment with hardcoded good/bad/medium responses"""

    env = ADHDEnvironment(seed=42)

    print("=" * 80)
    print("ADHD COACHING ENVIRONMENT - MANUAL TESTING (V1)")
    print("=" * 80)

    # Get environment info
    info = env.get_info()
    print("\n Environment Info:")
    print(f"  Version: {info['version']}")
    print(f"  Stage: {info['stage']}")
    print(f"  Features: {info['features']}")
    print(f"  Scoring Criteria: {info['scoring_criteria']}")

    # ===== Test 1: Good Response (with tool) =====
    print("\n" + "=" * 80)
    print("TEST 1: Good Response (with primary tool)")
    print("=" * 80)

    result = env.reset()
    print(f"\nScenario: {result.observation.scenario}")
    print(f"State: {result.observation.state}")

    action_good = ADHDAction(
        tool_calls=["adhd_task_initiation_coach"],
        message="Open your email draft and type just the subject line. That's it."
    )

    print(f"\nAction:")
    print(f"  Tool calls: {action_good.tool_calls}")
    print(f"  Message: {action_good.message}")

    step_result = await env.step(action_good)
    print(f"\nResult:")
    print(f"  Reward: {step_result.reward:.2f}")
    print(f"  Done: {step_result.done}")
    print(f"  Explanation: {step_result.observation.metadata['criteria']['tool_calling']['explanation']}")

    # ===== Test 2: Bad Response (no tool) =====
    print("\n" + "=" * 80)
    print("TEST 2: Bad Response (no tool)")
    print("=" * 80)

    result = env.reset()
    print(f"\nScenario: {result.observation.scenario}")

    action_bad = ADHDAction(
        tool_calls=[],  # No tool
        message="What do you want to work on? How can I help you?"
    )

    print(f"\nAction:")
    print(f"  Tool calls: {action_bad.tool_calls}")
    print(f"  Message: {action_bad.message}")

    step_result = await env.step(action_bad)
    print(f"\nResult:")
    print(f"  Reward: {step_result.reward:.2f}")
    print(f"  Done: {step_result.done}")
    print(f"  Explanation: {step_result.observation.metadata['criteria']['tool_calling']['explanation']}")

    # ===== Test 3: Medium Response (wrong tool) =====
    print("\n" + "=" * 80)
    print("TEST 3: Medium Response (secondary tool instead of primary)")
    print("=" * 80)

    result = env.reset()
    print(f"\nScenario: {result.observation.scenario}")

    action_medium = ADHDAction(
        tool_calls=["set_timer"],  # Valid tool, but not primary
        message="Let's set a 5-minute timer to work on this."
    )

    print(f"\nAction:")
    print(f"  Tool calls: {action_medium.tool_calls}")
    print(f"  Message: {action_medium.message}")

    step_result = await env.step(action_medium)
    print(f"\nResult:")
    print(f"  Reward: {step_result.reward:.2f}")
    print(f"  Done: {step_result.done}")
    print(f"  Explanation: {step_result.observation.metadata['criteria']['tool_calling']['explanation']}")

    # ===== Verification =====
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    print("\nExpected behavior:")
    print("  ✓ Good action (with primary tool) should score 1.0")
    print("  ✓ Bad action (no tool) should score 0.0")
    print("  ✓ Medium action (secondary tool) should score 0.5")

    print("\n" + "=" * 80)
    print("MANUAL TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_manual_interactions())

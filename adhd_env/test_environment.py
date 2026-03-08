#!/usr/bin/env python3
"""Test script for the ADHD coaching environment.

Tests the environment directly (no server needed) and via HTTP if a server is running.

Usage:
    # Direct test (no server):
    cd adhd_env && .venv/bin/python test_environment.py

    # With server running:
    cd adhd_env && .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000 &
    cd adhd_env && .venv/bin/python test_environment.py --http
"""

import sys


def test_direct():
    """Test environment directly without HTTP server."""
    from server.adhd_env_environment import ADHDEnvironment
    from models import ADHDAction

    env = ADHDEnvironment()
    print("=" * 60)
    print("DIRECT ENVIRONMENT TEST")
    print("=" * 60)

    # Test reset returns valid state
    obs = env.reset()
    print(f"\n--- Reset ---")
    print(f"Scenario: {obs.scenario}")
    print(f"State: {obs.state}")
    print(f"Done: {obs.done}")
    print(f"Reward: {obs.reward}")

    assert obs.scenario, "Scenario should not be empty"
    assert obs.done is False
    assert obs.reward == 0.0

    # Validate state has all 3 keys
    assert "time_of_day" in obs.state, "Missing time_of_day"
    assert "position_in_chair" in obs.state, "Missing position_in_chair"
    assert "minutes_since_last_stood" in obs.state, "Missing minutes_since_last_stood"
    assert obs.state["position_in_chair"] in ("normal", "slouching", "standing")
    assert 0 <= obs.state["minutes_since_last_stood"] <= 240
    print("State validation: PASS")

    # Variety check: reset 10x and verify we get at least 2 distinct states
    states = []
    for _ in range(10):
        o = env.reset()
        states.append(
            (o.state["time_of_day"], o.state["position_in_chair"], o.state["minutes_since_last_stood"])
        )
    unique_states = len(set(states))
    assert unique_states >= 2, f"Expected at least 2 distinct states, got {unique_states}"
    print(f"State variety check ({unique_states} unique in 10 resets): PASS")

    print(f"\n{'=' * 60}")
    print("ALL DIRECT TESTS PASSED")
    print(f"{'=' * 60}")


def test_rubric():
    """Test rubric scoring with positive and negative cases."""
    from server.adhd_env_environment import ADHDEnvironment
    from models import ADHDAction
    from reward import score_rubric

    print(f"\n{'=' * 60}")
    print("RUBRIC TEST")
    print(f"{'=' * 60}")

    # State where user has been sitting a long time and is slouching
    tired_state = {
        "time_of_day": "14:00",
        "position_in_chair": "slouching",
        "minutes_since_last_stood": 90,
    }

    evening_state = {
        "time_of_day": "21:00",
        "position_in_chair": "normal",
        "minutes_since_last_stood": 30,
    }

    # POSITIVE: ADHD scenario + primary tool + state-aware message
    action_good = ADHDAction(
        tool_calls=["adhd_coach_tool"],
        message="Stand up and stretch for 30 seconds, then type just the recipient name.",
    )
    result = score_rubric(action_good, "I can't start the email", tired_state, True, None)
    print(f"\nPOSITIVE (ADHD + primary tool + state-aware): {result['total_score']}")
    assert result["total_score"] >= 0.7, f"Expected >= 0.7, got {result['total_score']}"
    print("PASS")

    # NEGATIVE: ADHD scenario + wrong-domain tool
    action_wrong_tool = ADHDAction(
        tool_calls=["web_search_tool"],
        message="Let me search for tips on email writing.",
    )
    result = score_rubric(action_wrong_tool, "I can't start the email", tired_state, True, None)
    print(f"\nNEGATIVE (ADHD + web_search_tool): {result['total_score']}")
    assert result["total_score"] < 0.3, f"Expected < 0.3, got {result['total_score']}"
    print("PASS")

    # NEGATIVE: Non-ADHD scenario + ADHD tool
    action_adhd_on_non = ADHDAction(
        tool_calls=["adhd_coach_tool"],
        message="Let me help you initiate that task.",
    )
    result = score_rubric(action_adhd_on_non, "What's the weather?", tired_state, False, "web_search_tool")
    print(f"\nNEGATIVE (non-ADHD + ADHD tool): {result['total_score']}")
    assert result["total_score"] < 0.3, f"Expected < 0.3, got {result['total_score']}"
    print("PASS")

    # SLIGHTLY POSITIVE: Non-ADHD factual + correct tool
    action_correct_non_adhd = ADHDAction(
        tool_calls=["web_search_tool"],
        message="Let me look that up for you.",
    )
    result = score_rubric(action_correct_non_adhd, "What is the capital of France?", tired_state, False, "web_search_tool")
    print(f"\nSLIGHTLY POSITIVE (non-ADHD + correct tool): {result['total_score']}")
    assert result["total_score"] >= 0.5, f"Expected >= 0.5, got {result['total_score']}"
    print("PASS")

    # NEUTRAL: Non-ADHD creative + no tool
    action_no_tool_creative = ADHDAction(
        tool_calls=[],
        message="Here is a poem about cats.",
    )
    result = score_rubric(action_no_tool_creative, "Write me a poem about cats", tired_state, False, None)
    print(f"\nNEUTRAL (non-ADHD creative + no tool): {result['total_score']}")
    assert 0.3 <= result["total_score"] <= 0.7, f"Expected 0.3-0.7, got {result['total_score']}"
    print("PASS")

    # MEDIUM: ADHD + primary tool + generic message (no state awareness)
    action_generic = ADHDAction(
        tool_calls=["adhd_coach_tool"],
        message="Try breaking this task into smaller pieces.",
    )
    result = score_rubric(action_generic, "I'm stuck on this report", tired_state, True, None)
    print(f"\nMEDIUM (ADHD + primary tool + generic): {result['total_score']}")
    assert 0.4 <= result["total_score"] <= 0.85, f"Expected 0.4-0.85, got {result['total_score']}"
    print("PASS")

    # EVENING: ADHD + primary tool + evening-aware message
    action_evening = ADHDAction(
        tool_calls=["adhd_coach_tool"],
        message="It's late. Pick a small easy task to finish tonight, save the rest for tomorrow.",
    )
    result = score_rubric(action_evening, "I can't focus on this", evening_state, True, None)
    print(f"\nEVENING AWARE (ADHD + primary tool + evening tips): {result['total_score']}")
    assert result["total_score"] >= 0.7, f"Expected >= 0.7, got {result['total_score']}"
    print("PASS")

    # REFLECTIVE QUESTION: ADHD + primary tool + clarifying question
    action_reflective = ADHDAction(
        tool_calls=["adhd_coach_tool"],
        message="What are you specifically stuck on? Explain the first step you think you need to take.",
    )
    result_reflective = score_rubric(action_reflective, "I've been stuck for 30 minutes", tired_state, True, None)
    # Compare against same scenario with generic non-reflective message
    action_plain = ADHDAction(
        tool_calls=["adhd_coach_tool"],
        message="Just try to get started on it.",
    )
    result_plain = score_rubric(action_plain, "I've been stuck for 30 minutes", tired_state, True, None)
    print(f"\nREFLECTIVE Q (ADHD + primary tool + clarifying question): {result_reflective['total_score']}")
    print(f"  vs PLAIN (ADHD + primary tool + generic): {result_plain['total_score']}")
    assert result_reflective["total_score"] > result_plain["total_score"], \
        f"Reflective question should score higher than plain: {result_reflective['total_score']} vs {result_plain['total_score']}"
    print("PASS")

    print(f"\n{'=' * 60}")
    print("ALL RUBRIC TESTS PASSED")
    print(f"{'=' * 60}")


def test_http(base_url="http://localhost:8000"):
    """Test environment via HTTP endpoints."""
    import requests

    print(f"\n{'=' * 60}")
    print(f"HTTP TEST ({base_url})")
    print(f"{'=' * 60}")

    # Health check
    r = requests.get(f"{base_url}/health")
    assert r.status_code == 200
    print(f"\nHealth: {r.json()}")

    # Schema
    r = requests.get(f"{base_url}/schema")
    assert r.status_code == 200
    schema = r.json()
    assert "action" in schema
    assert "observation" in schema
    print(f"Schema: action has {list(schema['action']['properties'].keys())}")
    print(f"Schema: observation has {list(schema['observation']['properties'].keys())}")

    # Reset
    r = requests.post(f"{base_url}/reset")
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is False
    assert data["reward"] == 0.0
    assert "scenario" in data["observation"]
    obs = data["observation"]
    assert "state" in obs
    assert "time_of_day" in obs["state"]
    assert "position_in_chair" in obs["state"]
    assert "minutes_since_last_stood" in obs["state"]
    print(f"\nReset: scenario='{obs['scenario']}'")
    print(f"  state={obs['state']}")
    print(f"  State keys present: PASS")

    # Good action (ADHD scenario + primary tool)
    r = requests.post(f"{base_url}/step", json={
        "action": {
            "tool_calls": ["adhd_coach_tool"],
            "message": "Stand up and stretch, then type just the recipient name.",
        }
    })
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is True
    assert data["reward"] > 0
    print(f"Good action: reward={data['reward']} PASS")

    # Bad action (no tools on presumed ADHD scenario)
    r = requests.post(f"{base_url}/step", json={
        "action": {
            "tool_calls": [],
            "message": "What do you want to work on?",
        }
    })
    assert r.status_code == 200
    data = r.json()
    print(f"No-tool action: reward={data['reward']}")

    # Verify scoring details in response
    assert "scoring" in data["observation"]
    assert "total_score" in data["observation"]["scoring"]
    assert "criteria" in data["observation"]["scoring"]
    print(f"Scoring details present: PASS")

    print(f"\n{'=' * 60}")
    print("ALL HTTP TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_direct()
    test_rubric()

    if "--http" in sys.argv:
        url = "http://localhost:8000"
        for arg in sys.argv:
            if arg.startswith("http"):
                url = arg
        test_http(url)

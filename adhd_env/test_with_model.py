#!/usr/bin/env python3
"""End-to-end test: LLM with tool calling -> ADHD environment scoring.

Tests whether LLMs pick the correct tools for ADHD vs non-ADHD scenarios,
and scores their responses using the environment's rubric.

Usage:
    cd adhd_env && .venv/bin/python test_with_model.py
    cd adhd_env && .venv/bin/python test_with_model.py --model Qwen/Qwen3.5-9B
    cd adhd_env && .venv/bin/python test_with_model.py --all

Requires HF_TOKEN environment variable.
"""

import argparse
import os
import sys

from openai import OpenAI

from models import ADHDAction
from reward import score_rubric

MODELS = [
    "HuggingFaceTB/SmolLM3-3B",
    "Qwen/Qwen3.5-9B",
    "allenai/OLMo-3-7B-Instruct",
]

# Tool definitions the LLM sees
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "adhd_assist_tool",
            "description": (
                "Help a user with ADHD task initiation paralysis. "
                "Use when someone is stuck starting a task, procrastinating, "
                "or overwhelmed by executive function challenges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coaching_message": {
                        "type": "string",
                        "description": "The coaching response to help the user start their task.",
                    }
                },
                "required": ["coaching_message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": (
                "Search the web for information. Use for general knowledge questions, "
                "weather, facts, latest news, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# LLM tool name -> environment tool name
TOOL_NAME_MAP = {
    "adhd_assist_tool": "adhd_coach_tool",
    "web_search_tool": "web_search_tool",
}

# Test cases: (scenario, user_state, is_adhd, expected_tool, expected_llm_tool, description)
TEST_CASES = [
    {
        "scenario": "I can't start writing the email to my manager",
        "user_state": {"time_of_day": "10:00", "position_in_chair": "normal", "minutes_since_last_stood": 30},
        "is_adhd": True,
        "expected_tool": None,
        "expected_llm_tool": "adhd_assist_tool",
        "description": "ADHD task initiation - should use adhd_assist_tool",
    },
    {
        "scenario": "What's the weather like today?",
        "user_state": {"time_of_day": "12:00", "position_in_chair": "normal", "minutes_since_last_stood": 15},
        "is_adhd": False,
        "expected_tool": "web_search_tool",
        "expected_llm_tool": "web_search_tool",
        "description": "Weather question - should use web_search_tool",
    },
    {
        "scenario": "I've been procrastinating on this assignment for hours and I'm exhausted",
        "user_state": {"time_of_day": "21:30", "position_in_chair": "slouching", "minutes_since_last_stood": 120},
        "is_adhd": True,
        "expected_tool": None,
        "expected_llm_tool": "adhd_assist_tool",
        "description": "Evening ADHD with fatigue - should use adhd_assist_tool",
    },
    {
        "scenario": "Write me a poem about cats",
        "user_state": {"time_of_day": "14:00", "position_in_chair": "normal", "minutes_since_last_stood": 20},
        "is_adhd": False,
        "expected_tool": None,
        "expected_llm_tool": None,
        "description": "Creative request - should NOT use adhd_assist_tool",
    },
]


def call_model(client: OpenAI, model: str, scenario: str, user_state: dict) -> dict:
    """Send scenario to LLM and parse tool call response."""
    system_prompt = (
        "You are a helpful assistant. You have access to tools. "
        "Use the appropriate tool when the user's request matches a tool's purpose. "
        "If no tool is appropriate, respond directly without calling any tool.\n\n"
        f"User context: time={user_state['time_of_day']}, "
        f"position={user_state['position_in_chair']}, "
        f"minutes since last stood={user_state['minutes_since_last_stood']}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scenario},
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=256,
        )
    except Exception as e:
        return {"error": str(e), "tool_calls": [], "message": ""}

    msg = response.choices[0].message
    tool_calls_raw = msg.tool_calls or []

    # Map LLM tool names to environment tool names
    env_tool_calls = []
    llm_tool_names = []
    for tc in tool_calls_raw:
        llm_tool_names.append(tc.function.name)
        env_name = TOOL_NAME_MAP.get(tc.function.name, tc.function.name)
        env_tool_calls.append(env_name)

    # Extract message from tool args or content
    message = msg.content or ""
    if not message and tool_calls_raw:
        import json
        try:
            args = json.loads(tool_calls_raw[0].function.arguments)
            message = args.get("coaching_message", args.get("query", ""))
        except (json.JSONDecodeError, IndexError):
            pass

    return {
        "tool_calls": env_tool_calls,
        "llm_tool_names": llm_tool_names,
        "message": message,
        "error": None,
    }


def run_model_tests(client: OpenAI, model: str) -> dict:
    """Run all test cases against a model and return results."""
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model}")
    print(f"{'=' * 60}")

    correct = 0
    total = len(TEST_CASES)
    total_reward = 0.0
    results = []

    for i, tc in enumerate(TEST_CASES):
        print(f"\n--- Test {i+1}: {tc['description']} ---")
        print(f"  Scenario: {tc['scenario']}")

        resp = call_model(client, model, tc["scenario"], tc["user_state"])

        if resp.get("error"):
            print(f"  ERROR: {resp['error']}")
            results.append({"test": i+1, "error": resp["error"]})
            continue

        print(f"  LLM tools: {resp['llm_tool_names']}")
        print(f"  Message: {resp['message'][:80]}...")

        # Score with environment rubric
        action = ADHDAction(tool_calls=resp["tool_calls"], message=resp["message"])
        scoring = score_rubric(
            action, tc["scenario"], tc["user_state"],
            tc["is_adhd"], tc["expected_tool"],
        )
        reward = scoring["total_score"]
        total_reward += reward

        # Check if LLM picked the right tool
        llm_picked = resp["llm_tool_names"][0] if resp["llm_tool_names"] else None
        expected = tc["expected_llm_tool"]

        if expected is None:
            # For "no tool expected", correct if didn't pick adhd_assist_tool
            tool_correct = llm_picked != "adhd_assist_tool"
        else:
            tool_correct = llm_picked == expected

        if tool_correct:
            correct += 1

        status = "CORRECT" if tool_correct else "WRONG"
        print(f"  Tool choice: {status} (picked={llm_picked}, expected={expected})")
        print(f"  Reward: {reward}")
        results.append({
            "test": i+1,
            "tool_correct": tool_correct,
            "reward": reward,
            "picked": llm_picked,
            "expected": expected,
        })

    avg_reward = total_reward / total if total > 0 else 0
    print(f"\n--- Summary for {model} ---")
    print(f"  Tool accuracy: {correct}/{total}")
    print(f"  Avg reward: {avg_reward:.3f}")

    return {
        "model": model,
        "correct": correct,
        "total": total,
        "avg_reward": avg_reward,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test LLM tool calling with ADHD environment")
    parser.add_argument("--model", type=str, help="Model to test (default: first in list)")
    parser.add_argument("--all", action="store_true", help="Test all models and show leaderboard")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Run: export HF_TOKEN=hf_...")
        sys.exit(1)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    if args.all:
        models = MODELS
    elif args.model:
        models = [args.model]
    else:
        models = [MODELS[0]]

    all_results = []
    for model in models:
        result = run_model_tests(client, model)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("MODEL LEADERBOARD")
        print(f"{'=' * 60}")
        print(f"{'Model':<40} {'Accuracy':>10} {'Avg Reward':>12}")
        print("-" * 62)

        for r in sorted(all_results, key=lambda x: x["avg_reward"], reverse=True):
            print(f"{r['model']:<40} {r['correct']}/{r['total']:>8} {r['avg_reward']:>11.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark LLMs against the ADHD coaching environment rubric.

Runs models locally via transformers + torch (ROCm compatible).
Scores responses by calling the adhd_env server via HTTP.

Each model uses its native tool calling format:
  - SmolLM3-3B: xml_tools parameter in apply_chat_template
  - Qwen3-8B: tools parameter (Hermes-style) in apply_chat_template
  - OLMo-3-7B-Instruct: tools parameter in apply_chat_template

The adhd_env server must be running locally before running this script.
Start it with: cd adhd_env && .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000

Requires: transformers>=4.57.0, accelerate, torch, requests
Install in base devcontainer venv (NOT adhd_env/.venv).

Outputs:
  - benchmark_results.json  (summary for web page)
  - benchmark_details.csv   (per-prompt detailed results)

Usage:
    python benchmark.py
    python benchmark.py --episodes 15
    python benchmark.py --env-url http://localhost:8000

View results:
    python -m http.server 8080
    # Then open http://localhost:8080/benchmark.html in your browser
"""

import argparse
import csv
import gc
import json
import re
import time
from dataclasses import dataclass

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tool definitions in JSON schema format (used by all models)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "adhd_coach_tool",
            "description": "Help a user with ADHD task initiation paralysis. Call this tool when someone is stuck, can't start, avoiding, exhausted, procrastinating, overwhelmed, can't focus, or staring at a blank page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {
                        "type": "string",
                        "description": "Restate the user's original message, then append coaching advice that incorporates the user context (time of day, posture, minutes since last stood).",
                    },
                },
                "required": ["user_message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "Search the web for information. Use for general knowledge questions, weather, facts, latest news, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# SmolLM3 uses a slightly different tool schema (no "type"/"function" wrapper)
TOOLS_SMOLLM3 = [
    {
        "name": "adhd_coach_tool",
        "description": "Help a user with ADHD task initiation paralysis. Call this tool when someone is stuck, can't start, avoiding, exhausted, procrastinating, overwhelmed, can't focus, or staring at a blank page.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_message": {
                    "type": "string",
                    "description": "Restate the user's original message, then append coaching advice that incorporates the user context (time of day, posture, minutes since last stood).",
                },
            },
            "required": ["user_message"],
        },
    },
    {
        "name": "web_search_tool",
        "description": "Search the web for information. Use for general knowledge questions, weather, facts, latest news, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    },
]

MODELS = [
    "HuggingFaceTB/SmolLM3-3B",
    "allenai/Olmo-3-7B-Instruct",
]
# MODELS = [
#     "HuggingFaceTB/SmolLM3-3B",
#     "Qwen/Qwen3-8B",
#     "allenai/Olmo-3-7B-Instruct",
# ]

SYSTEM_PROMPT = """\
You are a helpful personal assistant with access to tools: \
adhd_coach_tool for helping users struggling with ADHD task initiation, and web_search_tool for answering general knowledge questions. \
When a user's request matches one of your available tools, call that tool. \
If no tool is relevant, respond directly with plain text.

User context: time={time_of_day}, position={position_in_chair}, minutes_since_last_stood={minutes_since_last_stood}

How to use "User context" information: Incorporate this context into your tool call responses. \
For example if the user has not gotten up from their seat for a long time, include a suggestion \
to get up, go get a drink of water or a snack and then come back to figure out next steps.

time:
    * Executive function tends to ebb in the middle of the day more likely leading to the user being stuck. More likely to need an adhd tool call
position:
    * If the user is slouching then they probably need to get up and move or get a drink of water. Their executive function and energy level is low
    * If the user is standing they have probably recognized their energy and executive function was low so moving their body is not an immediate fix
minutes_since_last_stood:
    * If it has been more than 45 minutes but less than 120 minutes since the user last stood then before any planning they should get up, walk around a bit and maybe get a drink of water \
before attempting any fixes to being stuck
    * If it has been more than 120 minutes then not only should the user get up but they should go do some sort of physical activity outside that is not \
computer related - chop wood, do some woodworking, walk around the garden, go birding a bit, take some photographs - NO VIDEO GAMING on computer or on the phone"""


@dataclass
class EpisodeResult:
    model_name: str
    scenario: str
    time_of_day: str
    position_in_chair: str
    minutes_since_last_stood: int
    model_response: str
    tool_calls: list
    total_score: float
    tool_score: float
    state_score: float
    relevance_score: float


@dataclass
class ModelSummary:
    model_name: str
    avg_total_score: float
    avg_tool_score: float
    avg_state_score: float
    avg_relevance_score: float


def parse_tool_calls_smollm3(raw_output: str) -> tuple[list[str], str]:
    """Parse SmolLM3 XML-style tool calls.

    SmolLM3 outputs: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
    """
    tool_calls = []
    message = raw_output.strip()

    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", raw_output, re.DOTALL):
        try:
            call = json.loads(match.group(1).strip())
            tool_calls.append(call.get("name", ""))
            # Extract the user_message argument as the response text
            args = call.get("arguments", {})
            if "user_message" in args:
                message = args["user_message"]
            elif "query" in args:
                message = args["query"]
        except json.JSONDecodeError:
            continue

    # If no tool_call tags found, the response text is the raw output
    if not tool_calls:
        # Strip any thinking tags
        message = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL).strip()

    return tool_calls, message


def parse_tool_calls_hermes(raw_output: str) -> tuple[list[str], str]:
    """Parse Hermes-style tool calls (Qwen3, OLMo-3).

    Hermes format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
    OLMo-3 may also use: <function_calls>tool_name(...)</function_calls>
    """
    tool_calls = []
    message = raw_output.strip()

    # Strip thinking blocks
    message = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL).strip()

    # Try Hermes XML tool_call format (shared by Qwen3 and some OLMo-3 outputs)
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", raw_output, re.DOTALL):
        try:
            call = json.loads(match.group(1).strip())
            tool_calls.append(call.get("name", ""))
            args = call.get("arguments", {})
            if "user_message" in args:
                message = args["user_message"]
            elif "query" in args:
                message = args["query"]
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls, message

    # Try OLMo-3 pythonic format: <function_calls>tool_name(arg=val)</function_calls>
    fc_match = re.search(r"<function_calls>(.*?)</function_calls>", raw_output, re.DOTALL)
    if fc_match:
        for line in fc_match.group(1).strip().splitlines():
            func_match = re.match(r"(\w+)\(", line.strip())
            if func_match:
                tool_calls.append(func_match.group(1))
                # Try to extract user_message argument
                arg_match = re.search(r'user_message=["\'](.+?)["\']', line)
                if arg_match:
                    message = arg_match.group(1)

    return tool_calls, message


def build_messages(scenario: str, state: dict) -> list[dict]:
    """Build the chat messages with user context baked into system prompt."""
    system = SYSTEM_PROMPT.format(**state)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": scenario},
    ]


def generate_response_smollm3(model, tokenizer, scenario: str, state: dict) -> str:
    """Generate response using SmolLM3's xml_tools format."""
    messages = build_messages(scenario, state)

    input_text = tokenizer.apply_chat_template(
        messages,
        xml_tools=TOOLS_SMOLLM3,
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def generate_response_hermes(model, tokenizer, scenario: str, state: dict) -> str:
    """Generate response using Hermes-style tools (Qwen3, OLMo-3)."""
    messages = build_messages(scenario, state)

    input_text = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# Map model name -> (generate_fn, parse_fn)
MODEL_HANDLERS = {
    "HuggingFaceTB/SmolLM3-3B": (generate_response_smollm3, parse_tool_calls_smollm3),
    "Qwen/Qwen3-8B": (generate_response_hermes, parse_tool_calls_hermes),
    "allenai/Olmo-3-7B-Instruct": (generate_response_hermes, parse_tool_calls_hermes),
}


def load_model(model_name: str):
    """Load model and tokenizer onto GPU."""
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory after finishing with a model.

    NOTE: If we hit OOM loading the next model, we may need to also call
    torch.cuda.synchronize() or move to subprocess-per-model isolation.
    """
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def env_reset(env_url: str) -> dict:
    """Call /reset on the environment server, return observation."""
    r = requests.post(f"{env_url}/reset")
    r.raise_for_status()
    return r.json()


def env_step(env_url: str, tool_calls: list[str], message: str) -> dict:
    """Call /step on the environment server, return scored observation."""
    r = requests.post(f"{env_url}/step", json={
        "action": {"tool_calls": tool_calls, "message": message}
    })
    r.raise_for_status()
    return r.json()


def run_benchmark(
    num_episodes: int = 4,
    env_url: str = "http://localhost:8000",
) -> tuple[list[ModelSummary], list[EpisodeResult]]:
    """Run the full benchmark across all models.

    Args:
        num_episodes: Number of episodes per model. Single config point.
        env_url: URL of the running adhd_env server.
    """
    # Verify environment is reachable
    try:
        r = requests.get(f"{env_url}/health")
        r.raise_for_status()
        print(f"Environment server healthy: {r.json()}")
    except requests.ConnectionError:
        print(f"ERROR: Cannot reach environment at {env_url}")
        print("Start it with: cd adhd_env && .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000")
        raise SystemExit(1)

    all_results: list[EpisodeResult] = []
    summaries: list[ModelSummary] = []

    for model_name in MODELS:
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 60}")

        generate_fn, parse_fn = MODEL_HANDLERS[model_name]
        model, tokenizer = load_model(model_name)
        model_results: list[EpisodeResult] = []

        for i in range(num_episodes):
            # Get a scenario from the environment
            reset_data = env_reset(env_url)
            obs = reset_data["observation"]
            scenario = obs["scenario"]
            state = obs["state"]

            print(f"\n  Episode {i+1}/{num_episodes}: {scenario[:50]}...")
            print(f"  State: {state}")

            # Generate model response using model-specific handler
            raw = generate_fn(model, tokenizer, scenario, state)
            print(f"  Raw output: {raw[:200]}...")

            # Parse tool calls using model-specific parser
            tool_calls, message = parse_fn(raw)
            print(f"  Parsed tools: {tool_calls}")
            print(f"  Parsed message: {message[:100]}...")

            # Score via environment
            step_data = env_step(env_url, tool_calls, message)
            scoring = step_data["observation"]["scoring"]
            criteria = scoring["criteria"]

            result = EpisodeResult(
                model_name=model_name,
                scenario=scenario,
                time_of_day=state["time_of_day"],
                position_in_chair=state["position_in_chair"],
                minutes_since_last_stood=state["minutes_since_last_stood"],
                model_response=message,
                tool_calls=tool_calls,
                total_score=scoring["total_score"],
                tool_score=criteria["tool_calling"]["score"],
                state_score=criteria["state_awareness"]["score"],
                relevance_score=criteria["adhd_relevance"]["score"],
            )
            model_results.append(result)
            print(f"  Score: {result.total_score} (tool={result.tool_score}, state={result.state_score}, rel={result.relevance_score})")

        all_results.extend(model_results)

        summary = ModelSummary(
            model_name=model_name,
            avg_total_score=round(sum(r.total_score for r in model_results) / len(model_results), 3),
            avg_tool_score=round(sum(r.tool_score for r in model_results) / len(model_results), 3),
            avg_state_score=round(sum(r.state_score for r in model_results) / len(model_results), 3),
            avg_relevance_score=round(sum(r.relevance_score for r in model_results) / len(model_results), 3),
        )
        summaries.append(summary)
        print(f"\n  Summary: avg_total={summary.avg_total_score}, avg_tool={summary.avg_tool_score}, avg_state={summary.avg_state_score}, avg_rel={summary.avg_relevance_score}")

        print(f"  Unloading {model_name}...")
        unload_model(model, tokenizer)

    return summaries, all_results


def write_json(summaries: list[ModelSummary], output_path: str = "benchmark_results.json"):
    """Write summary JSON for the benchmark web page."""
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": [
            {
                "model_name": s.model_name,
                "avg_total_score": s.avg_total_score,
                "avg_tool_score": s.avg_tool_score,
                "avg_state_score": s.avg_state_score,
                "avg_relevance_score": s.avg_relevance_score,
            }
            for s in sorted(summaries, key=lambda x: x.avg_total_score, reverse=True)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nJSON written to {output_path}")


def write_csv(results: list[EpisodeResult], output_path: str = "benchmark_details.csv"):
    """Write detailed CSV with all per-prompt results."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model Name",
            "Final Score",
            "Prompt",
            "Time of Day",
            "Position in Chair",
            "Time Since Last Break",
            "Model Response",
            "Tool Score",
            "State Score",
            "Relevance Score",
        ])
        for r in results:
            writer.writerow([
                r.model_name,
                r.total_score,
                r.scenario,
                r.time_of_day,
                r.position_in_chair,
                r.minutes_since_last_stood,
                r.model_response,
                r.tool_score,
                r.state_score,
                r.relevance_score,
            ])
    print(f"CSV written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs against ADHD coaching rubric")
    parser.add_argument("--episodes", type=int, default=4, help="Number of episodes per model (default: 4)")
    parser.add_argument("--env-url", type=str, default="http://localhost:8000", help="URL of adhd_env server")
    args = parser.parse_args()

    print(f"Running benchmark: {len(MODELS)} models x {args.episodes} episodes")
    print(f"Models: {MODELS}")
    print(f"Environment: {args.env_url}")

    summaries, results = run_benchmark(num_episodes=args.episodes, env_url=args.env_url)

    write_json(summaries)
    write_csv(results)

    # Print leaderboard
    print(f"\n{'=' * 70}")
    print("LEADERBOARD")
    print(f"{'=' * 70}")
    print(f"{'Model':<40} {'Total':>8} {'Tool':>8} {'State':>8} {'Relev':>8}")
    print("-" * 72)
    for s in sorted(summaries, key=lambda x: x.avg_total_score, reverse=True):
        print(f"{s.model_name:<40} {s.avg_total_score:>8.3f} {s.avg_tool_score:>8.3f} {s.avg_state_score:>8.3f} {s.avg_relevance_score:>8.3f}")


if __name__ == "__main__":
    main()

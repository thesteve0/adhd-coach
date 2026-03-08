# Plan: ADHD Environment Phase 2 — State, Rubric, and LLM Integration

## Context

The ADHD coaching OpenEnv environment is deployed and working at https://thesteve0-adhd-env.hf.space. V1 has a single hardcoded scenario with tool-calling-only scoring. The judges specifically asked for **state tracking** and **tool calling evaluation** as the innovation. We need to add state variables ("knobs"), formalize the rubric with positive/negative test cases, and connect an actual LLM via HF inference providers to validate the whole pipeline end-to-end.

---

## Task 1: Add State Variables (3 knobs)

**Files to modify:**
- `adhd_env/server/adhd_env_environment.py`
- `adhd_env/test_environment.py`

**Changes to `adhd_env_environment.py`:**
- Add `import random`
- Add `self.current_user_state: dict = {}` in `__init__`
- In `reset()`: generate randomized state with 3 variables:
  - `time_of_day`: string `"HH:MM"` in 24h format (hours 6–22, minutes 0–59)
  - `position_in_chair`: one of `"normal"`, `"slouching"`, `"standing"`
  - `minutes_since_last_stood`: int 0–240
- Store as `self.current_user_state`, pass to `ADHDObservation(state=self.current_user_state)`
- In `step()`: replace `state={}` with `state=self.current_user_state`

**Changes to `test_environment.py`:**
- After `reset()`, assert state has all 3 keys with valid values
- Assert `position_in_chair` is one of `"normal"`, `"slouching"`, `"standing"`
- Assert `minutes_since_last_stood` is 0–240
- Run `reset()` 10x and assert we get at least 2 distinct states (variety check)

**Test:** `cd adhd_env && .venv/bin/python test_environment.py`
**Deploy:** `cd adhd_env && .venv/bin/openenv push --repo-id TheSteve0/adhd-env`

---

## Task 2: Rubric with Positive & Negative Test Cases

**Files to modify:**
- `adhd_env/reward.py` — add new scoring functions + combined rubric
- `adhd_env/server/adhd_env_environment.py` — add scenario lists, use rubric in `step()`
- `adhd_env/test_environment.py` — add rubric test cases, update assertions

**New scoring functions in `reward.py`:**

1. `score_tool_calling(action, scenario) -> float` — updated version:
   - Detect if scenario is ADHD-related via keyword indicators
   - **ADHD scenario:**
     - 1.0: called `adhd_task_initiation_coach`
     - 0.25: called a valid ADHD tool but not the primary one (`set_timer`, `break_down_task`)
     - 0.0: called no tool at all
     - **-0.5**: called a non-ADHD tool (e.g., `web_search_tool`) — worse than no tool
   - **Non-ADHD scenario** (each scenario tagged with `expected_tool`):
     - **-0.5**: called `adhd_task_initiation_coach` — penalized, wrong domain
     - **0.7**: called the expected non-ADHD tool (e.g., `web_search_tool` for factual questions) — slightly rewarded
     - **0.5**: called no tool — neutral
     - **0.5**: called a non-ADHD tool when none was expected — neutral (not harmful)

   Design notes:
   - Using the wrong-domain tool is penalized (-0.5) in both directions: ADHD tool on non-ADHD prompt, or non-ADHD tool on ADHD prompt.
   - Using no tool at all is always better than using the wrong tool.
   - Non-ADHD scenarios are tagged with their expected tool (or None) to distinguish factual questions (should use web_search) from creative requests (no tool needed).

2. `score_state_awareness(action, user_state) -> float`
   - 1.0: response mentions movement/stretching when sitting 60+ min or slouching
   - 1.0: response suggests simpler tasks when evening (hour >= 20)
   - 0.5: generic response (default — neutral, not penalized)
   - Keyword-based matching (simple for hackathon)

3. `score_rubric(action, scenario, user_state, expected_tool=None) -> dict`
   - Weighted combination: tool_calling 40% + state_awareness 30% + adhd_relevance 30%
   - `expected_tool` is passed for non-ADHD scenarios to enable appropriate-tool rewarding
   - Returns dict with `total_score` and per-criterion breakdown
   - Total score clamped to 0.0–1.0 range (penalties can push raw score negative)

**Scenario lists in `adhd_env_environment.py`:**

ADHD scenarios (~10):
- "I can't start writing the email to my manager"
- "I've been staring at this blank document for 30 minutes"
- "I need to make a phone call but I keep putting it off"
- "I'm stuck on starting this presentation"
- "I've been avoiding this report all day"
- "I don't know how to begin this project proposal"
- "I keep switching tabs instead of starting my work"
- "I'm overwhelmed by this task list"
- "I can't focus on writing this code review"
- "I've been procrastinating on this assignment for hours"

Non-ADHD scenarios (~5, each tagged with expected tool):
- ("What's the weather like today?", "web_search_tool") — factual, needs search
- ("What is the latest revenue for IBM?", "web_search_tool") — factual, needs search
- ("What is the capital of France?", "web_search_tool") — factual, needs search
- ("Write me a poem about cats", None) — creative, no tool needed
- ("Translate this sentence to Spanish", None) — language task, no tool needed

`reset()` picks ADHD 80% / non-ADHD 20% randomly. For non-ADHD, stores the expected_tool in internal state (not exposed to the model).

**Test cases — new `test_rubric()` function:**
- POSITIVE: ADHD scenario + primary tool + state-aware message → high reward
- NEGATIVE (wrong tool on ADHD): ADHD scenario + web_search_tool → penalized (reward < 0.3)
- NEGATIVE (ADHD tool on non-ADHD): non-ADHD scenario + adhd tool → penalized (reward < 0.3)
- SLIGHTLY POSITIVE: non-ADHD factual scenario + web_search_tool → slightly rewarded (~0.7)
- NEUTRAL: non-ADHD creative scenario + no tool → neutral (~0.5)
- MEDIUM: ADHD scenario + primary tool + generic message → moderate reward

**Test:** `cd adhd_env && .venv/bin/python test_environment.py`
**Deploy:** `cd adhd_env && .venv/bin/openenv push --repo-id TheSteve0/adhd-env`

---

## Task 3: LLM Inference with Tool Calling (Multi-Model Leaderboard)

**Files to create/modify:**
- `adhd_env/test_with_model.py` — NEW: end-to-end test with LLM, multi-model support
- `adhd_env/pyproject.toml` — add `openai` to dev dependencies

**Architecture: trivial model swapping for leaderboard comparison.**

Models are defined as a list at the top of the script. Each model runs the same test cases and results are compared side-by-side.

```python
MODELS = [
    "HuggingFaceTB/SmolLM3-3B",           # Primary: truly open source, HF staff recommended
    "Qwen/Qwen3.5-9B",                     # Backup 1
    "allenai/Olmo-3-7B-Instruct",          # Backup 2
]
```

The script accepts `--model` to run a single model, or `--all` to run all models and produce a comparison leaderboard. Default: first model in the list.

**Tool definitions (what the LLM sees):**
- `adhd_assist_tool` — "Help a user with ADHD task initiation paralysis. Use when someone is stuck starting a task, procrastinating, or overwhelmed by executive function challenges."
- `web_search_tool` — "Search the web for information. Use for general knowledge questions, weather, facts, latest news, etc."

Both are dummy tools — not actually implemented. We only care about which one the model chooses to call.

**Tool name mapping (LLM → environment):**
- `adhd_assist_tool` → `adhd_task_initiation_coach`
- `web_search_tool` → `web_search_tool` (not in env's valid tools, so will be scored accordingly)

**Test cases (4+ scenarios):**
1. ADHD task initiation → expect `adhd_assist_tool`
2. Weather question → expect `web_search_tool`
3. Evening ADHD procrastination with fatigue state → expect `adhd_assist_tool`
4. Creative request (poem) → expect no tool (or either is OK, but NOT adhd_assist_tool)

**Flow per test case:**
1. Send scenario + user state + tools to LLM via `OpenAI(base_url="https://router.huggingface.co/v1", api_key=os.environ["HF_TOKEN"])`
2. Parse `response.choices[0].message.tool_calls`
3. Map LLM tool names → environment tool names
4. Extract coaching message from tool arguments or content
5. Create `ADHDAction` and call `env.step(action)` to get reward
6. Report per-scenario and aggregate results

**Leaderboard output format:**
```
=== MODEL LEADERBOARD ===
Model                              Accuracy  Avg Reward
HuggingFaceTB/SmolLM3-3B          3/4       0.72
Qwen/Qwen3.5-9B                   4/4       0.85
allenai/Olmo-3-7B-Instruct        2/4       0.55
```

**Security:** HF_TOKEN is read from `os.environ["HF_TOKEN"]` only. Never hardcoded. The script fails with a clear error if the env var is not set.

**Dependency:** Add `"openai>=1.0.0"` to `[project.optional-dependencies] dev` to keep the deployed image lean.

**Test:** `cd adhd_env && uv sync && .venv/bin/python test_with_model.py` (single model) or `--all` for leaderboard.
No redeployment needed (test script is local-only).

---

## Implementation Order

1. **Task 1** → Task 2 depends on state being available
2. **Task 2** → Task 3 depends on the rubric being in place
3. **Task 3** → validates the whole pipeline end-to-end

**Between each task**: stop, test, play with results, commit, push to GitHub, then user confirms ready for next task.

## Verification

After all 3 tasks:
1. `cd adhd_env && .venv/bin/python test_environment.py` — all direct tests pass
2. `cd adhd_env && .venv/bin/python test_environment.py --http` — all HTTP tests pass (with server running)
3. `cd adhd_env && .venv/bin/python test_with_model.py` — LLM picks correct tools, scores are reasonable
4. `.venv/bin/openenv validate --verbose` — still passes all 4 deployment modes
5. `.venv/bin/openenv push --repo-id TheSteve0/adhd-env` — deployed and working
6. `curl -s -X POST https://thesteve0-adhd-env.hf.space/reset | python3 -m json.tool` — shows state variables in response

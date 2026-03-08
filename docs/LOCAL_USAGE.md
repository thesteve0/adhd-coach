# Local Usage Guide - ADHD Coaching Environment

This guide shows you how to run, test, and interact with the ADHD coaching evaluation environment on your local machine.

---

## Prerequisites

You're in the ROCm devcontainer with all dependencies already installed via `uv`.

---

## Quick Start

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Run Manual Tests

The fastest way to see the environment in action:

```bash
PYTHONPATH=/workspaces/adhd-coach python scripts/test_environment_manual.py
```

**What this does:**
- Tests 3 scenarios: good response (with tool), bad response (no tool), medium response (wrong tool)
- Shows scores: 1.0, 0.0, and 0.5 respectively
- Validates the environment is working correctly

**Expected Output:**
```
================================================================================
ADHD COACHING ENVIRONMENT - MANUAL TESTING (V1)
================================================================================

 Environment Info:
  Version: 0.1.0-v1
  Stage: minimal
  Features: {'tool_calling_evaluation': True, 'state_tracking': False, ...}

TEST 1: Good Response (with primary tool)
  Reward: 1.00
  Explanation: Called primary tool (adhd_task_initiation_coach)

TEST 2: Bad Response (no tool)
  Reward: 0.00
  Explanation: No tools called

TEST 3: Medium Response (secondary tool)
  Reward: 0.50
  Explanation: Called tool but not primary
```

---

## Running the FastAPI Server

### Start the Server

```bash
source .venv/bin/activate
python -m uvicorn src.environment.server:app --host 0.0.0.0 --port 8001
```

**Output:**
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

The server is now running and ready to accept requests!

### Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Environment info |
| `/info` | GET | Detailed environment metadata |
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Score a coaching response |

---

## Interacting with the Environment

### Option 1: Using curl (Terminal)

Open a **new terminal** (keep the server running in the first one).

#### Get Environment Info
```bash
curl http://localhost:8001/
```

**Response:**
```json
{
  "name": "ADHD Task Initiation Coaching Environment",
  "version": "0.1.0-v1",
  "stage": "minimal",
  "endpoints": {
    "info": "GET /info",
    "reset": "POST /reset",
    "step": "POST /step",
    "health": "GET /health"
  }
}
```

#### Reset Environment
```bash
curl -X POST http://localhost:8001/reset
```

**Response:**
```json
{
  "observation": {
    "scenario": "I can't start writing the email to my manager",
    "state": {},
    "done": false,
    "reward": 0.0,
    "metadata": {
      "version": "v1",
      "note": "State tracking will be added in afternoon iteration"
    }
  },
  "reward": 0.0,
  "done": false
}
```

#### Score a Good Response (with tool)
```bash
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": ["adhd_task_initiation_coach"],
    "message": "Open your email draft and type just the subject line."
  }'
```

**Response:**
```json
{
  "observation": {
    "scenario": "I can't start writing the email to my manager",
    "state": {},
    "done": true,
    "reward": 1.0,
    "metadata": {
      "version": "v1",
      "total_score": 1.0,
      "criteria": {
        "tool_calling": {
          "score": 1.0,
          "weight": 1.0,
          "tools_called": ["adhd_task_initiation_coach"],
          "explanation": "Called primary tool (adhd_task_initiation_coach)"
        }
      }
    }
  },
  "reward": 1.0,
  "done": true
}
```

#### Score a Bad Response (no tool)
```bash
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": [],
    "message": "What do you want to work on?"
  }'
```

**Response:**
```json
{
  "observation": {
    "reward": 0.0,
    "metadata": {
      "criteria": {
        "tool_calling": {
          "score": 0.0,
          "explanation": "No tools called"
        }
      }
    }
  },
  "reward": 0.0,
  "done": true
}
```

---

### Option 2: Using Python (Programmatic)

Create a test script or use Python interactively:

```python
import requests

BASE_URL = "http://localhost:8001"

# Reset environment
response = requests.post(f"{BASE_URL}/reset")
result = response.json()
print(f"Scenario: {result['observation']['scenario']}")

# Score a response with tool calling
action = {
    "tool_calls": ["adhd_task_initiation_coach"],
    "message": "Open the document and type just the title."
}
response = requests.post(f"{BASE_URL}/step", json=action)
result = response.json()

print(f"Reward: {result['reward']}")
print(f"Explanation: {result['observation']['metadata']['criteria']['tool_calling']['explanation']}")
```

**Output:**
```
Scenario: I can't start writing the email to my manager
Reward: 1.0
Explanation: Called primary tool (adhd_task_initiation_coach)
```

---

### Option 3: Using the Environment Directly (No Server)

If you don't want to run the server, you can use the environment class directly:

```python
# File: test_direct.py
import asyncio
from src.environment import ADHDEnvironment, ADHDAction

async def test():
    env = ADHDEnvironment()

    # Reset
    result = env.reset()
    print(f"Scenario: {result.observation.scenario}")

    # Score an action
    action = ADHDAction(
        tool_calls=["adhd_task_initiation_coach"],
        message="Open the doc and type the title."
    )
    result = await env.step(action)

    print(f"Reward: {result.reward}")
    print(f"Done: {result.done}")

asyncio.run(test())
```

**Run it:**
```bash
source .venv/bin/activate
PYTHONPATH=/workspaces/adhd-coach python test_direct.py
```

---

## Understanding the Scoring

### Current Version (V1): Tool Calling Only

The environment currently evaluates **one thing**: Did the model call the appropriate tool?

| Tool Called | Score | Explanation |
|-------------|-------|-------------|
| `adhd_task_initiation_coach` | 1.0 | ✅ Called the primary tool |
| `set_timer` or `break_down_task` | 0.5 | ⚠️ Called a valid tool, but not the primary one |
| No tool / empty list | 0.0 | ❌ No tools called |
| Invalid tool name | 0.0 | ❌ Invalid tool |

### Available Tools

Three tools are currently defined:

1. **`adhd_task_initiation_coach`** (PRIMARY) - Main coaching tool for task initiation
2. **`set_timer`** (SECONDARY) - Focus timer for task boxing
3. **`break_down_task`** (SECONDARY) - Task decomposition helper

---

## Testing Different Scenarios

### Test 1: Perfect Response
```bash
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": ["adhd_task_initiation_coach"],
    "message": "Open your email and type just the recipient name. Stop there."
  }'
```
**Expected Score:** 1.0

### Test 2: Wrong Tool
```bash
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": ["set_timer"],
    "message": "Let me set a 5-minute timer for you."
  }'
```
**Expected Score:** 0.5

### Test 3: No Tool
```bash
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": [],
    "message": "What would you like to work on today?"
  }'
```
**Expected Score:** 0.0

### Test 4: Multiple Tools (Including Primary)
```bash
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": ["adhd_task_initiation_coach", "set_timer"],
    "message": "Let me help you start. First, open the document."
  }'
```
**Expected Score:** 1.0 (has primary tool)

---

## Current Limitations (V1)

This is the **minimal version** to validate the approach:

- ✅ **Tool calling evaluation**: Working
- ❌ **State tracking**: Not yet (coming in afternoon iteration)
- ❌ **Scenario variety**: Single hardcoded scenario
- ❌ **Multiple scoring criteria**: Only tool calling right now
- ❌ **State-aware scoring**: Not yet

**Scenario is hardcoded as:** `"I can't start writing the email to my manager"`

---

## Troubleshooting

### Server won't start
```bash
# Check if port 8001 is already in use
lsof -i :8001

# Kill existing process if needed
pkill -f uvicorn

# Try again
source .venv/bin/activate
python -m uvicorn src.environment.server:app --host 0.0.0.0 --port 8001
```

### Import errors
```bash
# Make sure you're using PYTHONPATH
PYTHONPATH=/workspaces/adhd-coach python your_script.py

# Or activate the venv first
source .venv/bin/activate
```

### Can't connect to server
```bash
# Check server is running
curl http://localhost:8001/health

# Expected: {"status": "healthy"}
```

---

## Next Steps

After validating the minimal environment works:

1. **Afternoon**: Add state tracking (6 dimensions: sitting time, time of day, etc.)
2. **Afternoon**: Add scenario variety (30+ different task initiation scenarios)
3. **Evening**: Add more scoring criteria (not_question, length, state-aware)
4. **Day 2**: Deploy to HuggingFace Spaces
5. **Day 2**: Train a model using the environment

---

## Quick Reference

### Start Server
```bash
source .venv/bin/activate
python -m uvicorn src.environment.server:app --host 0.0.0.0 --port 8001
```

### Run Tests
```bash
source .venv/bin/activate
PYTHONPATH=/workspaces/adhd-coach python scripts/test_environment_manual.py
```

### Test with curl
```bash
# Reset
curl -X POST http://localhost:8001/reset

# Step (with tool)
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{"tool_calls": ["adhd_task_initiation_coach"], "message": "Your response"}'
```

### Stop Server
```
Ctrl+C (in server terminal)
```

---

**Server URL:** http://localhost:8001
**Interactive Docs:** http://localhost:8001/docs (FastAPI auto-generated)

---
title: ADHD Task Initiation Coaching Environment
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - adhd
  - executive-function
---

# ADHD Task Initiation Coaching Environment

An OpenEnv environment that evaluates ADHD coaching response quality. It scores AI coaching responses for task initiation paralysis based on tool calling and response quality.

**Innovation**: State tracking ("knobs") + tool calling evaluation - not just text scoring.

## Quick Start

```python
from adhd_env import ADHDAction, ADHDEnv

# Connect to deployed environment
with ADHDEnv(base_url="https://YOUR-SPACE.hf.space") as env:
    # Get an ADHD scenario
    result = env.reset()
    print(f"Scenario: {result.observation.scenario}")

    # Submit a coaching response for scoring
    result = env.step(ADHDAction(
        tool_calls=["adhd_task_initiation_coach"],
        message="Open email and type just the recipient name. Stop there."
    ))
    print(f"Reward: {result.reward}")  # 1.0
```

## How Scoring Works

The environment evaluates coaching responses on tool calling (V1):

| Action | Reward | Why |
|--------|--------|-----|
| Called `adhd_task_initiation_coach` | **1.0** | Used the primary coaching tool |
| Called `set_timer` or `break_down_task` | **0.5** | Valid tool, but not the primary one |
| No tools called | **0.0** | No tool engagement |

### Available Tools
- `adhd_task_initiation_coach` - Primary coaching tool for task initiation
- `set_timer` - Focus timers for task boxing
- `break_down_task` - Decompose large tasks into micro-steps

## API

### POST /reset
Returns a new ADHD scenario with user state.

### POST /step
Scores a coaching response. Body: `{"action": {"tool_calls": [...], "message": "..."}}`

### GET /health
Health check endpoint.

### GET /schema
JSON schemas for action and observation models.

## Environment Details

### ADHDAction
- `tool_calls` (list[str]) - Tools the model would call
- `message` (str) - The coaching response text

### ADHDObservation
- `scenario` (str) - The ADHD task initiation scenario
- `state` (dict) - User state tracking (sitting time, energy, etc.)
- `scoring` (dict) - Detailed scoring breakdown with explanations
- `reward` (float) - Score 0.0-1.0
- `done` (bool) - Episode complete flag

## Development

```bash
# Install dependencies
cd adhd_env && uv sync

# Run locally
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Test
python test_environment.py        # Direct test
python test_environment.py --http  # HTTP test (server must be running)

# Validate structure
openenv validate --verbose

# Deploy to HF Spaces
openenv push --repo-id USERNAME/adhd-env
```

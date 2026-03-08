# Local Testing Guide

This document describes how to test the ADHD coaching environment locally before deploying to HuggingFace Spaces.

## Quick Start

### 1. Start the Local Server

```bash
# Make script executable (first time only)
chmod +x scripts/run_local_server.sh

# Start server
./scripts/run_local_server.sh
```

Server will start on `http://localhost:8001`

### 2. Test the Deployment

In another terminal:

```bash
# Make script executable (first time only)
chmod +x scripts/test_local_deployment.sh

# Run all tests
./scripts/test_local_deployment.sh
```

## Manual Testing

### Health Check

```bash
curl http://localhost:8001/health
# Expected: {"status":"healthy"}
```

### Environment Info

```bash
curl http://localhost:8001/info | jq '.'
```

Shows:
- Environment name and version
- Available features (tool_calling, state_tracking, etc.)
- Available tools
- Scoring criteria

### Reset Environment

```bash
curl -X POST http://localhost:8001/reset | jq '.'
```

Returns:
- `observation.scenario`: The ADHD scenario (e.g., "I can't start writing the email")
- `observation.state`: User state (empty in V1, will have sitting_time, etc. in V2)
- `done`: false (episode just started)
- `reward`: 0.0

### Step (Score a Response)

**Good action (with primary tool) - Expected reward: 1.0**

```bash
curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": ["adhd_task_initiation_coach"],
    "message": "Open the document and type just the title."
  }' | jq '.'
```

**Bad action (no tool) - Expected reward: 0.0**

```bash
curl -X POST http://localhost:8001/reset  # Reset first

curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": [],
    "message": "What do you want to work on?"
  }' | jq '.'
```

**Medium action (secondary tool) - Expected reward: 0.5**

```bash
curl -X POST http://localhost:8001/reset  # Reset first

curl -X POST http://localhost:8001/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": ["set_timer"],
    "message": "Let me set a 5 minute timer."
  }' | jq '.'
```

## Manual Test Script

Run the Python test script directly:

```bash
PYTHONPATH=/workspaces/adhd-coach .venv/bin/python scripts/test_environment_manual.py
```

This tests:
1. Good action (with primary tool) → reward 1.0
2. Bad action (no tool) → reward 0.0
3. Medium action (secondary tool) → reward 0.5

## API Documentation

Once server is running, visit: http://localhost:8001/docs

This provides interactive Swagger UI for all endpoints.

## Troubleshooting

### "Connection refused" error

Server isn't running. Start it with `./scripts/run_local_server.sh`

### "Module not found" error

Virtual environment not activated or dependencies not installed:

```bash
uv sync
```

### Port 8001 already in use

Kill existing server:

```bash
lsof -ti:8001 | xargs kill -9
```

Or change port in `scripts/run_local_server.sh`

## What Gets Tested

| Test | Endpoint | Expected Result |
|------|----------|-----------------|
| Health check | `GET /health` | `{"status":"healthy"}` |
| Root info | `GET /` | Environment metadata |
| Detailed info | `GET /info` | Features, tools, criteria |
| Reset | `POST /reset` | New scenario + state |
| Good action | `POST /step` | Reward: 1.0 |
| Bad action | `POST /step` | Reward: 0.0 |
| Medium action | `POST /step` | Reward: 0.5 |

## Expected Scoring (V1)

**V1 (Current)**: Tool calling only

- ✅ Called `adhd_task_initiation_coach`: **1.0**
- ⚠️ Called secondary tool (`set_timer`, `break_down_task`): **0.5**
- ❌ No tool called: **0.0**

**V2 (Coming)**: Will add state tracking and multiple criteria

## Files Involved

```
scripts/
├── run_local_server.sh          # Starts FastAPI server
├── test_local_deployment.sh     # Runs all curl tests
└── test_environment_manual.py   # Python test script

src/environment/
├── server.py                    # FastAPI server
├── adhd_env.py                  # Environment logic
├── reward.py                    # Scoring (rubric)
└── models.py                    # Data models
```

## Next Steps After Local Testing

Once local testing passes:

1. Deploy to HuggingFace Spaces using OpenEnv CLI
2. Test deployed environment
3. Add state tracking (V2)
4. Create training demonstration

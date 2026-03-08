# OpenEnv Devcontainer Setup Guide

**Validated on**: ROCm devcontainer (Python 3.12, Ubuntu 24.04)
**Date**: March 6, 2025
**OpenEnv version**: 0.2.1

---

## Installation

```bash
# Install echo_env (includes openenv-core 0.2.1 dependency)
uv add "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"

# Verify installation
uv pip list | grep openenv
# Expected output:
# openenv-core              0.2.1
# openenv-echo-env          0.1.0
```

---

## Import Names (CRITICAL)

```python
# ✓ CORRECT
from echo_env import EchoEnv, EchoAction

# ✗ WRONG - These do NOT work
from openenv_echo import EchoEnv        # ModuleNotFoundError
from openenv_echo_env import EchoEnv    # ModuleNotFoundError
from openenv_core import TextMessage    # Does not exist (use Message instead)
```

**Why**: The package name is `openenv-echo-env` but the module name is `echo_env`.

---

## Local Dev vs Hackathon Deployment

### Understanding the Workflow

**What we're doing NOW (Local Development):**
```bash
# Running uvicorn locally to TEST your environment
python -m uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8001
```
- Purpose: **Test** that your environment logic works
- You connect to: `http://0.0.0.0:8001`
- This validates `reset()`, `step()`, and reward functions locally

**What happens at the HACKATHON (Production Deployment):**

1. **Deploy your ADHD environment to HF Spaces**
   - HF Spaces automatically runs uvicorn for you
   - You get a URL like: `https://your-username-adhd-env.hf.space`

2. **Train in Google Colab**
   ```python
   # In your GRPO training script
   from your_env import ADHDEnv, ADHDAction

   # Connect to your deployed HF Space (NOT localhost!)
   client = ADHDEnv(base_url="https://your-username-adhd-env.hf.space")

   # GRPOTrainer calls reset()/step() on this remote environment
   trainer = GRPOTrainer(
       model="SmolLM3-3B",
       rollout_func=lambda prompts, trainer: rollout_with_env(client, prompts, trainer),
       ...
   )
   trainer.train()  # Training loop queries your HF Space for rewards
   ```

### Hackathon Architecture

```
┌─────────────────────┐         ┌──────────────────────┐         ┌────────────────┐
│  HF Spaces          │         │  Google Colab        │         │  HuggingFace   │
│                     │         │                      │         │  Hub           │
│  Your ADHD Env      │◄────────│  GRPO Trainer        │◄────────│  SmolLM3-3B    │
│  (uvicorn runs      │  HTTP   │  + vLLM              │  load   │  (model)       │
│   automatically)    │  calls  │                      │  model  │                │
│                     │         │                      │         │                │
│  reset()  ──────────┼────────►│  Get initial state   │         │                │
│  step(action) ──────┼────────►│  Get reward for      │         │                │
│  reward_function()  │         │  model's response    │         │                │
└─────────────────────┘         └──────────────────────┘         └────────────────┘
      ▲                                                                    │
      │                                                                    │
      └────────────────────────────────────────────────────────────────────┘
                         Both pushed to HF Hub for submission
```

### Key Differences

| Aspect | Local Dev (Now) | Hackathon (Later) |
|--------|-----------------|-------------------|
| **Server** | You run `python -m uvicorn ...` manually | HF Spaces runs it automatically |
| **URL** | `http://0.0.0.0:8001` | `https://your-space.hf.space` |
| **Purpose** | Test environment logic | Production deployment for training |
| **Client location** | Same machine (devcontainer) | Google Colab (remote) |

**Bottom line**: The uvicorn command is what HF Spaces runs behind the scenes. You just push your code to a Space and it handles the server automatically.

---

## Testing OpenEnv Echo

### Option 1: Remote HF Space (Easiest - No Local Server)

```python
from echo_env import EchoEnv, EchoAction

# Connect to hosted environment
client = EchoEnv(base_url="https://openenv-echo-env.hf.space")

# Reset the environment
result = client.reset()
print("Initial observation:", result.observation)
print("Reward:", result.reward)
print("Done:", result.done)

# Step through the environment
action = EchoAction(message="Hello!")
step_result = client.step(action)
print("\nEchoed message:", step_result.observation.echoed_message)
print("Message length:", step_result.observation.message_length)
print("Reward:", step_result.reward)
```

**Expected Output:**
```
Initial observation: done=False reward=0.0 metadata={} echoed_message='Echo environment ready!' message_length=0
Reward: 0.0
Done: False

Echoed message: Hello!
Message length: 6
Reward: 0.6000000000000001
```

---

### Option 2: Local Server (For Testing Deployment)

**Terminal 1: Start the server**
```bash
# The correct server path is echo_env.server.app:app
python -m uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8001

# Expected output:
# INFO:     Started server process [PID]
# INFO:     Uvicorn running on http://0.0.0.0:8001
```

**Terminal 2: Test the client**
```python
from echo_env import EchoEnv, EchoAction

# Connect to local server
client = EchoEnv(base_url="http://0.0.0.0:8001")

# Same test code as Option 1
result = client.reset()
step_result = client.step(EchoAction(message="Test"))
print(step_result)
```

---

## API Reference

### Core Types

```python
from openenv.core import StepResult

# reset() returns StepResult
result = client.reset()
result.observation  # Environment-specific observation object
result.reward       # float
result.done         # bool

# step(action) returns StepResult
step_result = client.step(action)
step_result.observation  # Updated observation
step_result.reward       # float
step_result.done         # bool
```

### Echo Environment Specifics

```python
# EchoObservation attributes
observation = result.observation
observation.echoed_message  # str - the message that was echoed
observation.message_length  # int - length of the message
observation.reward          # float - same as StepResult.reward
observation.done            # bool - same as StepResult.done
observation.metadata        # dict - additional metadata

# EchoAction
action = EchoAction(message="Your message here")
```

---

## Common Errors and Fixes

### Error: `ModuleNotFoundError: No module named 'openenv_echo_env'`
**Fix**: Use `from echo_env import EchoEnv` (underscore, not hyphen)

### Error: `TypeError: EnvClient.__init__() missing 1 required positional argument: 'base_url'`
**Fix**: Must provide `base_url`:
```python
client = EchoEnv(base_url="https://openenv-echo-env.hf.space")
# OR
client = EchoEnv(base_url="http://0.0.0.0:8001")
```

### Error: `cannot import name 'TextMessage' from 'openenv_core'`
**Fix**: Use environment-specific action classes:
```python
from echo_env import EchoAction  # For echo_env
from textarena_env import TextArenaAction  # For TextArena
```

### Error: `Attribute "app" not found in module "echo_env.server"`
**Fix**: Use correct server path:
```bash
# ✓ CORRECT
python -m uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8001

# ✗ WRONG
python -m uvicorn echo_env.server:app --host 0.0.0.0 --port 8001
```

---

## Next Steps

Once echo_env works, you can:

1. **Deploy to HF Spaces** - See [HF Spaces docs](https://huggingface.co/docs/hub/spaces)
2. **Connect from Colab** - Use the same client code with your Space URL
3. **Build your own environment** - Follow the pattern in echo_env
4. **Integrate with TRL GRPO** - See [TRL OpenEnv docs](https://huggingface.co/docs/trl/en/openenv)

---

## File Locations (for reference)

After installation via uv, files are located at:
```
.venv/lib/python3.12/site-packages/echo_env/
├── __init__.py
├── server/
│   ├── __init__.py
│   ├── app.py              # ← FastAPI app
│   └── echo_environment.py # ← Environment logic
└── ... (other files)
```

To find the installation path:
```bash
python -c "import echo_env; import os; print(os.path.dirname(echo_env.__file__))"
```

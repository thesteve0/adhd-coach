# Deploying ADHD Environment to HuggingFace Spaces

**Manual Step-by-Step Guide for OpenEnv Hackathon Submission**

This guide will walk you through deploying your ADHD coaching environment to HuggingFace Spaces as required for the OpenEnv Hackathon.

---

## Prerequisites

✅ You have a HuggingFace account
✅ Local environment is working (tested via `test_environment_manual.py`)
✅ You're ready to make your code public

---

## Part 1: Prepare Your Repository for HF Spaces

### Step 1: Create Required Files for HF Spaces

HuggingFace Spaces requires specific files in your repository root. We need to create:

1. **README.md** (with YAML frontmatter)
2. **Dockerfile**
3. **pyproject.toml** (update existing)
4. **server/app.py** (entry point)

#### 1.1 Create README.md for HF Spaces

Create a file `README.md` in your project root:

```markdown
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

An OpenEnv environment that evaluates ADHD task initiation coaching quality through:

1. **Tool calling evaluation** - Rewards calling appropriate coaching tools
2. **State tracking** - Observable user state (sitting time, exercise, posture)
3. **Rubric-based scoring** - Composable evaluation criteria

## Problem Statement

Addresses **Statement 3.2: Personalized Tasks** from the OpenEnv Hackathon.

Current AI assistants give unhelpful responses to ADHD task initiation paralysis (e.g., "What would you like to work on?" when someone is stuck). This environment quantifies what makes a good ADHD coaching response.

## Innovation

### 1. Tool Calling Evaluation
The environment evaluates whether models call appropriate tools:
- **Primary tool**: `adhd_task_initiation_coach`
- **Secondary tools**: `set_timer`, `break_down_task`

### 2. State Tracking (Coming Soon)
Tracks observable user state:
- Sitting time
- Time of day
- Exercise minutes
- Posture
- Work session duration
- Cognitive load (high exec tasks completed)

### 3. Composable Rubric
Uses async scoring functions that are easy to extend.

## Quick Start

### Using the Environment from Python

```python
import requests

BASE_URL = "https://YOUR-USERNAME-adhd-env.hf.space"

# Reset environment
response = requests.post(f"{BASE_URL}/reset")
result = response.json()
print(f"Scenario: {result['observation']['scenario']}")

# Score a response
action = {
    "tool_calls": ["adhd_task_initiation_coach"],
    "message": "Open the document and type just the title."
}
response = requests.post(f"{BASE_URL}/step", json=action)
result = response.json()
print(f"Reward: {result['reward']}")
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Environment info |
| `/health` | GET | Health check |
| `/info` | GET | Detailed metadata |
| `/reset` | POST | Start new episode |
| `/step` | POST | Score coaching response |

## Current Scoring (V1)

**Tool Calling Evaluation:**
- ✅ Called `adhd_task_initiation_coach`: **1.0**
- ⚠️ Called secondary tool (`set_timer`, `break_down_task`): **0.5**
- ❌ No tool called: **0.0**

## Example Responses

### Good Response (Score: 1.0)
```json
{
  "tool_calls": ["adhd_task_initiation_coach"],
  "message": "Open your email draft and type just the subject line. That's it."
}
```

### Bad Response (Score: 0.0)
```json
{
  "tool_calls": [],
  "message": "What do you want to work on today?"
}
```

## Roadmap

- [x] V1: Tool calling evaluation
- [ ] V2: Add state tracking (6 dimensions)
- [ ] V3: Multiple scoring criteria (directive vs question, length, state-aware)
- [ ] V4: Multi-turn conversations with state evolution

## Hackathon Info

**Event**: OpenEnv Hackathon SF (March 7-8, 2025)
**Category**: Statement 3.2 - Personalized Tasks
**Innovation**: State tracking + tool calling evaluation for ADHD coaching

## Training Example

See our Colab notebook for training a model with this environment using TRL GRPO: [Link coming soon]

## Local Development

```bash
# Clone repository
git clone https://huggingface.co/spaces/YOUR-USERNAME/adhd-env

# Run locally
docker run -p 8000:8000 registry.hf.space/YOUR-USERNAME-adhd-env:latest
```

## License

MIT License - Built for OpenEnv Hackathon 2025
```

#### 1.2 Create Dockerfile

Create `Dockerfile` in your project root:

```dockerfile
# ADHD Coaching Environment Dockerfile for HuggingFace Spaces
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy project files
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install Python dependencies using uv
RUN uv pip install --system --no-cache \
    "openenv-core>=0.2.1" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "pydantic>=2.0.0"

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run server
CMD ["uvicorn", "src.environment.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 1.3 Update pyproject.toml

Update your existing `/workspaces/adhd-coach/pyproject.toml`:

```toml
[project]
name = "adhd-coach-env"
version = "0.1.0"
description = "OpenEnv environment for ADHD task initiation coaching evaluation"
requires-python = ">=3.12"
dependencies = [
    "openenv-core>=0.2.1",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.0.0",
]

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "tomli>=2.0.0",
    "tomli-w>=1.0.0",
    "ruff>=0.4.0",
]
```

---

## Part 2: Create HuggingFace Space

### Step 2: Create a New Space on HuggingFace

1. **Go to HuggingFace Spaces**: https://huggingface.co/new-space

2. **Configure your Space**:
   - **Owner**: Your username
   - **Space name**: `adhd-env` (or your preferred name)
   - **License**: MIT
   - **Select the Space SDK**: Choose **Docker**
   - **Visibility**: **Public** (required for hackathon)

3. **Click "Create Space"**

You'll be taken to your new Space repository.

---

## Part 3: Push Code to HuggingFace Space

### Step 3: Set Up Git Remote

In your local repository:

```bash
# Add HuggingFace as a remote (replace YOUR-USERNAME)
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/adhd-env

# Or if you prefer SSH
git remote add hf git@hf.co:spaces/YOUR-USERNAME/adhd-env
```

### Step 4: Authenticate with HuggingFace

You need a HuggingFace access token:

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name: `openenv-hackathon`
4. Role: **Write**
5. Click **"Generate token"**
6. **Copy the token** (you won't see it again!)

**Set up authentication:**

```bash
# Using HTTPS (you'll be prompted for username and token)
# Username: your HF username
# Password: paste your access token

# OR configure git to store credentials
git config credential.helper store
```

### Step 5: Prepare Files for Deployment

Create a clean deployment branch:

```bash
# Make sure you're on main
git checkout main

# Stage the required files
git add README.md
git add Dockerfile
git add pyproject.toml
git add src/environment/*.py
git add scripts/test_environment_manual.py

# Commit
git commit -m "Initial HuggingFace Spaces deployment"
```

### Step 6: Push to HuggingFace

```bash
# Push to HuggingFace Space
git push hf main

# You may be prompted for credentials:
# Username: YOUR-HF-USERNAME
# Password: YOUR-ACCESS-TOKEN (paste it)
```

---

## Part 4: Monitor Deployment

### Step 7: Check Build Status

1. Go to your Space: `https://huggingface.co/spaces/YOUR-USERNAME/adhd-env`

2. Click on the **"Build"** tab at the top

3. Watch the build logs:
   - ⚙️ Building Docker image...
   - 📦 Installing dependencies...
   - ✅ Build succeeded!

**Build time**: Usually 3-5 minutes

### Step 8: Test Your Deployed Environment

Once the build succeeds and the Space shows "Running":

```bash
# Test the health endpoint
curl https://YOUR-USERNAME-adhd-env.hf.space/health

# Should return: {"status": "healthy"}

# Test reset
curl -X POST https://YOUR-USERNAME-adhd-env.hf.space/reset

# Test step
curl -X POST https://YOUR-USERNAME-adhd-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "tool_calls": ["adhd_task_initiation_coach"],
    "message": "Open the doc and type the title."
  }'
```

---

## Part 5: Hackathon Submission Requirements

### Step 9: Create Demo Video (Required)

**Requirements**:
- ⏱️ 1 minute maximum
- 📹 Upload to YouTube
- 🎯 Show: Problem, Environment, Scoring in action

**Demo Script** (60 seconds):

```
[0:00-0:15] Problem
"Current AI assistants fail at ADHD task initiation. When someone says
'I'm stuck on writing an email,' generic responses like 'What would you
like to work on?' don't help."

[0:15-0:35] Solution
"Our OpenEnv environment evaluates ADHD coaching quality through tool
calling and scoring. Watch: calling the right tool scores 1.0, calling
no tool scores 0.0. The environment tracks user state like sitting time
and energy level."

[0:35-0:55] Demo
[Screen recording showing curl commands or Python script]
"Here's a good response getting score 1.0... and a bad response getting
0.0. The environment provides clear feedback to train better models."

[0:55-1:00] Impact
"This enables training models to be genuinely helpful ADHD coaches."
```

**Recording Tips**:
- Use OBS Studio (free) or QuickTime (Mac)
- Show terminal with curl commands OR
- Show Python script making API calls
- Include environment URL in video

### Step 10: Create Training Script (Required)

**You must show a minimal training script using Unsloth or HF TRL in Colab.**

For now, create a placeholder notebook or note:

```python
# training_demo.py
# Minimal training demonstration (to be completed)

from trl import GRPOTrainer, GRPOConfig
import requests

ENV_URL = "https://YOUR-USERNAME-adhd-env.hf.space"

# This will be developed on Day 2 of hackathon
# For now, verify environment is accessible

def test_env_connection():
    response = requests.get(f"{ENV_URL}/health")
    print(f"Environment health: {response.json()}")

    response = requests.post(f"{ENV_URL}/reset")
    print(f"Reset: {response.json()}")

if __name__ == "__main__":
    test_env_connection()
```

### Step 11: Submit to Hackathon

Go to submission form: https://cerebralvalley.ai/e/openenv-hackathon-sf

**Required information**:
- ✅ HuggingFace Space URL: `https://huggingface.co/spaces/YOUR-USERNAME/adhd-env`
- ✅ GitHub repository URL (if you have one)
- ✅ YouTube demo video URL
- ✅ Team members (up to 3)
- ✅ Problem statement: **3.2 - Personalized Tasks**
- ✅ Training script (Colab notebook link or .py file)

---

## Troubleshooting

### Build Fails

**Check logs** in the Build tab. Common issues:

1. **Missing dependencies**: Add to Dockerfile
2. **Port mismatch**: Ensure CMD uses port 8000
3. **Import errors**: Check PYTHONPATH in Dockerfile

### Space Shows "Runtime Error"

```bash
# Check logs in "Logs" tab
# Common fix: Restart the space
# Settings → Factory Reboot
```

### Can't Connect to Space

```bash
# Check Space is "Running" (not "Building" or "Sleeping")
# Free tier Spaces sleep after 48h of inactivity
# Visit the Space URL to wake it up
```

### Authentication Issues

```bash
# Generate new token: https://huggingface.co/settings/tokens
# Use token as password when pushing

# Or configure git credential helper
git config --global credential.helper store
```

---

## Quick Reference

### Essential Commands

```bash
# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/adhd-env

# Push to HuggingFace
git push hf main

# Test deployed environment
curl https://YOUR-USERNAME-adhd-env.hf.space/health
```

### Key Files Checklist

- [ ] `README.md` with YAML frontmatter
- [ ] `Dockerfile`
- [ ] `pyproject.toml` with dependencies
- [ ] `src/environment/*.py` (all environment files)
- [ ] Demo video uploaded to YouTube
- [ ] Training script (even if minimal)

### Important URLs

- **Your Space**: `https://huggingface.co/spaces/YOUR-USERNAME/adhd-env`
- **API Base**: `https://YOUR-USERNAME-adhd-env.hf.space`
- **Hackathon Submission**: https://cerebralvalley.ai/e/openenv-hackathon-sf
- **HF Tokens**: https://huggingface.co/settings/tokens

---

## Next Steps After Deployment

1. ✅ **Verify Space is running** - visit your Space URL
2. ✅ **Test all endpoints** - use curl or Python script
3. ✅ **Record demo video** - 1 minute showing the environment
4. ✅ **Upload to YouTube** - set as unlisted if you prefer
5. ✅ **Submit to hackathon** - include all required links
6. 📝 **Day 2**: Add state tracking, more criteria, training demo

---

## Support

- **HuggingFace Discord**: Join PyTorch Discord → #openenv channel
- **Hackathon Discord**: Message for help
- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **OpenEnv Docs**: https://meta-pytorch.org/OpenEnv/

---

**Good luck with your deployment! 🚀**

*Remember: The environment is the main deliverable. Model training is important but secondary to having a working, deployed environment.*

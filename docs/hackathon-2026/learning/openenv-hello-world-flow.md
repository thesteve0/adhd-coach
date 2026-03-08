# OpenEnv Hello World Flow - Understanding Client vs Server

This document explains how OpenEnv environments work in practice, specifically the relationship between the training client (Colab) and the environment server (HF Spaces).

---

## Where to See Source Code on HF Spaces

When you visit a HF Space (like https://huggingface.co/spaces/openenv/echo_env), click the **"Files"** tab at the top:

```
┌─────────────────────────────────────────────┐
│ openenv/echo_env                            │
│ [App] [Files] [Community] [Settings]       │  ← Click "Files"
└─────────────────────────────────────────────┘
```

The **Files tab** shows the complete source code, just like a GitHub repo:
```
echo_env/
├── README.md
├── app.py                  # FastAPI server entry point
├── server/
│   ├── app.py             # The actual server code
│   └── echo_environment.py # Environment logic
├── __init__.py
└── requirements.txt
```

**Key insight**: HF Spaces are **git repositories** - you can clone them:
```bash
git clone https://huggingface.co/spaces/openenv/echo_env
```

---

## The Request-Response Flow

### What Actually Happens During Training

```
┌──────────────────────────────────────┐
│  COLAB (Client)                      │
│  ─────────────────────                │
│  1. Model generates text response    │  ← GRPO trainer + vLLM
│     "Open a blank doc and type       │     generate completions
│      the title. That's it."          │
│                                       │
│  2. Send to environment ─────────────┼─────┐
│     step(ADHDAction(message="..."))  │     │
│                                       │     │ HTTP call
│  4. Receive reward ◄─────────────────┼─────┘
│     reward = 0.85                    │     │
│                                       │     │
│  5. Update model weights             │     │
│     using GRPO algorithm             │     │
└──────────────────────────────────────┘     │
                                             │
                                             ▼
┌──────────────────────────────────────────────┐
│  HF SPACE (Server - Your Environment)        │
│  ────────────────────────────────────        │
│  3. Evaluate the completion                  │
│     - Run reward_function(response)          │
│     - Check task decomposition: 0.30 ✓       │
│     - Check tone safety: 0.20 ✓              │
│     - Check length: 0.10 ✓                   │
│     - Total: 0.85                            │
│                                               │
│     return StepResult(reward=0.85, ...)      │
└──────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | What It Does | What It Does NOT Do |
|-----------|--------------|---------------------|
| **Colab (Client)** | - Loads and runs the model (SmolLM3-3B)<br>- Generates text completions via vLLM<br>- Calls `step(action)` with completions<br>- Receives rewards from environment<br>- Updates model weights using GRPO | - Does NOT evaluate quality<br>- Does NOT compute rewards<br>- Does NOT know what makes a "good" response |
| **HF Space (Server)** | - Receives model completions via HTTP<br>- Evaluates text quality (reward function)<br>- Returns numerical scores (0.0 - 1.0)<br>- Maintains environment state (if needed) | - Does NOT see the model<br>- Does NOT generate text<br>- Does NOT update model weights |

---

## Key Insight: Separation of Concerns

**The HF Space (server) never touches the model** - it only evaluates text and returns scores. All model training happens in Colab.

Think of it like a teacher and student:
- **Student (Colab/Model)**: Writes essays, gets feedback, learns from it
- **Teacher (HF Space/Environment)**: Reads essays, assigns grades, but doesn't write the essays

---

## Step-by-Step Example: ADHD Environment

### Step 1: Reset the Environment
```python
# In Colab
client = ADHDEnv(base_url="https://your-username-adhd-env.hf.space")
result = client.reset()

# HF Space returns initial observation:
# "User state: Energy low. Task: Write introduction. User: I'm frozen."
```

### Step 2: Model Generates Response
```python
# In Colab - GRPO trainer uses vLLM to generate
prompt = result.observation.prompt
completion = model.generate(prompt)
# completion = "You should just start with an outline..."
```

### Step 3: Send to Environment
```python
# In Colab
action = ADHDAction(message=completion)
step_result = client.step(action)
```

### Step 4: Environment Evaluates (on HF Space)
```python
# On HF Space - reward_function() runs
def reward_function(response, user_state):
    score = 0.0

    # Bad: uses "you should" (pressure language)
    if "you should" in response.lower():
        score += 0.0  # fails tone safety

    # Bad: too long (> 3 sentences)
    if sentence_count > 3:
        score += 0.0  # fails length

    # Doesn't suggest micro-action
    score += 0.0  # fails decomposition

    return 0.15  # Low score!
```

### Step 5: Colab Receives Reward
```python
# In Colab
print(step_result.reward)  # 0.15

# GRPO algorithm uses this low reward to:
# - Decrease probability of generating similar responses
# - Increase probability of better responses
```

### Step 6: Model Improves
After many episodes, the model learns to generate:
```
"Open a blank doc and type the title. That's it."
```
Which scores 0.85 because it:
- Suggests micro-action ✓
- No pressure language ✓
- Short and concise ✓

---

## Code Locations

### What's in Colab (Client Code)
```python
from your_env import ADHDEnv, ADHDAction
from trl import GRPOTrainer, GRPOConfig

# Environment client
client = ADHDEnv(base_url="https://your-username-adhd-env.hf.space")

# Custom rollout function
def rollout_func(prompts, trainer):
    outputs = generate_rollout_completions(trainer, prompts)

    for completion in outputs:
        result = client.step(ADHDAction(message=completion))
        rewards.append(result.reward)

    return {"prompt_ids": ..., "completion_ids": ..., "env_reward": rewards}

# GRPO trainer
trainer = GRPOTrainer(
    model="SmolLM3-3B",
    rollout_func=rollout_func,
    reward_funcs=[extract_env_reward],
    ...
)
trainer.train()
```

### What's in HF Space (Server Code)
```python
# server/app.py
from fastapi import FastAPI
from openenv.core import StepResult

app = FastAPI()

@app.post("/step")
def step(action: ADHDAction):
    # Evaluate the response
    reward = reward_function(action.message, user_state)
    observation = ADHDObservation(...)

    return StepResult(observation=observation, reward=reward, done=False)

def reward_function(response: str, user_state: dict) -> float:
    score = 0.0
    # Your scoring logic here
    return score
```

---

## Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "The environment trains the model" | The environment only provides scores. GRPO in Colab does the training. |
| "The model runs on HF Spaces" | The model runs in Colab. HF Spaces only sees text strings. |
| "HF Spaces needs a GPU" | No! Environment is just Python logic (reward functions, state management). CPU is fine. |
| "The client alters the model" | The client **uses** the model to generate text, then uses rewards to **train** it. |

---

## Why This Architecture?

### Benefits of Client-Server Separation

1. **Scalability**: Multiple training runs can share one environment server
2. **Modularity**: Change reward function without touching training code
3. **Simplicity**: Environment is just Python functions, no ML dependencies
4. **Shareability**: Anyone can test your environment without downloading the model

### Example: Multiple Researchers

```
┌─────────────┐
│ Researcher A│───┐
│ Colab       │   │
└─────────────┘   │
                  ├──► https://your-adhd-env.hf.space
┌─────────────┐   │    (Single environment server)
│ Researcher B│───┤
│ Colab       │   │
└─────────────┘   │
                  │
┌─────────────┐   │
│ Researcher C│───┘
│ Local GPU   │
└─────────────┘
```

All three can train different models using the same reward logic!

---

## Next Steps

Now that you understand the flow:

1. **Test locally**: Run uvicorn to validate reward function logic
2. **Deploy to HF Spaces**: Push your environment code
3. **Connect from Colab**: Point your GRPO trainer to your Space URL
4. **Watch it learn**: Monitor reward curves as the model improves

The server (HF Space) is the judge. The client (Colab) is the student. The model learns by trying to impress the judge!

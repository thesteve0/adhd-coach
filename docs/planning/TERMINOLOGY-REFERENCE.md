# HF TRL + OpenEnv Terminology Reference

**Print this page for quick reference during the hackathon**

---

## Core Terminology Mapping

| HF TRL/OpenEnv Term | What It Really Means | Your Mental Model |
|---------------------|----------------------|-------------------|
| **Rollout** | Generate N model completions for scoring | "Batch generation run" - NOT Kubernetes rollout! |
| **Step** | Send 1 completion to environment, get 1 reward back | "Score this response" - single request/response |
| **Reset** | Start a new environment episode (get initial prompt) | "Give me a fresh scenario to work with" |
| **Episode** | Complete interaction sequence from reset() to done=True | "One full conversation/game/task from start to finish" |
| **Observation** | What the environment tells the model (prompt, context, state) | "Here's the situation you need to respond to" |
| **Action** | What the model sends to the environment (text response) | "Here's my response to your situation" |
| **Reward** | Score from environment (0.0-1.0) for that action | "How good was that response?" |
| **Dataset** | Collection of text prompts (one per item) | "Training scenarios the model learns from" |

---

## Dataset Explained

### What Is a Dataset?

**Dataset = Collection of prompts (text strings)**

Each item in the dataset is **ONE prompt** that the model will respond to during training.

### Simple Example (Echo environment)

```python
dataset = Dataset.from_dict({
    "prompt": [
        "You are an AI that interacts with an *Echo* environment. Word to echo:",
        "You are an AI that interacts with an *Echo* environment. Word to echo:",
        # ... repeated 64 times
    ]
})
```

- **1 dataset item** = 1 text prompt
- During training, GRPO samples a batch of prompts (e.g., 8 prompts)
- For each prompt, generates `num_generations` completions (e.g., 4-12)
- Result: 8 prompts × 12 generations = 96 completions scored per batch

### ADHD Coaching Example

```python
dataset = Dataset.from_dict({
    "prompt": [
        "You are an ADHD coach. User says: I can't start this task. I'm frozen.",
        "You are an ADHD coach. User says: I keep getting distracted and can't focus.",
        "You are an ADHD coach. User says: I don't know where to begin with this project.",
        # ... more scenarios (20-50 for variety)
    ]
})
```

**Why variety matters**:
- Prevents overfitting to one scenario
- Model learns general patterns, not memorization
- Better demos (show it works on different situations)

**For MVP**: Start with 1 repeated prompt to validate the pipeline works!

```python
dataset = Dataset.from_dict({
    "prompt": ["User: I'm stuck on this task."] * 64
})
```

---

## Single-turn vs Multi-turn Datasets

### Single-turn (simpler, your ADHD project)

**Dataset item**: The complete context the model needs

```python
"You are an ADHD coach. User (low energy, stuck 45 min): I can't write this intro."
```

**Training flow**:
1. Model gets this prompt
2. Model generates N responses (e.g., 4 different ways to respond)
3. Each response gets scored independently
4. GRPO updates model to favor higher-scoring responses

**Key**: Environment `reset()` returns this prompt. Each `step()` scores a different response to the SAME prompt.

---

### Multi-turn (more complex, like Wordle)

**Dataset item**: Just the initial instruction

```python
"You are playing Wordle. Guess the 5-letter word."
```

**Training flow**:
1. Environment `reset()` starts a new game
2. Model generates guess #1 → `step()` → feedback (e.g., "G=green, Y=yellow, X=blank")
3. Model uses feedback → generates guess #2 → `step()` → more feedback
4. ... continues for up to 6 guesses
5. Episode ends when word is guessed or 6 attempts used

**Key**: The prompt is just the starting point. The conversation builds through multiple `step()` calls within one episode.

---

## How Dataset Items Are Used During Training

```python
# During training, GRPO does this:

for batch_of_prompts in dataset:  # e.g., 8 prompts per batch
    # batch_of_prompts = ["prompt1", "prompt2", ..., "prompt8"]

    rollout_results = rollout_func(batch_of_prompts, trainer)
    # For EACH prompt, generates num_generations completions
    # e.g., 8 prompts × 12 generations = 96 completions total

    # Update model weights using all 96 completion rewards
```

---

## The Complete Flow (ADHD Coaching Example)

```
┌─────────────────────────────────────────────────────────────────┐
│  ROLLOUT (happens in Colab)                                     │
│  "Generate N completions and score them"                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. RESET → Get initial scenario                                │
│     env.reset()                                                  │
│     Returns: "User is frozen on task: Write introduction"       │
│                                                                  │
│  2. GENERATE N COMPLETIONS (using vLLM)                          │
│     Model generates 4 different responses:                       │
│       A: "You should just start..."                             │
│       B: "Open a blank doc..."                                  │
│       C: "Let me break this down..."                            │
│       D: "What specifically are you stuck on?"                  │
│                                                                  │
│  3. STEP EACH COMPLETION (send to HF Space)                     │
│     for completion in [A, B, C, D]:                             │
│         result = env.step(Action(message=completion))           │
│         rewards.append(result.reward)                           │
│                                                                  │
│     Returns: [0.15, 0.85, 0.40, 0.75]                           │
│                                                                  │
│  4. GRPO USES REWARDS TO UPDATE MODEL                            │
│     Mean reward = 0.54                                           │
│     Advantage B = +0.31 → Increase probability of B's tokens    │
│     Advantage A = -0.39 → Decrease probability of A's tokens    │
│                                                                  │
│  5. NEXT ROLLOUT (repeat with improved model)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## When Reset Happens

| Context | When Reset Is Called | Why |
|---------|---------------------|-----|
| **Single-turn (ADHD coaching)** | Once per rollout | Get a fresh scenario for each batch of completions |
| **Multi-turn (Wordle game)** | Once per episode | Start a new game (6 guesses = 1 episode) |
| **During training** | Every rollout (or at episode completion) | Prevents model from seeing same scenario repeatedly |

**Key insight**: Reset gives you a NEW scenario. Step processes ONE response to that scenario.

---

## Rollout Function Parameters Explained

```python
def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
```

### Parameter: `prompts: list[str]`

**What it is**: Text prompts from your dataset that the model should respond to

**Example**: `["You are an ADHD coach. User says: I'm frozen on task."]`

**Why you need it**: This is the INPUT to your model - what it responds to

**How it's used**: You pass this to `generate_rollout_completions()` which feeds it to the model

---

### Parameter: `trainer: GRPOTrainer`

**What it is**: The active GRPOTrainer object running your training

**Why you need it**: Gives you access to:
- `trainer.processing_class` → The tokenizer (decode tokens to text)
- `trainer.config` → Training config (max_new_tokens, num_generations, etc.)
- Generation utilities → `generate_rollout_completions(trainer, prompts)`

**How it's used**:
```python
# Generate completions using trainer's model + config
outputs = generate_rollout_completions(trainer, prompts)

# Access tokenizer to decode tokens
tokenizer = trainer.processing_class
text = tokenizer.decode(output["completion_ids"])
```

---

## Rollout Function Return Values

```python
return {
    "prompt_ids": [...],      # REQUIRED - token IDs of the prompt
    "completion_ids": [...],  # REQUIRED - token IDs of model's response
    "logprobs": [...],        # REQUIRED - log probabilities of each token
    "env_reward": [0.85],     # CUSTOM - your reward from environment
}
```

**Key insight**: Anything beyond the 3 required fields gets passed to your reward functions as `**kwargs`

---

## Step vs Rollout Boundaries

| Concept | Boundary | What Happens Inside |
|---------|----------|---------------------|
| **Step** | 1 completion → 1 reward | `env.step(action)` sends text, gets back reward |
| **Rollout** | N steps (1 per completion) | Generate N completions → step each one → collect N rewards |
| **Episode** | reset() to done=True | Could be 1 step (single-turn) or many steps (multi-turn game) |

### Example: Single-turn ADHD Coaching

```python
# 1 ROLLOUT = 1 EPISODE = 4 STEPS
result = env.reset()  # Get scenario
for completion in [A, B, C, D]:  # 4 steps
    step_result = env.step(Action(message=completion))
    # Each step is independent - same scenario, different response
```

### Example: Multi-turn Wordle

```python
# 1 ROLLOUT = 1 EPISODE = 6 STEPS (max)
result = env.reset()  # Start new game
for turn in range(6):  # Up to 6 steps
    guess = model.generate()
    result = env.step(Action(message=guess))
    if result.done:  # Won or lost
        break  # Episode ends
    # Each step builds on previous (multi-turn conversation)
```

---

## Quick Reference: What Lives Where

| Component | Lives In | Does What |
|-----------|----------|-----------|
| **Model (SmolLM3-3B)** | Colab (vLLM) | Generates text completions |
| **GRPO Trainer** | Colab | Updates model weights using rewards |
| **Rollout Function** | Colab (your code) | Orchestrates: generate → step → collect rewards |
| **Environment** | HF Space (FastAPI server) | Scores completions, returns rewards |
| **Reward Function** | HF Space (your code) | Implements scoring logic (0.0-1.0) |
| **Dataset** | Colab (loaded into trainer) | Provides prompts for training |

---

## Common Confusions Clarified

### "Do I call step() for each token?"
**NO.** Step takes the COMPLETE text response and returns 1 reward.

### "Does reset() happen between every step?"
**NO.** Reset happens at the START of an episode. Then you step multiple times within that episode.

### "Is rollout the same as an episode?"
**USUALLY YES for single-turn tasks.** For multi-turn (like Wordle), 1 episode = many steps, 1 rollout = 1 episode.

### "Why do I need to return prompt_ids if I already have prompts?"
**GRPO needs token IDs to calculate loss.** The trainer needs to know which tokens to update and their log probabilities.

### "Can my dataset have different prompt formats?"
**YES.** Each item can be any text. The model just generates completions for whatever prompt you give it.

---

## Minimal ADHD Coaching Rollout (Annotated)

```python
def rollout_func(prompts: list[str], trainer: GRPOTrainer):
    # GENERATE: Model creates N completions
    outputs = generate_rollout_completions(trainer, prompts)

    # DECODE: Convert tokens to text
    tokenizer = trainer.processing_class
    completions_text = [
        tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
        for out in outputs
    ]

    # RESET: Get fresh scenario from environment
    client.reset()

    # STEP: Score each completion
    env_rewards = []
    for completion in completions_text:
        result = client.step(ADHDAction(message=completion))
        env_rewards.append(result.reward)

    # RETURN: Required fields + custom reward
    return {
        "prompt_ids": [out["prompt_ids"] for out in outputs],
        "completion_ids": [out["completion_ids"] for out in outputs],
        "logprobs": [out["logprobs"] for out in outputs],
        "env_reward": env_rewards,  # Custom field passed to reward_func
    }
```

---

## Print-Friendly Summary

**DATASET** = "Collection of text prompts (one per item) that model learns to respond to"

**ROLLOUT** = "Generate a batch of N responses and score them all"

**STEP** = "Score this one response" (HTTP call to HF Space)

**RESET** = "Give me a new scenario to work with"

**EPISODE** = "One complete interaction from start to finish"

**Flow**: Reset (new scenario) → Generate N responses → Step each one (get N rewards) → GRPO updates model → Repeat

---

## For Your ADHD Project

**Recommended starter dataset**:
```python
# MVP (validate pipeline)
dataset = Dataset.from_dict({
    "prompt": ["User: I'm stuck on this task."] * 64
})

# Production (after MVP works)
dataset = Dataset.from_dict({
    "prompt": [
        "User (low energy, stuck 45min): I can't start writing this introduction.",
        "User (moderate energy, procrastinating): I keep avoiding this email.",
        "User (high anxiety): This project feels overwhelming.",
        "User (distracted): I lose focus every 5 minutes.",
        # Add 20-50 different scenarios for variety
    ]
})
```

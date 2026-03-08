# The Basics of Reinforcement Learning and GRPO

**Purpose**: Understand how RL differs from traditional fine-tuning, how models learn from rewards, and why GRPO is well-suited for LLM alignment tasks.

---

## Key Insight: RL vs Traditional Fine-Tuning

### Traditional Supervised Fine-Tuning (SFT)

```
Batch of pre-labeled data:
  Prompt: "User is frozen on task"
  Label: "Open a blank doc and type the title."

Loss = How different is model output from the label?
Update weights to minimize difference from labels.
```

**Process:**
1. Human expert writes the "correct" response
2. Model tries to generate that response
3. Measure how different model output is from the label
4. Adjust weights to make model output closer to the label

### Reinforcement Learning (RL)

```
Model generates its own "training data":
  Prompt: "User is frozen on task"
  Model generates: "You should just start..."
  Environment scores: 0.15

Loss = How different is this score from what we want?
Update weights to maximize expected reward.
```

**Process:**
1. Model generates its own responses
2. Environment scores those responses (0.0 to 1.0)
3. Compare scores to find better vs worse responses
4. Adjust weights to make better responses more likely

### The Fundamental Difference

**RL doesn't need pre-defined "correct answers"** - it learns from scores on self-generated responses.

In essence, we are **deriving training data on the fly**:
- The model generates responses to prompts
- The environment (teacher) scores those responses
- We use those generated responses and their scores to drive gradient descent
- The model learns which patterns tend to get higher scores

This is true for **all RL algorithms**, not just GRPO. The algorithms differ in **how** they use those scores to update the model.

---

## How the Model Gets Updated in GRPO

### The Complete GRPO Training Loop

```
┌─────────────────────────────────────────────────────────────┐
│  COLAB: Single Training Step                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. MODEL GENERATES (vLLM inference)                        │
│     Prompt: "User is frozen on task: Write introduction"   │
│                                                              │
│     Model generates N completions (e.g., N=4):             │
│     A: "You should just start with an outline..."          │
│     B: "Open a blank doc and type the title."              │
│     C: "Let me help you break this down into steps..."     │
│     D: "Just write one sentence about your topic."         │
│                                                              │
│  2. SEND TO ENVIRONMENT                                     │
│     → Send A, B, C, D to HF Space                          │
│                                                              │
│  3. RECEIVE REWARDS (from environment)                      │
│     ← Reward A: 0.15 (bad: pressure language, too long)    │
│     ← Reward B: 0.85 (good: micro-action, concise)         │
│     ← Reward C: 0.40 (okay: decomposes but too complex)    │
│     ← Reward D: 0.65 (good: simple but less specific)      │
│                                                              │
│  4. COMPUTE ADVANTAGES (GRPO algorithm)                     │
│     Mean reward = (0.15 + 0.85 + 0.40 + 0.65) / 4 = 0.51   │
│                                                              │
│     Advantage A = 0.15 - 0.51 = -0.36  (BAD)               │
│     Advantage B = 0.85 - 0.51 = +0.34  (GOOD!)             │
│     Advantage C = 0.40 - 0.51 = -0.11  (MEH)               │
│     Advantage D = 0.65 - 0.51 = +0.14  (OKAY)              │
│                                                              │
│  5. UPDATE MODEL WEIGHTS (backpropagation)                  │
│     For each token in completion B:                         │
│       → INCREASE probability of generating these tokens    │
│     For each token in completion A:                         │
│       → DECREASE probability of generating these tokens    │
│                                                              │
│     The model's transformer weights are adjusted via        │
│     gradient descent to maximize expected reward.           │
│                                                              │
│  6. NEXT BATCH                                              │
│     Model is now slightly better → generates new rollouts   │
│     → Repeat steps 1-5 thousands of times                   │
└─────────────────────────────────────────────────────────────┘
```

### Key GRPO Concepts

| Concept | Explanation |
|---------|-------------|
| **Relative comparison** | GRPO doesn't care that 0.85 is "good" - it cares that 0.85 is **better than the other options in this batch** |
| **Group mean baseline** | The average reward of the group serves as the baseline - no separate value network needed |
| **Advantage** | How much better/worse a response is compared to the group average |
| **Policy gradient** | The model learns by trying different "actions" (text completions) and reinforcing the ones that work |
| **No explicit labels** | Unlike supervised learning ("this is the correct answer"), RL learns from **degrees of success** |
| **Exploration** | Early in training, the model tries diverse responses. Later, it converges on high-reward patterns |

---

## What Actually Changes in the Model

The model is a neural network with billions of parameters (weights). The GRPO algorithm:

1. **Computes a loss function** based on:
   - How much reward each completion got
   - How different the new model is from the reference model (KL divergence penalty - prevents the model from changing too drastically)

2. **Calculates gradients** - for each weight in the network, compute: "If I increase this weight slightly, will the model generate more high-reward completions?"

3. **Updates weights via backpropagation**:
   ```python
   # Simplified conceptual code
   for weight in model.parameters():
       # If this weight contributed to high-reward completion B:
       weight.data += learning_rate * positive_gradient

       # If this weight contributed to low-reward completion A:
       weight.data -= learning_rate * negative_gradient
   ```

### Concrete Example: Token Probability Changes

**Before training (Episode 1):**
```python
Prompt: "User is frozen on task"
Token probabilities for next word:

P("You") = 0.25      ← High probability (common start)
P("Open") = 0.08     ← Low probability
P("Let") = 0.15
P("Just") = 0.12
```

**After 100 episodes of training where "Open..." got high rewards:**
```python
Prompt: "User is frozen on task"
Token probabilities for next word:

P("You") = 0.12      ← Decreased (led to low rewards)
P("Open") = 0.35     ← Increased! (led to high rewards)
P("Let") = 0.10      ← Slightly decreased
P("Just") = 0.05     ← Decreased (pressure language got penalized)
```

The model has **learned** that starting with "Open" tends to get better rewards than starting with "You should..."

### Why This Works for ADHD Coaching

After thousands of training steps:
- The model has seen that "Open a blank doc..." gets rewards of ~0.80-0.90
- The model has seen that "You should just start..." gets rewards of ~0.10-0.20
- The weights have been adjusted so the first pattern is **much more likely** to be generated

**Important**: The model doesn't "understand" ADHD - it has just learned the statistical pattern: *these token sequences tend to produce higher numbers from that HTTP endpoint*.

---

## Comparing RL Algorithms

There are many RL algorithms. They all share the same fundamental approach (model generates data, environment scores it, update weights), but differ in **how** they compute updates.

### 1. REINFORCE (Vanilla Policy Gradient) - The Simplest

```
For each prompt:
  1. Generate 1 completion
  2. Get reward from environment
  3. Update: gradient ∝ reward × log_prob(tokens)

Problem: VERY high variance
  - If you get lucky once (reward=0.9), huge weight update
  - If you get unlucky (reward=0.1), huge opposite update
  - Model thrashes around, unstable training
```

**Example:**
```
Step 1: Generate "Open a doc" → reward 0.85 → BIG positive update
Step 2: Generate "You should..." → reward 0.15 → BIG negative update
Step 3: Generate "Open a doc" again → reward 0.80 → BIG positive update

The model is being yanked around by single samples!
```

**Why this is problematic:**
- The model doesn't know if 0.15 is good or bad without context
- Is 0.15 good (compared to what it usually gets)?
- Is 0.15 bad (compared to optimal)?
- Or is it just noise?
- High variance = unstable, slow training

### 2. PPO (Proximal Policy Optimization) - Industry Standard

```
For each prompt:
  1. Generate N completions
  2. Get rewards from environment
  3. Train a VALUE NETWORK to predict expected reward
  4. Advantage = actual_reward - value_network_prediction
  5. Update policy, but clip changes to prevent large jumps

Used in: ChatGPT, GPT-4 RLHF, Claude RLHF
```

**Better than REINFORCE because:**
- Value network learns a baseline (reduces variance)
- Clipping prevents destructive updates
- More stable training

**Complexity cost:**
- You're training TWO models: policy (LLM) + value network
- More hyperparameters to tune
- More memory/compute overhead
- More code to debug

### 3. GRPO (Group Relative Policy Optimization) - Newer, Simpler

```
For each prompt:
  1. Generate N completions (a "group")
  2. Get rewards: [0.15, 0.85, 0.40, 0.65]
  3. Baseline = mean(group_rewards) = 0.51
  4. Advantage = reward - baseline
  5. Update based on advantage

NO value network needed!
```

**Example with ADHD environment:**
```
Prompt: "User is frozen on task"

Completions generated:
  A: "You should start..." → reward 0.15
  B: "Open a blank doc..." → reward 0.85
  C: "Let me break this down..." → reward 0.40
  D: "Just write one sentence..." → reward 0.65

Group mean = 0.51

Advantages:
  A: 0.15 - 0.51 = -0.36  (much worse than average)
  B: 0.85 - 0.51 = +0.34  (much better than average!)
  C: 0.40 - 0.51 = -0.11  (slightly worse)
  D: 0.65 - 0.51 = +0.14  (slightly better)

Update:
  - Increase probability of B's tokens (best in group)
  - Decrease probability of A's tokens (worst in group)
  - Small changes to C and D
```

---

## Algorithm Comparison Table

| Algorithm | Complexity | Stability | Memory | Works with Small Models? | Code Complexity |
|-----------|------------|-----------|--------|--------------------------|-----------------|
| **REINFORCE** | Low (1 network) | Poor (high variance) | Low | No | Simple |
| **PPO** | High (2 networks) | Excellent | High | Yes, but overhead | Complex |
| **GRPO** | Low (1 network) | Good | Low | Yes! | Moderate |

---

## Why GRPO is Good for LLM Alignment

### Key Benefits of GRPO

1. **Relative comparison**: The model doesn't need to know "0.85 is good" - it just learns "this was better than the alternatives I tried"

2. **Automatic baseline**: The group mean is a natural baseline, no separate value network needed

3. **Stable updates**: Even if all responses are bad (all scores < 0.3), the best one still gets reinforced relative to the worst

4. **Simpler code**: Easier to implement and debug at a hackathon

5. **Less memory**: No value network = fewer parameters to store/train

6. **Sample efficiency**: Learns from relative comparisons (more signal per sample)

### Concrete Example: Why Group Comparison Helps

**Scenario 1: All responses are terrible (early in training)**
```
Rewards: [0.10, 0.15, 0.12, 0.13]
Mean: 0.125

Advantage of 0.15: +0.025 → Still gets reinforced!
Advantage of 0.10: -0.025 → Still gets penalized!

The model learns: "Even though all my responses suck,
response B was the LEAST bad, so do more of that."
```

**Scenario 2: All responses are great (late in training)**
```
Rewards: [0.90, 0.95, 0.88, 0.92]
Mean: 0.9125

Advantage of 0.95: +0.0375 → Gets reinforced
Advantage of 0.88: -0.0325 → Gets slightly penalized

The model learns: "Even though I'm doing well,
I should slightly prefer the 0.95 pattern."
```

**Comparison with REINFORCE (single response):**
```
Generate "You should..." → reward 0.15 → Update

The model doesn't know if 0.15 is:
  - Good (compared to what it usually gets)?
  - Bad (compared to optimal)?
  - Just noise?

No baseline = high variance = unstable training.
```

---

## Why GRPO for Your Hackathon

GRPO is specifically designed for **LLM alignment** use cases, making it perfect for the OpenEnv hackathon:

### Technical Reasons

1. **Sample efficiency**: Learns from relative comparisons (more signal per sample)
2. **Simplicity**: No value network (easier to implement and debug!)
3. **Stability**: Group mean baseline reduces variance
4. **Small model friendly**: Works well with SmolLM3-3B (1.7B-3B params)
5. **Modest compute**: Runs in Colab with T4 GPU

### Practical Reasons for Hackathon

1. **Focus on what matters**: You can spend time on the **reward function** (the interesting, creative part!) instead of debugging training instability
2. **Less code to write**: One model instead of two
3. **Easier to debug**: Fewer moving parts, clearer failure modes
4. **Faster iterations**: Less memory overhead = faster training loops
5. **Well-supported**: HuggingFace TRL has excellent GRPO integration with OpenEnv

### What You Should Focus On

With GRPO handling the training stability, you can focus on:
- **Reward function design**: What makes a good ADHD coaching response?
- **Environment scenarios**: What situations should the model learn from?
- **Evaluation**: How do you measure if the model is actually improving?

The algorithm is doing the hard work of stable learning - you just need to teach it what "good" looks like!

---

## Further Reading

- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **OpenAI Spinning Up**: [Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- **HuggingFace TRL**: [GRPO + OpenEnv Integration](https://huggingface.co/docs/trl/en/openenv)

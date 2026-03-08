# OpenEnv Hackathon SF — Project Planning Notes
**Event**: March 7-8, Shack15, San Francisco  
**Format**: Solo (seeking teammate on PyTorch Discord)  
**Submission deadline**: Sunday March 8, 1:00 PM

---

## Project Concept: ADHD Executive Function Scaffolding Environment

### Problem Statement
**Statement 3.2 — Personalized Tasks**: Build an environment that trains an LLM to scaffold task initiation for people with ADHD.

### Core Idea
Train a small LLM to be a better ADHD task initiation coach via RL. The environment simulates an ADHD user with a daunting task; the model learns to generate responses that are measurably better at reducing initiation paralysis.

### The Gym Analogy
- The **environment** is the gym
- The **LLM** is the athlete being trained
- The **reward function** scores response quality
- The **deliverable** is before/after evidence that training improved the model

### Longer-Term Vision: Reachy Mini Integration
The trained model weights can eventually be deployed on a Reachy Mini desktop robot (Pollen Robotics / HuggingFace, ~$299), which has:
- Camera, microphone, speakers
- Expressive head/antenna movement
- Python SDK, HuggingFace-native
- Physical presence (body doubling research supports this for ADHD)

**The pitch story**: "We trained a model to be an ADHD coach, and here's the physical robot it will run on."

### What the Hackathon Builds vs. What Comes Later
| Hackathon scope | Post-hackathon |
|---|---|
| OpenEnv environment with `reset()`, `step()`, reward function | Deploy model to Reachy Mini |
| GRPO training loop in Colab/HF Spaces | Camera/posture/break detection features |
| Before/after reward curves | Full conversational app |
| 1-min YouTube demo video | |

**Explicitly out of scope for hackathon**: posture detection, hydration tracking, break timing via CV — these are a separate project.

---

## Reward Function Rubric

The reward function is the heart of the environment. It takes the model's text response and returns a score between 0.0 and 1.0. Each criterion is scored independently then summed and normalized.

### Example Scenario Fed to the Model
```
User state: Energy level: low. Has been staring at task for 45 minutes. 
Task: "Write the introduction section of my conference talk."
User message: "I just can't start. I know what I need to do but I'm frozen."
```

### Scoring Criteria

| Criterion | Max points | What earns full points | What earns zero |
|---|---|---|---|
| Task decomposition | 0.30 | Breaks task into ≥1 concrete micro-step small enough to start in 2 min | Restates the full task, gives a plan with 5+ steps |
| Single focus | 0.20 | Gives the user exactly ONE thing to do or ask | Asks multiple questions, offers multiple options |
| Energy matching | 0.20 | Acknowledges low energy; suggests the smallest possible action | Suggests high-effort first step; ignores stated state |
| Tone safety | 0.20 | Warm, non-judgmental, no urgency pressure | Uses "you should", "just", "it's easy", guilt language |
| Length appropriateness | 0.10 | Response is ≤ 3 sentences (suitable for voice delivery) | Wall of text, bullet lists, lengthy explanation |

**Total: 1.0**

### Example Scoring in Python

```python
def reward_function(response: str, user_state: dict) -> float:
    score = 0.0

    # Task decomposition (0.30)
    # Check: does it suggest a single concrete action under ~2 minutes?
    micro_action_phrases = ["open", "write one", "just type", "start with", "first word"]
    if any(phrase in response.lower() for phrase in micro_action_phrases):
        score += 0.30
    elif "step" in response.lower():
        score += 0.15  # partial credit for decomposition attempt

    # Single focus (0.20)
    # Penalize responses with multiple questions or options
    question_count = response.count("?")
    if question_count == 0:
        score += 0.20  # statement, not interrogation
    elif question_count == 1:
        score += 0.15  # one question is acceptable
    # 2+ questions = 0

    # Energy matching (0.20)
    # If user_state indicates low energy, reward acknowledging it
    if user_state.get("energy") == "low":
        low_energy_phrases = ["small", "tiny", "just one", "no pressure", "easy"]
        if any(phrase in response.lower() for phrase in low_energy_phrases):
            score += 0.20
        else:
            score += 0.05  # partial: didn't hurt but didn't help
    else:
        score += 0.20  # not low energy, energy matching less critical

    # Tone safety (0.20)
    # Penalize shame/pressure language
    bad_phrases = ["you should", "you need to", "just do it", "it's simple",
                   "it's easy", "just start", "stop procrastinating"]
    if not any(phrase in response.lower() for phrase in bad_phrases):
        score += 0.20

    # Length (0.10)
    sentence_count = response.count(".") + response.count("!") + response.count("?")
    if sentence_count <= 3:
        score += 0.10
    elif sentence_count <= 5:
        score += 0.05

    return score


# Example usage in OpenEnv step()
def step(self, action):
    response = action.message
    reward = reward_function(response, self.current_user_state)
    observation = self._get_next_observation(response)
    done = self._check_episode_done()
    return StepResult(observation=observation, reward=reward, done=done)
```

### What "Getting Better" Looks Like
Early in training the model might respond:
> "You should really just start with an outline. It's not that hard — just write down your main points and go from there. You've done this before!"

Score: ~0.15 (fails tone safety, too long, not decomposed, pressure language)

After training it should respond more like:
> "Open a blank doc and type just the title of your talk. That's it for now."

Score: ~0.85 (micro-action, single focus, low-effort, no pressure, short)

### Notes for Hackathon
- The phrase-matching approach above is intentionally simple — good enough to show a reward signal
- For a stronger version, use an LLM-as-judge scoring the response against criteria (but adds latency)
- You only need to show the reward curve is going UP — perfect scoring is not the goal
- Edit the `bad_phrases` and `micro_action_phrases` lists based on what you observe the model doing wrong in early rollouts

---

## Technical Architecture

### Environment Structure
```
reset() → presents scenario: "User hasn't started X task, says they feel frozen"
step(action) → model generates scaffolding response
reward() → scores response on rubric above
```

### Training Stack
- **Framework**: OpenEnv 0.2.1 + HF TRL GRPOTrainer
- **Model**: SmolLM3-3B (primary), Qwen 2.5 0.5B-1.5B (fallback if SmolLM proves difficult)
- **Training location**: Google Colab (T4 GPU) or HF Spaces
- **Local dev**: ROCm devcontainer (Strix Halo) — no GPU needed for env logic

### Deployment Architecture (Hackathon)

```
┌─────────────────────┐         ┌──────────────────────┐         ┌────────────────┐
│  HF Spaces          │         │  Google Colab        │         │  HuggingFace   │
│                     │         │                      │         │  Hub           │
│  ADHD Environment   │◄────────│  GRPO Trainer        │◄────────│  SmolLM3-3B    │
│  (FastAPI/uvicorn   │  HTTP   │  + vLLM              │  load   │  (model)       │
│   auto-runs)        │  calls  │                      │  model  │                │
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

**Workflow:**
1. **Local dev** (devcontainer): Test environment logic with `python -m uvicorn your_env.server.app:app`
2. **Deploy to HF Spaces**: Push environment code → HF Spaces runs uvicorn automatically
3. **Training in Colab**: GRPO trainer connects to HF Space URL, queries environment for rewards
4. **Submission**: Both environment (HF Space) and trained model (HF Hub) are public

### Submission Requirements Checklist
- [ ] OpenEnv 0.2.1 deployed on HF Spaces
- [ ] Public GitHub repo (new, no prior work)
- [ ] Minimal training script in Colab using Unsloth or HF TRL
- [ ] 1-minute YouTube demo video
- [ ] Show reward improvement curves

---

## Dev Environment Notes

### ROCm Devcontainer (local)
- Stack: ROCm 7.2 / PyTorch 2.9.1 / Python 3.12 / Ubuntu 24.04
- Repo: https://github.com/thesteve0/datascience-template-ROCm
- FP16 only (officially validated on Strix Halo gfx1151)
- **BF16 untested** — TRL GRPO defaults to BF16, watch for this locally

### Installing OpenEnv Locally
```bash
# Install echo_env for testing (includes openenv-core 0.2.1)
uv add "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"

# For actual project, install additional dependencies
uv add trl transformers accelerate datasets
```
- `openenv_core` is pure Python (FastAPI/Pydantic/uvicorn) — no GPU dependency
- ROCm package protection in `pyproject.toml` handles transitive torch deps
- **Import name**: `from echo_env import EchoEnv, EchoAction` (NOT openenv_echo or openenv_echo_env)

### Testing Echo Environment Locally

**Quick test with remote HF Space** (no server needed):
```python
from echo_env import EchoEnv, EchoAction

client = EchoEnv(base_url="https://openenv-echo-env.hf.space")
result = client.reset()
print(result.observation)  # EchoObservation with .echoed_message, .message_length

step_result = client.step(EchoAction(message="Hello!"))
print(f"Reward: {step_result.reward}, Done: {step_result.done}")
```

**Run local server for testing deployment**:
```bash
# Terminal 1: Start server (correct path is echo_env.server.app:app)
python -m uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8001

# Terminal 2: Connect client
# from echo_env import EchoEnv
# client = EchoEnv(base_url="http://0.0.0.0:8001")
```

**Key API facts**:
- `reset()` → returns `StepResult(observation=..., reward=float, done=bool)`
- `step(Action)` → returns `StepResult` with same structure
- Each environment defines its own Action class (e.g., `EchoAction`, `TextArenaAction`)
- Access results via `.observation`, `.reward`, `.done` attributes

### vLLM Note
- PyPI vLLM = CUDA only; ROCm build requires AMD-specific wheel or source compile
- **Workaround for local testing**: set `use_vllm=False` in GRPOConfig (slower but works)
- Actual training runs use Colab (CUDA) anyway — not a blocker

### Local Dev Workflow
1. Develop and test environment logic (`reset()`, `step()`, reward) — CPU only, no GPU needed
2. Validate a forward pass through chosen model in FP16 locally
3. Push to Colab for actual GRPO training runs

---

## Learning Curriculum (Friday evening → Saturday 11:30 AM)
- [ ] Get OpenEnv echo_env running locally
- [ ] Read TRL GRPO + OpenEnv integration docs
- [ ] Run blackjack example end-to-end in Colab
- [ ] Sketch first version of ADHD environment `reset()` and `step()`
- [ ] Draft reward function rubric

---

## Resources
- OpenEnv repo: https://github.com/meta-pytorch/OpenEnv
- TRL OpenEnv integration: https://huggingface.co/docs/trl/main/en/openenv
- Unsloth GRPO notebooks: https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks
- Reachy Mini: https://huggingface.co/spaces/pollen-robotics/Reachy_Mini
- HF submission form: https://cerebralvalley.ai/e/openenv-hack…

---

## Sponsors & Tracks

### Sponsors
- **Meta/PyTorch** — Framework creators, OpenEnv maintainers
- **Hugging Face** — Hub hosting, TRL integration, Spaces deployment
- **Unsloth** — Training optimization (potential stretch goal)
- **Mercor** — Sponsor track focused on agent training via RL

### Prize Opportunities
- **Total prize pool**: $100K+
- **OpenEnv Challenge**: $10K HF credits + PyTorch.org blog feature
- **Mercor Track**: Focused on training agents through RL — potential separate prize category

---

## Stretch Goals

Based on judging weights (40% Environment Innovation + 30% Storytelling), prioritized stretch goals:

| Goal | Why It Helps | Effort |
|------|--------------|--------|
| Multiple ADHD scenarios (task types, energy levels) | Environment Innovation (40%) — shows breadth | Medium |
| Voice output demo (TTS of model responses) | Storytelling (30%) — compelling demo, Reachy Mini tie-in | Low |
| Real-time reward visualization dashboard | Training improvement (20%) — better evidence | Medium |
| LLM-as-judge for more nuanced reward scoring | Reward pipeline (10%) — more sophisticated | High |
| Mercor agent benchmark integration | Sponsor track alignment — potential separate prize | TBD |

# CLAUDE.md

This file provides context to Claude Code when working on this project.

## TL;DR - What We're Building

An **OpenEnv environment that evaluates ADHD coaching quality** by:
1. **Tracking user state** (sitting time, exercise, posture) - the "knobs"
2. **Scoring responses** using composable rubric criteria (start with 1, iterate)
3. **Evaluating tool calling** (did model call appropriate tools?)
4. **Using composable rubrics** (PrimeIntellect verifiers framework)

**Deliverable**: Environment deployed on HF Spaces + manual testing demonstrations
**Innovation**: State tracking + tool calling evaluation (not just text scoring)
**Focus**: Perfect the environment first, model training is optional
**Approach**: Start simple (1 criterion), iterate based on manual testing

## Project Overview

**Purpose**: Create an innovative OpenEnv environment that evaluates and scores ADHD task initiation coaching responses, enabling reinforcement learning for better AI executive function support.

**Problem Domain**: ADHD executive function scaffolding - helping people with ADHD overcome task initiation paralysis through AI coaching that can be evaluated and improved.

**Hackathon Context**: OpenEnv Hackathon SF (March 7-8, 2025) at Shack15. Submission deadline: Sunday March 8, 1:00 PM.

**Hackathon Category**: **Personalized Tasks** - Executive function assistant for personal task initiation challenges

**Primary Deliverable**: The **environment itself** (deployed on HF Spaces) - an innovative evaluation system for ADHD coaching quality

**Key Technologies**:
- **OpenEnv 0.2.1** - RL environment framework (FastAPI-based) - **CORE DELIVERABLE**
- **PrimeIntellect verifiers** - Rubric framework for composable reward functions
- **HuggingFace Spaces** - Environment deployment platform
- **ROCm devcontainer** - Local development environment

## Judge Feedback (Critical Architecture Decisions)

**Key insights from hackathon judges:**

1. **"Just back-and-forth text doesn't need an environment. You need state tracking."**
   - Environment must track observable user state ("knobs")
   - Examples: sitting time, exercise, posture, time of day
   - Innovation is in state-aware evaluation, not just text scoring

2. **"Stage 1 is manual interaction. Don't worry about models yet."**
   - Focus on hardcoded/manual testing first
   - Verify environment exposes right "knobs" and scores reasonably
   - Model training is optional demonstration

3. **"We want to see tool calling evaluation."**
   - Reward models for calling appropriate tools
   - Example: `adhd_task_initiation_coach` tool for task initiation
   - Adds agent reasoning evaluation, not just language generation

4. **"Use OpenEnv's rubric framework."**
   - Built-in Rubric class for composable scoring functions
   - Each criterion is a separate async function
   - Makes scoring explainable and extensible

**How this changed our approach:**
- ❌ Not: Simple text scoring environment
- ✅ Yes: State-tracking environment with tool calling evaluation
- ❌ Not: Rush to train a model
- ✅ Yes: Perfect the environment with manual testing first
- ❌ Not: Single monolithic reward function
- ✅ Yes: Rubric with composable scoring functions

## The Core Concept

### What We're Building

**The Environment is the Product**: We're building an innovative OpenEnv environment that can evaluate ADHD coaching response quality. Think of it as a sophisticated grading system for AI executive function coaches.

**Why This Matters**:
- Current AI assistants often give unhelpful responses to ADHD task initiation paralysis (e.g., "What would you like to work on?" when someone is stuck)
- Our environment quantifies what makes a good ADHD coaching response
- This enables anyone to train/fine-tune models to be better ADHD coaches

### The Gym Analogy
- The **environment** (HF Space) is the gym that evaluates responses - **THIS IS WHAT WE'RE BUILDING**
- The **reward function** scores response quality (0.0-1.0) - **THE INNOVATION**
- The **LLM** (SmolLM3-3B) is an athlete that can be trained using our gym - **DEMONSTRATION**
- The **deliverable** is the environment + evidence it can improve models - **ENVIRONMENT FIRST**

### What Makes a Good ADHD Coaching Response?

**Approach: Start simple, iterate based on manual testing**

We'll build up the reward function incrementally:
1. Start with **1 criterion** (e.g., tool calling)
2. Test manually with hardcoded examples
3. Add criteria one at a time
4. Iterate based on what we observe

**Rubric Framework Pattern** (using PrimeIntellect verifiers):
```python
from verifiers import Rubric

async def criterion_1(action, state) -> float:
    """First scoring criterion - TBD during development"""
    # Will define during implementation
    pass

async def criterion_2(action, state) -> float:
    """Second scoring criterion - added after testing criterion_1"""
    # Will add if needed
    pass

# Compose into rubric (start with 1, add more as we go)
rubric = Rubric(funcs=[criterion_1])  # Start simple!
```

**Potential Criteria to Consider** (will choose during development):
- Tool calling (did model call the right tool?)
- Response format (directive vs question)
- Response length (cognitive load proxy)
- State awareness (responds to sitting time, energy, etc.)
- Task decomposition (micro-steps vs large steps)
- Tone safety (avoiding pressure language)

**Philosophy**:
- **Better to have 1 working criterion than 5 broken ones**
- **Manual testing will reveal what criteria matter**
- **Rubric framework makes it easy to add/remove criteria**

## Environment-First Development Strategy

### What "Environment is the Product" Means

**The environment is a service that:**
1. **Receives** ADHD coaching responses (text)
2. **Evaluates** response quality against ADHD-specific criteria
3. **Returns** numerical scores (rewards) and feedback
4. **Generates** diverse realistic ADHD task initiation scenarios

**Think of it like:**
- A unit testing framework for ADHD coaching quality
- A grading API for executive function support responses
- A gym where any AI model can practice and improve

### What Makes Our Environment Innovative

**Judge Feedback**: "Just back-and-forth text doesn't need an environment. We need state tracking."

**1. State Tracking ("Knobs")**: Environment tracks realistic user state that models must respond to
   - **Sitting time**: How long at desk (minutes)
   - **Time of day**: Morning/afternoon/evening (affects energy)
   - **Exercise today**: Minutes of physical activity
   - **High executive function tasks completed**: Cognitive load for the day
   - **Posture**: Upright/slouched/standing (fatigue indicator)
   - **Work session duration**: Time since starting work today

**2. Tool Calling Evaluation**: Rewards models for calling appropriate tools
   - `adhd_task_initiation_coach`: Main coaching tool
   - `set_timer`: Focus timers for task boxing
   - `break_down_task`: Decompose large tasks into micro-steps
   - Scoring includes: Did model call the right tool for this state?

**3. ADHD-Specific Evaluation**: Not generic helpfulness, but specific to task initiation paralysis
   - Directive vs question-based responses
   - Appropriate cognitive load (word count as proxy)
   - State-aware coaching (different advice for low energy vs high sitting time)

**4. Rich Scenario Modeling**: 20-50 diverse ADHD task initiation situations
   - Task types (email, writing, phone call, presentation)
   - Combined with randomized state for high variety

**5. Deployable Service**: Anyone can use it via HTTP API
   - Deployed on HF Spaces
   - OpenEnv 0.2.1 standard interface
   - Can be used to train any model, not just ours

### Demonstration vs Deliverable

**Judge Feedback**: "Stage 1 is to get the environment running and interact with it manually. Don't worry about getting a model plugged in yet."

**The Deliverable (REQUIRED):**
- The environment code with state tracking
- Deployed on HF Spaces
- Documentation of how it works
- Manual/hardcoded interactions demonstrating it works

**Stage 1: Manual Testing (PRIMARY FOCUS)**
```python
# Manually test the environment with hardcoded interactions

# Reset environment - get a scenario with state
result = env.reset()
print(f"Scenario: {result.observation.scenario}")
print(f"State: {result.observation.state}")
# State: {sitting_time: 90, energy: "low", exercise: 0, posture: "slouched"}

# Manually craft a response
action = ADHDAction(
    tool_calls=["adhd_task_initiation_coach"],
    message="Stand and stretch for 30 seconds, then type just the recipient name."
)

result = env.step(action)
print(f"Reward: {result.reward}")
print(f"Explanation: {result.observation.metadata}")

# Try another scenario
result = env.reset()  # Different state this time
# Shows: environment exposes the right "knobs" and scores appropriately
```

**Stage 2: Model Training (OPTIONAL - if time permits)**
- Evidence the environment can improve a model
- Before/after response quality
- Reward curves from training

## Codebase Structure

```
adhd-coach/
├── docs/
│   └── planning/                         # Hackathon planning documents
│
├── src/
│   └── environment/                      # ⭐ CORE DELIVERABLE ⭐
│       ├── __init__.py
│       ├── adhd_env.py                  # Environment with reset()/step() - FOCUS
│       ├── user_state.py                # State tracking ("knobs") - INNOVATION
│       ├── scenarios.py                 # 20-50 ADHD scenarios - VARIETY
│       ├── reward.py                    # Rubric with tool calling - INNOVATION
│       └── server.py                    # FastAPI server for deployment
│
├── scripts/
│   ├── test_state_generation.py        # Test state variety
│   ├── test_scenarios.py               # Test scenario variety
│   ├── test_rubric.py                  # Test rubric scoring functions
│   ├── test_environment_manual.py      # Manual interactions - PRIMARY TEST
│   ├── test_environment_local.py       # Test with uvicorn locally
│   ├── demo_environment.py             # Interactive demo for video
│   └── test_environment_deployed.py    # Test deployed HF Space
│
├── training/                            # Optional - only if time permits
│   ├── train.py                        # Training script (Colab)
│   └── config.py                       # GRPO config
│
├── pyproject.toml                       # Dependencies (uv-managed)
├── README.md                            # Project documentation
└── CLAUDE.md                            # This file
```

**Where to focus development time:**
1. **src/environment/** (85% of effort)
   - State tracking implementation (30%)
   - Rubric with tool calling (25%)
   - Scenario generation (20%)
   - Environment integration (10%)
2. **scripts/test_environment_manual.py** (10% of effort) - Manual testing
3. **scripts/demo_environment.py** (5% of effort) - Demo video
4. **training/** (0% initially) - Only if time permits

## Development Workflow

### Primary Focus: Environment Development (Local)

**This is 90% of the work** - develop and test the environment locally:

```bash
# Install dependencies
uv add openenv_core

# Develop environment logic
cd src/environment
vim adhd_env.py reward.py scenarios.py

# Test locally with uvicorn
python -m uvicorn adhd_env.server.app:app --host 0.0.0.0 --port 8001

# Test client in another terminal
python scripts/test_environment_local.py

# Test reward function with hardcoded examples
python scripts/test_reward_function.py
```

**Deploy to HF Spaces**:
```bash
# Push environment code to HF Space
# HF Spaces runs uvicorn automatically
# Get URL: https://your-username-adhd-env.hf.space

# Test deployed environment
python scripts/test_environment_deployed.py --url https://your-username-adhd-env.hf.space
```

**Commit to GitHub**:
```bash
git add src/environment/ scripts/
git commit -m "Add ADHD coaching evaluation environment"
git push
```

### Optional: Model Training (Only if time permits)

**Judge Feedback**: Focus on the environment first, model training is optional.

If time permits after environment is working:
1. Deploy environment to HF Spaces
2. Run quick training in Colab with TRL
3. Show before/after response quality

**Priority**: Environment with manual testing > Model training

## Architectural Decisions

### The Environment as a Service

**What we're building:**
```
┌────────────────────────────────────────────────────┐
│  HF Spaces (The Deliverable)                       │
│                                                    │
│  ADHD Coaching Evaluation Environment              │
│  (FastAPI/uvicorn)                                 │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  reset()                                     │  │
│  │  → Generate ADHD scenario                    │  │
│  │  → Generate randomized user state (knobs)    │  │
│  │  → Return scenario + state                   │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  step(tool_calls, message)                   │  │
│  │  → Score tool selection                      │  │
│  │  → Score response quality                    │  │
│  │  → Return reward (0.0-1.0) + explanation     │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  State Tracking ("Knobs")                    │  │
│  │  • Sitting time (minutes)                    │  │
│  │  • Time of day                               │  │
│  │  • Exercise minutes today                    │  │
│  │  • High exec tasks completed                 │  │
│  │  • Posture (upright/slouched)                │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Rubric (Innovation)                         │  │
│  │  • Called correct tool (0.40)                │  │
│  │  • Not a question (0.30)                     │  │
│  │  • Appropriate length (0.30)                 │  │
│  │  • Future: State-aware scoring               │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
                    │
                    │ HTTP API (OpenEnv 0.2.1)
                    ▼
        ┌───────────────────────┐
        │  Anyone can use to:   │
        │  • Test responses     │
        │  • Train models       │
        │  • Build ADHD tools   │
        └───────────────────────┘
```

**Key insight**:
- **State tracking** is the innovation (not just text scoring)
- **Tool calling evaluation** adds real agent reasoning
- **Rubric framework** makes scoring composable and explainable

### Production Deployment: Generic LLM + Tool Calling

**Architecture for Reachy Mini** (post-hackathon):
```
User: "Reachy, I'm stuck"
    ↓
Generic LLM (on Reachy Mini)
    ↓
Decides: This is an ADHD task initiation issue
    ↓
Calls tool: adhd_task_initiation_coach(context={sitting: 90, energy: "low"})
    ↓
Tool returns: "Stand and stretch 30sec, then type recipient name"
    ↓
LLM delivers to user
```

**Why this approach:**
- ✅ Extensible: Can add other tools (calendar, cooking, etc.)
- ✅ Modular: ADHD coaching is one tool among many
- ✅ Easier to debug: Tool logic is explicit
- ✅ Multi-domain: Reachy can help with multiple task types

**Judge feedback**: They want to see tool calling in the environment evaluation

### Why Local Development Works

**Environment development needs:**
- Pure Python logic (scenario generation, reward scoring)
- No GPU required (CPU-only evaluation)
- Fast iteration (edit code, test immediately)
- uvicorn testing (validates HF Spaces deployment will work)

**Training demonstration needs (optional - if time permits):**
- GPU for model inference
- Can use Colab/HF Spaces
- Not required for minimum viable submission

## Important Patterns

### Environment Implementation (CORE FOCUS)

```python
from openenv.core import StepResult
from pydantic import BaseModel
from verifiers import Rubric
from typing import List

class ADHDAction(BaseModel):
    """Action: Tool calls + coaching response to evaluate"""
    tool_calls: List[str]  # e.g., ["adhd_task_initiation_coach", "set_timer"]
    message: str           # The coaching response text

class ADHDState(BaseModel):
    """User state - the 'knobs' the environment tracks"""
    sitting_time_minutes: int = 0
    time_of_day: str = "morning"  # morning, afternoon, evening
    work_session_minutes: int = 0
    exercise_minutes_today: int = 0
    high_exec_tasks_completed: int = 0
    posture: str = "upright"  # upright, slouched, standing

class ADHDObservation(BaseModel):
    """Observation: ADHD scenario + user state"""
    scenario: str          # The coaching prompt
    state: dict            # User state (sitting time, energy, etc.)
    done: bool
    reward: float
    metadata: dict         # Why this score? Explanation

class ADHDEnvironment:
    def __init__(self):
        self.rubric = self._create_rubric()
        self.current_state = None
        self.current_scenario = None

    def reset(self) -> StepResult:
        """Generate a new ADHD scenario with randomized state - INNOVATION"""

        # Generate random but realistic user state
        self.current_state = ADHDState(
            sitting_time_minutes=random.randint(0, 120),
            time_of_day=random.choice(["morning", "afternoon", "evening"]),
            work_session_minutes=random.randint(0, 240),
            exercise_minutes_today=random.randint(0, 60),
            high_exec_tasks_completed=random.randint(0, 5),
            posture=random.choice(["upright", "slouched", "standing"])
        )

        # Generate scenario
        self.current_scenario = self._generate_scenario()

        observation = ADHDObservation(
            scenario=self.current_scenario,
            state=self.current_state.dict(),
            done=False,
            reward=0.0,
            metadata={}
        )
        return StepResult(observation=observation, reward=0.0, done=False)

    def step(self, action: ADHDAction) -> StepResult:
        """Score the coaching response - THE CORE INNOVATION

        Single-turn: Each step is independent, done=True after scoring
        """
        # Score using rubric (composable scoring functions)
        reward = self.rubric.score(action, self.current_state)
        metadata = self._explain_score(action, reward)

        observation = ADHDObservation(
            scenario=self.current_scenario,
            state=self.current_state.dict(),
            done=True,  # Single-turn: done after one response
            reward=reward,
            metadata=metadata
        )
        return StepResult(observation=observation, reward=reward, done=True)

    def _create_rubric(self) -> Rubric:
        """Create composable scoring rubric - THE INNOVATION

        Start with 1 criterion, add more based on manual testing
        """

        async def criterion_1(action, state) -> float:
            # Will define during implementation
            # Start simple - maybe just tool calling check
            pass

        # Start with 1 function, add more as we test
        return Rubric(funcs=[criterion_1])

    def _explain_score(self, action: ADHDAction, reward: float) -> dict:
        """Return metadata explaining why this score was given"""
        return {
            "tool_calls": action.tool_calls,
            "message": action.message,
            "total_reward": reward,
            "state_snapshot": self.current_state.dict(),
            # Will add criterion-specific explanations as we define them
        }
```

**Key points:**
- **State tracking**: Randomized realistic user state in each `reset()`
- **Tool calling**: Action includes both tool calls and message
- **Rubric framework**: Composable scoring functions (easy to extend)
- **Single-turn**: `done=True` after each step (can add multi-turn later)
- **Explainable**: Metadata shows why each score was given

### Scenario Generation Pattern (FOCUS HERE)

```python
class ScenarioGenerator:
    """Generate diverse ADHD task initiation scenarios"""

    def __init__(self):
        self.task_types = [
            "writing introduction",
            "replying to email",
            "starting presentation",
            "beginning project",
            "making phone call",
            "preparing for meeting",
            "creating outline",
            "drafting proposal",
            # ... 15-45 more for variety
        ]

    def generate_scenario(self) -> str:
        """Generate a random ADHD scenario (user utterance)

        Important: Scenarios should be what the USER WOULD SAY to Reachy,
        not internal states the robot can't observe.
        """
        import random

        task = random.choice(self.task_types)

        # User statements (observable via conversation)
        templates = [
            f"I can't start {task}",
            f"I'm stuck on {task}",
            f"I've been avoiding {task}",
            f"I don't know how to begin {task}",
        ]

        return random.choice(templates)

class StateGenerator:
    """Generate randomized user state (the 'knobs')"""

    def generate_state(self) -> ADHDState:
        """Generate random but realistic user state

        In production, these would come from Reachy sensors:
        - Sitting time: Camera tracking
        - Time of day: System clock
        - Exercise: Wearable integration
        - Posture: Camera/posture detection
        """
        import random

        return ADHDState(
            sitting_time_minutes=random.randint(0, 120),
            time_of_day=random.choice(["morning", "afternoon", "evening"]),
            work_session_minutes=random.randint(0, 240),
            exercise_minutes_today=random.randint(0, 60),
            high_exec_tasks_completed=random.randint(0, 5),
            posture=random.choice(["upright", "slouched", "standing"])
        )
```

**This is where environment innovation happens**:
- 20-50 diverse scenarios (variety of tasks)
- Randomized state combinations (variety of contexts)
- Same scenario + different state = different ideal responses

## Known Issues and Gotchas

### ROCm Devcontainer Limitations

- **FP16 only**: Officially validated on Strix Halo gfx1151
- **BF16 untested**: TRL GRPO defaults to BF16 - watch for this locally (use `torch_dtype=torch.float16`)
- **vLLM unavailable**: PyPI vLLM = CUDA only; ROCm requires custom build
  - **Workaround**: Set `use_vllm=False` in GRPOConfig for local testing (slow)
  - **Solution**: Use Colab for actual training (CUDA + vLLM)

### OpenEnv Import Names

```python
# CORRECT - echo_env example
from echo_env import EchoEnv, EchoAction

# WRONG - these don't work
from openenv_echo import EchoEnv        # ModuleNotFoundError
from openenv_echo_env import EchoEnv    # ModuleNotFoundError
```

**Pattern**: Package name uses hyphens (`openenv-echo-env`), module name uses underscores (`echo_env`)

### API Response Structure

```python
# Both reset() and step() return StepResult
result = client.reset()
result.observation  # Environment-specific observation object
result.reward       # float (0.0 for reset)
result.done         # bool

step_result = client.step(action)
step_result.observation  # Updated observation
step_result.reward       # float (actual reward)
step_result.done         # bool
```

### Single-turn vs Multi-turn

**Hackathon approach: Single-turn with rich state**
- Each `reset()` generates a scenario + randomized user state
- One `step()` scores the response, then `done=True`
- State provides context but doesn't evolve within episode
- Variety comes from randomized state combinations (not state evolution)

**Example:**
```python
# Episode 1
reset() → {scenario: "stuck on email", state: {sitting: 90, energy: "low"}}
step(response) → reward, done=True

# Episode 2 (different state)
reset() → {scenario: "stuck on email", state: {sitting: 15, energy: "high"}}
step(response) → reward, done=True
```

**Future enhancement: Multi-turn with state evolution**
- State updates after each step (sitting time increments, posture degrades)
- Conversational coaching (response builds on previous attempts)
- Multi-turn: reset → step → step → step (state evolves)

## External Dependencies

### HuggingFace Spaces
- **Environment deployment**: FastAPI app runs automatically
- **URL pattern**: `https://your-username-adhd-env.hf.space`
- **No GPU needed**: Environment is CPU-only (pure Python)

### OpenEnv Dependencies
```bash
# Core environment framework
uv add openenv_core

# Rubric framework for composable reward functions
uv add verifiers

# For testing echo_env locally
uv add "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"
```

### Optional: Training Infrastructure (if time permits)
```bash
# Only if demonstrating model training
uv add trl transformers accelerate datasets
```

## Testing Strategy

**Judge Feedback**: "Stage 1 is manual interaction. See if the environment exposes the right 'knobs' and gives reasonable scores."

### Stage 1: Manual Testing (PRIMARY FOCUS)

**1. Test state generation**:
```bash
python scripts/test_state_generation.py
```

```python
from src.environment.user_state import StateGenerator

generator = StateGenerator()

# Generate 20 states and verify variety
states = [generator.generate_state() for _ in range(20)]

# Check variety
unique_times = len(set(s.time_of_day for s in states))
assert unique_times == 3, "Should have morning/afternoon/evening"

unique_postures = len(set(s.posture for s in states))
assert unique_postures == 3, "Should have upright/slouched/standing"

print("Sample states:")
for state in states[:5]:
    print(state.dict())
```

**2. Test manual environment interaction**:
```bash
python scripts/test_environment_manual.py
```

```python
from src.environment.adhd_env import ADHDEnvironment, ADHDAction

env = ADHDEnvironment()

# Reset - get scenario with state
result = env.reset()
print(f"Scenario: {result.observation.scenario}")
print(f"State: {result.observation.state}")

# Manually craft what we think is a good response
action_good = ADHDAction(
    tool_calls=["adhd_task_initiation_coach"],
    message="Open email and type just the recipient name. Stop there."
)

result = env.step(action_good)
print(f"Good action reward: {result.reward}")
print(f"Explanation: {result.observation.metadata}")

# Try what we think is a bad response
result = env.reset()
action_bad = ADHDAction(
    tool_calls=[],
    message="What do you want to work on?"
)

result = env.step(action_bad)
print(f"Bad action reward: {result.reward}")
print(f"Explanation: {result.observation.metadata}")

# Verify scoring makes sense
assert result_good.reward > result_bad.reward, "Good should score higher than bad"
```

**3. Test rubric scoring**:
```bash
python scripts/test_rubric.py
```

```python
# Test each scoring function independently as we add them
from src.environment.reward import criterion_1  # Start with just 1

# Test the first criterion
action_good = ADHDAction(tool_calls=["adhd_task_initiation_coach"], message="Test")
score_good = await criterion_1(action_good, None)
print(f"Good action score: {score_good}")

action_bad = ADHDAction(tool_calls=[], message="Test")
score_bad = await criterion_1(action_bad, None)
print(f"Bad action score: {score_bad}")

# Verify it discriminates
assert score_good > score_bad, "Criterion should score good actions higher"
```

**4. Test scenario variety**:
```bash
python scripts/test_scenarios.py
```

```python
from src.environment.scenarios import ScenarioGenerator

generator = ScenarioGenerator()
scenarios = [generator.generate_scenario() for _ in range(50)]

# Check for variety
unique_scenarios = len(set(scenarios))
assert unique_scenarios >= 20, "Need at least 20 unique scenarios"

print(f"Generated {unique_scenarios} unique scenarios")
print("Sample scenarios:")
for s in scenarios[:10]:
    print(f"- {s}")
```

### Stage 2: Deployment Testing

**5. Test locally with uvicorn**:
```bash
# Start server
python -m uvicorn src.environment.adhd_env.server.app:app --host 0.0.0.0 --port 8001

# Test in another terminal
python scripts/test_environment_local.py
```

**6. Test deployed environment**:
```bash
# After deploying to HF Spaces
python scripts/test_environment_deployed.py --url https://your-username-adhd-env.hf.space
```

### Stage 3: Demo Preparation

**7. Create interactive demo**:
```bash
python scripts/demo_environment.py
```

This script should:
- Show environment generating scenarios with different states
- Demonstrate same scenario + different state = different ideal responses
- Score several good/bad/medium responses with explanations
- Show tool calling evaluation
- Highlight state tracking innovation

## Judging Criteria Focus

**Official Weights**: 40% Environment Innovation + 30% Storytelling + 20% Training + 10% Reward Pipeline

**Our Strategy**: Focus on the 40% (Environment Innovation) as the primary deliverable

**What This Means**:
- The **environment** is more important than the trained model
- Innovation in evaluation (reward function, scenarios, user state modeling) matters most
- Training is supporting evidence that the environment works, not the end goal
- Storytelling (ADHD-specific problem, Reachy Mini future vision) is second priority

## Submission Requirements Checklist

**MUST HAVE** (Environment Innovation - 40%):
- [ ] State tracking implementation (sitting time, exercise, posture, etc.)
- [ ] Tool calling evaluation in reward function
- [ ] OpenEnv 0.2.1 environment with Rubric framework
- [ ] 20-50 diverse ADHD scenarios
- [ ] Manual testing demonstrations (hardcoded interactions)
- [ ] Deployed on HF Spaces
- [ ] Public GitHub repo with environment code
- [ ] Clear documentation of innovation (state tracking + tool calling)

**SHOULD HAVE** (Storytelling - 30%):
- [ ] 1-minute YouTube demo video showing:
  - [ ] State tracking ("knobs") in action
  - [ ] Same scenario + different state = different ideal responses
  - [ ] Tool calling evaluation
  - [ ] Manual interactions with scoring explanations
- [ ] Clear explanation of ADHD task initiation problem
- [ ] README.md with environment innovation highlights

**NICE TO HAVE** (if time permits):
- [ ] State-aware scoring (bonus points for responding to state)
- [ ] Multi-turn state evolution
- [ ] Model training demonstration (Training - 20%)
- [ ] Evidence of training improvement (reward curves)

## Stretch Goals (Prioritized by Judging Criteria)

Judging weights: 40% Environment Innovation + 30% Storytelling + 20% Training + 10% Reward Pipeline

**Focus on Environment Innovation (40%):**

| Goal | Impact | Effort | Priority |
|------|--------|--------|----------|
| State tracking ("knobs") | Environment Innovation - **judge feedback** | Medium | **CRITICAL** |
| Tool calling evaluation | Environment Innovation - **judge feedback** | Medium | **CRITICAL** |
| 20-50 diverse ADHD scenarios | Environment Innovation - shows real variety | Medium | **HIGH** |
| Rubric framework integration | Reward Pipeline - composable scoring | Low | **HIGH** |
| Manual testing demonstrations | Storytelling - shows environment works | Low | **HIGH** |
| State-aware scoring | Environment Innovation - advanced rubric | Medium | MEDIUM |
| Interactive environment demo | Storytelling - shows environment in action | Low | MEDIUM |
| Voice output demo (TTS) | Storytelling - Reachy Mini tie-in | Low | LOW |
| Multi-turn state evolution | Environment Innovation - advanced | High | LOW |
| Actual model training | Training - optional if time permits | High | LOW |

## Longer-Term Vision (Post-Hackathon)

**Reachy Mini Integration**:
- Pollen Robotics/HuggingFace desktop robot (~$299)
- Physical presence for "body doubling" effect (ADHD research-backed)
- Camera, mic, speakers, expressive movement
- Deploy trained model weights to robot

**Additional Features**:
- Posture detection via camera
- Break timing suggestions
- Hydration reminders
- Full conversational interface

**The pitch story**: "We trained a model to be an ADHD coach, and here's the physical robot it will run on."

## Resources

- **OpenEnv**: https://github.com/meta-pytorch/OpenEnv
- **TRL OpenEnv Integration**: https://huggingface.co/docs/trl/main/en/openenv
- **GRPO Paper**: https://arxiv.org/abs/2402.03300
- **Unsloth GRPO Notebooks**: https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks
- **Reachy Mini**: https://huggingface.co/spaces/pollen-robotics/Reachy_Mini
- **Planning Docs**: `docs/planning/` (all hackathon planning)
- **HF Submission Form**: https://cerebralvalley.ai/e/openenv-hack...

---

## Quick Start Guide

**Judge Feedback**: "Don't worry about getting a model plugged in yet. Focus on manual interaction."

**Day 1 priorities** (Environment Innovation - 40% of judging):
1. Implement `src/environment/user_state.py` - ADHDState class with "knobs"
2. Implement `src/environment/scenarios.py` - 20-50 diverse ADHD scenarios
3. Implement `src/environment/reward.py` - Rubric with tool calling + response quality
4. Implement `src/environment/adhd_env.py` - OpenEnv interface with state tracking
5. **Manual testing**: Hardcoded interactions to verify state tracking works
6. Test locally: `python scripts/test_environment_manual.py`
7. Deploy to HF Spaces

**Day 2 priorities** (Storytelling - 30% + demonstration):
1. Create `scripts/demo_environment.py` - interactive demo showing state tracking
2. Record 1-minute demo video:
   - Show environment generating scenarios with different states
   - Show how same prompt + different states → different ideal responses
   - Show scoring with tool calls
3. Write README.md explaining the innovation (state tracking + tool calling)
4. **Optional**: Quick model training demo if time permits

**Remember**:
- The environment is the product
- Manual/hardcoded testing demonstrates it works
- Model training is optional

---

**Technical Note**: This is a ROCm devcontainer project. GPU is NOT required - environment development is pure Python logic (CPU-only). Optional training demonstration can use Colab/HF Spaces if needed.

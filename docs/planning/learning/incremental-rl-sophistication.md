# Incremental RL Sophistication Strategy

**Purpose**: Build the ADHD coaching environment incrementally, starting with the simplest possible automated reward function and adding sophistication only after validating the pipeline works.

**Key Constraint**: No code written before Saturday morning (hackathon rules). This document contains pseudocode for planning purposes only.

---

## Core Insight: Question-Based Coaching

### The Problem with Vague User Statements

When users are stuck, they tend to write vague statements:
- "I can't start this task"
- "I'm frozen"
- "I don't know where to begin"

### Better Initial Response: Clarifying Questions

Before jumping to prescriptive solutions, the model should:
1. **Ask a clarifying question** to understand the specific block
2. **Prompt self-reflection** to help the user articulate the issue
3. **Enable collaboration** rather than one-way instruction

**Examples of good initial responses:**
- "What in particular are you stuck on?"
- "What is your current fatigue level?"
- "Which part feels most overwhelming?"

**Examples of bad initial responses:**
- "You should start with an outline."
- "Just break it into smaller tasks."
- "Open a blank document and begin."

### Why This Is Better Than Micro-Actions

| Approach | Pros | Cons |
|----------|------|------|
| **Immediate micro-actions** | Direct, actionable | Might miss the actual block, feels prescriptive |
| **Clarifying questions first** | Collaborative, gathers context | Requires multi-turn conversation |

For an MVP, **teaching the model to ask questions** is:
- ✅ Simpler to score automatically
- ✅ Demonstrates understanding of ADHD coaching principles
- ✅ More unique than typical "follow instructions" RL environments
- ✅ Expandable to multi-turn conversations later

---

## MVP Strategy: Three Versions

### V1: Binary Question Detection (Saturday Morning)

**Goal**: Validate the entire pipeline works (HF Spaces → Colab → GRPO)

**Reward logic:**
```
FUNCTION reward_function(response, user_state):
    first_sentence = extract_first_sentence(response)

    IF first_sentence ends with "?":
        RETURN 1.0
    ELSE:
        RETURN 0.0
```

**What the model learns:**
- Responses that start with questions get rewarded
- Responses that start with statements get penalized
- Over time, model learns to ask questions instead of giving directives

**Success criteria:**
- ✅ Reward curve goes up over training
- ✅ Before training: model generates statements
- ✅ After training: model generates questions

**Test cases** (for validating logic before deployment):
```
"What in particular are you stuck on?"     → 1.0
"What is your current fatigue level?"      → 1.0
"You should start with an outline."        → 0.0
"Just break it into smaller tasks."        → 0.0
"Which part feels most overwhelming?"      → 1.0
```

---

### V1.5: Good vs Generic Questions (Saturday Afternoon)

**Goal**: Teach the model to ask GOOD clarifying questions, not just any question

**Reward logic:**
```
FUNCTION reward_function(response, user_state):
    response_lower = lowercase(response)
    first_sentence = extract_first_sentence(response)

    # Not a question at all
    IF NOT first_sentence ends with "?":
        RETURN 0.0

    # Great: Specific clarifying questions
    specific_patterns = [
        "what in particular",
        "what specifically",
        "which part",
        "what aspect",
        "what is your",
        "where are you stuck"
    ]
    IF any pattern in response_lower:
        RETURN 1.0

    # Good: Open-ended clarifying questions
    good_question_starts = ["what", "which", "how", "where"]
    IF first_sentence starts with any good_question_starts:
        RETURN 0.7

    # Generic/unhelpful questions
    generic_patterns = ["how can i help", "what do you want", "can i assist"]
    IF any pattern in response_lower:
        RETURN 0.3

    # Default: it's a question, but not sure if good
    RETURN 0.5
```

**What the model learns:**
- Specific questions ("What specifically...") get highest reward
- Open-ended questions ("What part...") get good reward
- Generic questions ("How can I help?") get low reward
- Statements get zero reward

**Success criteria:**
- ✅ Model generates "What specifically..." more often than "What..."
- ✅ Model avoids generic "How can I help?" responses
- ✅ Reward curve continues to improve

**Test cases:**
```
"What in particular are you stuck on?"     → 1.0 (specific)
"What's blocking you?"                     → 0.7 (good)
"How can I help?"                          → 0.3 (generic)
"You should start with..."                 → 0.0 (not a question)
```

---

### V2: Multi-Criteria Rubric (Sunday Morning - Stretch)

**Goal**: Add sophistication if V1/V1.5 work well and there's time

**Reward logic:**
```
FUNCTION reward_function(response, user_state):
    score = 0.0

    # Clarifying question (0.40)
    IF response starts with good question:
        score += 0.40

    # Energy-appropriate (0.30)
    IF user_state.energy == "low":
        IF response contains low_energy_phrases:
            score += 0.30
    ELSE:
        score += 0.30  # not critical if not low energy

    # Tone safety (0.20)
    bad_phrases = ["you should", "just do it", "it's easy"]
    IF NOT any bad_phrase in response:
        score += 0.20

    # Length (0.10)
    IF length(response) < 200 characters:
        score += 0.10

    RETURN score
```

**What the model learns:**
- Most important: Ask clarifying questions (40%)
- Important: Match user energy level (30%)
- Important: Avoid pressure language (20%)
- Nice to have: Keep it concise (10%)

---

## Why This Incremental Approach Works

### Technical Benefits

| Aspect | Benefit |
|--------|---------|
| **Validate pipeline first** | Simplest reward proves HF Spaces ↔ Colab ↔ GRPO works |
| **Fast iteration** | Binary scoring is trivial to compute |
| **Clear success** | Easy to see if model learned (does it ask questions?) |
| **Expandable** | Can add complexity once basics work |
| **Debuggable** | Fewer moving parts initially |

### Hackathon Benefits

| Aspect | Benefit |
|--------|---------|
| **MVP by noon** | V1 could be working within 2-3 hours Saturday |
| **Demonstrable** | Before/after examples are striking |
| **Unique** | Different from typical RL environments |
| **Defensible** | Based on ADHD coaching principles (collaborative vs prescriptive) |
| **Storytelling** | Clear narrative: "We taught the model to ask before telling" |

### Coaching Benefits

| Aspect | Benefit |
|--------|---------|
| **Better alignment** | Clarifying questions match real ADHD support patterns |
| **Collaborative** | Model learns to gather info before prescribing |
| **Reduces harm** | Avoids unhelpful "just do it" responses |
| **Expandable** | Foundation for multi-turn conversation |

---

## Testing Strategy (Conceptual)

### Before Deployment - Validate Reward Logic

**Pseudocode for local testing:**
```
FUNCTION test_reward_function():
    test_cases = [
        ("What in particular are you stuck on?", expected_score),
        ("You should start with an outline.", expected_score),
        ("Which part feels overwhelming?", expected_score),
        # ... more test cases
    ]

    FOR EACH (response, expected) IN test_cases:
        actual = reward_function(response, empty_user_state)
        ASSERT actual == expected
        PRINT "Score: {actual} | Response: {response}"
```

**Purpose:**
- Validate scoring logic before deploying to HF Spaces
- Catch edge cases (empty responses, malformed text, etc.)
- Confirm understanding of what "good" looks like

---

## Saturday Timeline (When Hackathon Starts)

### Saturday Morning: Hours 1-2 (8:00 AM - 10:00 AM)

**Goal**: Get V1 working end-to-end

**Tasks:**
1. Create ADHD environment structure (following echo_env pattern)
2. Implement V1 reward function (binary question detection)
3. Test locally with uvicorn
4. Deploy to HF Spaces
5. Verify HF Space is running

**Success criteria:**
- ✅ Local server responds to reset() and step()
- ✅ HF Space URL is accessible
- ✅ Reward function returns expected scores

---

### Saturday Morning: Hours 3-4 (10:00 AM - 12:00 PM)

**Goal**: Connect Colab and validate training

**Tasks:**
1. Create Colab notebook with GRPO trainer
2. Connect to HF Space environment
3. Run small training test (10 episodes, 4 completions per episode)
4. Check that reward curve trends upward

**Success criteria:**
- ✅ Colab can call environment.reset() and environment.step()
- ✅ GRPO training loop runs without errors
- ✅ Rewards are being returned correctly
- ✅ Model generates text (even if not good yet)

---

### Saturday Afternoon: Hours 5-8 (12:00 PM - 4:00 PM)

**Goal**: Full training run with V1

**Tasks:**
1. Run full GRPO training (100+ episodes)
2. Monitor reward curves
3. Sample model outputs at different checkpoints
4. Collect before/after examples

**Success criteria:**
- ✅ Training completes without crashes
- ✅ Reward curve shows improvement
- ✅ Model generates questions more often after training
- ✅ Have 3-5 good before/after examples

---

### Saturday Late Afternoon: Hours 9-10 (4:00 PM - 6:00 PM)

**Goal**: Upgrade to V1.5 if V1 worked

**Tasks:**
1. Update reward function to V1.5 (good vs generic questions)
2. Re-deploy to HF Spaces
3. Run new training session
4. Compare V1 vs V1.5 results

**Success criteria:**
- ✅ V1.5 reward function works
- ✅ Model learns to ask better questions
- ✅ Can demonstrate improvement over V1

---

### Sunday Morning: Hours 11-14 (8:00 AM - 12:00 PM)

**Goal**: Polish and prepare submission

**Tasks:**
1. Create demo video (1 minute)
2. Document environment on GitHub
3. Write README with examples
4. Prepare presentation

**Stretch (if time):**
- Add V2 multi-criteria reward
- Add multiple user scenarios
- Add LLM-as-judge for more nuanced scoring

---

## Comparison with Original Rubric Approach

### Original Plan (More Complex)

**Reward function:**
```
FUNCTION reward_function(response, user_state):
    score = 0.0

    # Task decomposition (0.30)
    IF response suggests micro-action:
        score += 0.30

    # Single focus (0.20)
    IF question_count <= 1:
        score += 0.20

    # Energy matching (0.20)
    IF user_state.energy == "low" AND response acknowledges it:
        score += 0.20

    # Tone safety (0.20)
    IF NOT contains pressure_language:
        score += 0.20

    # Length (0.10)
    IF sentence_count <= 3:
        score += 0.10

    RETURN score
```

**Problems:**
- More complex to implement
- Harder to debug
- Assumes one-shot response (no clarification phase)
- Multiple criteria = harder to see what model is learning

### New Plan (Simpler, Better)

**V1 Reward function:**
```
FUNCTION reward_function(response, user_state):
    IF starts_with_question(response):
        RETURN 1.0
    ELSE:
        RETURN 0.0
```

**Advantages:**
- Dead simple to implement and debug
- Clear learning signal
- Aligns with collaborative coaching approach
- Easy to expand incrementally

---

## Why the Question-Based Approach Is Superior

### From Coaching Perspective

**Original approach assumed:**
- User clearly states the problem
- Model can immediately suggest micro-action
- One-shot interaction is sufficient

**Question-based approach recognizes:**
- Users are often vague when stuck
- Clarification improves solution quality
- Collaboration beats prescription for ADHD support

### From Technical Perspective

**Original approach:**
- Multiple criteria = complex scoring
- Harder to validate correctness
- Unclear which criterion model is learning

**Question-based approach:**
- Binary signal = clear learning target
- Easy to validate (does it ask questions?)
- Can add complexity incrementally

### From Hackathon Perspective

**Original approach:**
- Risk of debugging complex reward function
- Harder to show clear improvement
- Less unique (similar to instruction-following)

**Question-based approach:**
- Can get MVP working quickly
- Striking before/after examples
- Unique angle: collaborative AI coaching
- Better narrative for judges

---

## Key Success Metrics

### Quantitative (Reward Curves)

- **V1**: Average reward should increase from ~0.5 to ~0.8+ over training
- **V1.5**: Average reward should increase from ~0.4 to ~0.7+ over training
- **Episodes needed**: 100-200 episodes should show clear improvement

### Qualitative (Before/After Examples)

**Before training (Episode 1):**
```
User: "I can't start this task. I'm just frozen."
Model: "You should break the task into smaller steps. Start by making
an outline of what you need to do, then pick the easiest part to
begin with."
Reward: 0.0 (statement, not question)
```

**After training (Episode 100):**
```
User: "I can't start this task. I'm just frozen."
Model: "What in particular are you stuck on?"
Reward: 1.0 (clarifying question)
```

### Demo-Ready Metrics

For judges, show:
- Reward curve graph (trending upward)
- 3-5 before/after comparison examples
- Live demo: ask the trained model to respond to stuck scenarios
- Explanation: "We taught it to ask before telling"

---

## Stretch Goals (If Time Permits)

### Multi-Turn Conversations

Extend the environment to handle:
1. User: "I'm stuck on this task"
2. Model: "What specifically is blocking you?" (gets reward)
3. User: "I don't know what the first step is"
4. Model: "Open a blank doc and write just the title" (gets reward for micro-action)

**Reward function would evolve:**
```
FUNCTION reward_function(response, conversation_state):
    IF conversation_state.turn == 1:
        # First turn: reward questions
        RETURN 1.0 if is_question(response) else 0.0
    ELSE:
        # Later turns: reward micro-actions
        RETURN score_micro_action(response)
```

### LLM-as-Judge (Still Automated)

Use a small LLM to score responses:
```
FUNCTION llm_judge_reward(response, user_state):
    judge_prompt = "
        Score this ADHD coaching response from 0.0 to 1.0.

        User state: {user_state}
        Response: {response}

        Criteria:
        - Asks clarifying questions before prescribing
        - No pressure language
        - Appropriate for user's energy level

        Score (0.0-1.0):
    "

    score = call_llm_api(judge_prompt)
    RETURN parse_float(score)
```

**Still automated** (no human in loop), but more sophisticated than keyword matching.

---

## Final Recommendation

### Start Simple, Add Complexity Only If Needed

1. **Saturday morning**: Get V1 working (binary question detection)
2. **Saturday afternoon**: If V1 works, upgrade to V1.5 (good vs generic questions)
3. **Sunday morning**: Polish, demo, submit
4. **Stretch**: Only add V2/multi-turn/LLM-judge if you have extra time

### Why This Approach Wins

- **Speed**: MVP in 2-3 hours
- **Reliability**: Simple = fewer bugs
- **Demonstrable**: Clear before/after examples
- **Unique**: Collaborative coaching angle
- **Expandable**: Foundation for future work (Reachy Mini deployment)

### The Pitch

> "We trained SmolLM to be a better ADHD coach by teaching it to ask clarifying questions
> instead of jumping to prescriptive solutions. Using GRPO and a simple reward function,
> the model learned to recognize when users need reflection prompts rather than direct
> instructions. This collaborative approach aligns with ADHD coaching best practices."

---

## Resources for Saturday Morning

### Environment Structure (Based on echo_env)
```
adhd_env/
├── __init__.py
├── server/
│   ├── __init__.py
│   ├── app.py              # FastAPI server
│   └── adhd_environment.py # Environment logic
└── requirements.txt
```

### Key Files to Reference

From this repo:
- `openenv-hello-world-flow.md` - Client-server architecture
- `openenv-devcontainer-setup.md` - Installation and deployment
- `the-basics-of-rl-and-grpo.md` - How GRPO training works

External:
- echo_env source: https://huggingface.co/spaces/openenv/echo_env (Files tab)
- TRL GRPO docs: https://huggingface.co/docs/trl/en/openenv
- GRPO paper: https://arxiv.org/abs/2402.03300

---

## Remember: No Code Before Saturday

This document contains **pseudocode for planning purposes only**.

All actual implementation happens **Saturday morning at Shack15** when the hackathon officially starts.

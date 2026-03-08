# OpenEnv & RL Learning Plan — Hackathon Prep
*OpenEnv Hackathon SF — March 7-8, 2025*
*Created: March 2026 — ~1.5 days of prep time available*

---

## What You're Actually Building

OpenEnv is a Meta/PyTorch framework for wrapping **environments** that LLMs interact with during RL training.

- Think: Gym for robots, but the **agent is an LLM** and its **actions are text outputs**
- You define: the environment + the reward function
- The training algorithm (GRPO) is boilerplate via HF TRL or Unsloth
- Your job = **environment design + reward logic**, not algorithm implementation

**Model choices:**
- **Primary model**: SmolLM3-3B with HF TRL
- **Fallback**: Qwen 2.5 (0.5B-1.5B) if SmolLM proves difficult
- **Stretch goal**: Migrate to UnSloth for faster training (post-MVP)

**Core loop:**
```
environment.reset() → observation → LLM generates action → environment.step(action) → reward + next observation → repeat
```

---

## Key Concepts to Understand

| Term | What it means in this context |
|------|-------------------------------|
| Environment | The world the LLM interacts with (you build this) |
| Observation | What the LLM sees each turn (text, state, context) |
| Action | What the LLM outputs (text response, command, decision) |
| Reward | Signal telling the model if it did well (you define this) |
| Policy | The LLM itself — what it "decides" to do given observation |
| GRPO | The RL training algorithm you'll use — lighter than PPO, used in DeepSeek-R1 |
| Episode | One full run of the environment from reset() to done |

---

## Learning Plan

### Day 1 — Conceptual Foundation (~3 hours)

**1. RL Key Concepts (~45 min)**
- URL: https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
- Read: "Key Concepts and Terminology" section only
- Goal: understand agent/environment/reward/policy vocabulary

**2. SmolLM3 Introduction (~30 min)**
- URL: https://huggingface.co/learn/smol-course/en/unit1/1
- Goal: understand SmolLM3 architecture, hybrid reasoning, and when it reasons vs. responds directly

**3. SmolLM3 + TRL Fine-tuning (~45 min)**
- URL: https://huggingface.co/learn/smol-course/en/unit1/3
- Goal: understand SFT with TRL's SFTTrainer on SmolLM3

**4. HF TRL + OpenEnv Integration Docs (~30 min)**
- URL: https://huggingface.co/docs/trl/en/openenv
- Goal: understand `reset()` → `step()` → reward → GRPO loop

### Day 1 Evening — Hands-On (~2 hours)

**5. OpenEnv BlackJack Example (~1 hour)**
- URL: https://github.com/meta-pytorch/OpenEnv
- Look at: `examples/grpo_blackjack/`
- Clone it, read through it — this is closest to what you'll build
- Goal: understand how an environment class is structured

**6. SmolLM GRPO Fine-tuning Guide (~1 hour)**
- URL: https://huggingface.co/blog/prithivMLmods/smollm-grpo-ft
- Goal: understand GRPOConfig, GRPOTrainer, and reward function structure
- This is the most directly applicable resource for your hackathon project

### Day 2 Morning — Build Readiness (~1 hour)

**7. OpenEnv CLI + HF Spaces Deploy (~1 hour)**
- URL: https://github.com/meta-pytorch/OpenEnv (README, deploy section)
- Install: `pip install openenv-core`
- Run the echo environment locally: `pip install git+https://huggingface.co/spaces/openenv/echo_env`
- Goal: confirm you can run an environment and connect to it before hacking starts

---

## Install Checklist (do before hackathon)

```bash
pip install openenv-core
pip install git+https://huggingface.co/spaces/openenv/echo_env
pip install trl transformers datasets accelerate
# Confirm you have a HF account and can push to Spaces
```

---

## Required Deliverables (from hackathon rules)

- [ ] Environment using OpenEnv stable release 0.2.1
- [ ] Environment deployed on HF Spaces
- [ ] Minimal training script using Unsloth or HF TRL in Colab
- [ ] Training script shows reward improvement over time
- [ ] 1-minute demo video on YouTube

---

## Judging Weights (know what matters)

| Criteria | Weight | What it means for your effort |
|----------|--------|-------------------------------|
| Environment Innovation | 40% | Novel, creative, meaningful — spend most time here |
| Storytelling | 30% | Clear demo, engaging explanation |
| Training showing reward improvement | 20% | Show reward curves or before/after |
| Reward + pipeline setup | 10% | Coherent reward logic |

---

## Key Resources Summary

| Resource | URL |
|----------|-----|
| SmolLM3-3B Model | https://huggingface.co/HuggingFaceTB/SmolLM3-3B |
| Smol Course (TRL) | https://huggingface.co/learn/smol-course/en/unit1/1 |
| SmolLM GRPO Guide | https://huggingface.co/blog/prithivMLmods/smollm-grpo-ft |
| OpenEnv GitHub | https://github.com/meta-pytorch/OpenEnv |
| HF TRL OpenEnv Docs | https://huggingface.co/docs/trl/en/openenv |
| OpenEnv HF Hub | https://huggingface.co/openenv |
| Spinning Up RL Intro | https://spinningup.openai.com/en/latest/spinningup/rl_intro.html |
| Hackathon Submission | https://cerebralvalley.ai/e/openenv-hack... |

### Stretch Resources (Post-MVP)

| Resource | URL |
|----------|-----|
| Unsloth GRPO Notebooks | https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks |

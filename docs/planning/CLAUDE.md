# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a **planning and documentation repository** for the OpenEnv Hackathon SF (March 7-8, 2025). It contains no executable code yet - only planning documents.

## Project Being Planned

**ADHD Executive Function Scaffolding Environment** - An OpenEnv environment that trains SmolLM3-3B (or Qwen 2.5 as fallback) via GRPO to be a better ADHD task initiation coach. The reward function scores responses on task decomposition, single focus, energy matching, tone safety, and brevity.

## Key Documents

- `openenv-hackathon-planning.md` - Full project concept, reward function rubric (with Python implementation), technical architecture, and submission checklist
- `openenv-prep-schedule.md` - Prep timeline leading up to the hackathon
- `openenv-rl-learning-plan.md` - RL concepts and learning resources

## Technical Stack (Planned)

- **Framework**: OpenEnv 0.2.1 + HuggingFace TRL GRPOTrainer
- **Model**: SmolLM3-3B (primary), Qwen 2.5 0.5B-1.5B (fallback)
- **Stretch**: Migrate to UnSloth for faster training (post-MVP)
- **Training**: Google Colab (T4) or HF Spaces
- **Local dev**: ROCm devcontainer (FP16 only, BF16 untested)

## When Code Is Added

Install dependencies with:
```bash
uv add openenv_core trl transformers accelerate datasets
```

The environment will implement `reset()`, `step()`, and a reward function as documented in `openenv-hackathon-planning.md`.
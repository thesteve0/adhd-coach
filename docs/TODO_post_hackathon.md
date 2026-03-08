# Post-Hackathon TODO

## Task 1: Enable Reasoning for All Models in benchmark.py (Small Fix)

### Problem
In `benchmark.py`, only Qwen3-8B uses chain-of-thought reasoning (thinking mode).

Root cause per model:
- **SmolLM3-3B**: `generate_response_smollm3()` (line 256) explicitly passes `enable_thinking=False`
  to `apply_chat_template`. Change to `enable_thinking=True`.
- **Qwen3-8B**: `generate_response_hermes()` passes no thinking parameter.
  Qwen3's tokenizer defaults to thinking enabled, so it already works. Optionally add
  `enable_thinking=True` explicitly for clarity.
- **OLMo-3-7B-Instruct**: DROPPED. OLMo has a split between Instruct variants (tool calling,
  no reasoning) and Think variants (reasoning, no tool calling). No OLMo model supports both.
  Replace with a different model (see below).

### Replace OLMo: Selected Model (researched March 2026)

**OLMo 3 is still out.** AllenAI's OLMo 3 family (released Nov 2025) confirmed the same split:
- `Olmo-3-7B-Instruct` — tool calling ✅, explicit `<think>` reasoning tokens ❌
- `Olmo-3-7B-Think` — explicit reasoning ✅, tool calling ❌
No single OLMo 3 model does both. Reference: https://allenai.org/blog/olmo3

**Selected replacement: Ministral-3-8B-Reasoning-2512** (`mistralai/Ministral-3-8B-Reasoning-2512`)

Mistral released the Ministral 3 family on December 2, 2025 with three variants per size
(base, instruct, reasoning). The reasoning variants support BOTH thinking tokens AND tool calling.

- Explicit reasoning tokens: ✅ — trained with GRPO on STEM + general RL stages
- Tool calling: ✅ — natively supported; vLLM serves it with both `--reasoning-parser mistral`
  and `--tool-call-parser mistral` flags simultaneously
- License: Apache 2.0 ✅
- Open weights: ✅ — available on HuggingFace
- Different family from Qwen: ✅
- Reference: https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
- Paper: https://arxiv.org/html/2601.08584v1

**Important implementation note — Mistral tool call format:**
Mistral uses its own tool call format (not Hermes-style XML). The model must be served via
vLLM with `--tool-call-parser mistral` for structured output, OR the transformers tokenizer's
`apply_chat_template(tools=...)` may handle it natively. Need to verify which approach works
for local inference in the benchmark script and write a new `generate_response_mistral()` +
`parse_tool_calls_mistral()` handler pair.

**Final benchmark lineup:**
| Model | Size | Family | Reasoning | Tool Calling | License |
|---|---|---|---|---|---|
| SmolLM3-3B | 3B | HuggingFace | ✅ | ✅ | Apache 2.0 |
| Qwen3-8B | 8B | Alibaba | ✅ | ✅ | Apache 2.0 |
| Ministral-3-8B-Reasoning-2512 | 8B | Mistral | ✅ | ✅ | Apache 2.0 |

### Fix Checklist
- [ ] Enable SmolLM3 reasoning: `enable_thinking=False` → `enable_thinking=True` at line 256
- [ ] Fix `parse_tool_calls_smollm3()` to strip `<think>` blocks at the top of the function,
      not just in the no-tool-call branch (line 192). Currently thinking output in tool-call
      paths would leak into the parsed message.
- [ ] Add `enable_thinking=True` to `generate_response_hermes()` for Qwen3 explicitly
- [ ] Research and select OLMo replacement model
- [ ] Add new model to `MODELS` list and `MODEL_HANDLERS` map
- [ ] Bump `max_new_tokens` from 512 to **1024** in both generate functions
      (thinking output easily exceeds 512 tokens)

---

## Task 2: GRPO Training with Unsloth (Large Task)

### Goal
Train one of the benchmark models using GRPO (Group Relative Policy Optimization) with the
adhd_env as the reward function. Show that a model can improve its ADHD coaching score
through RL training. Use **Unsloth** for efficient GRPO to reduce VRAM requirements and
speed up training.

### Why Unsloth for GRPO
- Unsloth patches the training loop for memory efficiency (2-3x less VRAM than vanilla TRL)
- Provides pre-built Colab notebooks for GRPO as a starting point
- Supports 4-bit quantization during training, making 7-8B models feasible on a T4 or A10G
- Has direct GRPO support integrated with TRL's GRPOTrainer
- Unsloth notebooks: https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks

### Infrastructure

**Compute:**
- Primary training: Google Colab with A100 or L4 (CUDA required — Unsloth is CUDA-only)
- Local ROCm devcontainer: useful for developing/debugging reward function logic only,
  not for running Unsloth training
- adhd_env server: use the deployed HF Space (`https://thesteve0-adhd-env.hf.space`) —
  no need to run it locally during training

**Key Libraries:**
- `unsloth` — memory-efficient GRPO + model loading with 4-bit quant
- `trl` (GRPOTrainer) — Unsloth wraps this
- `transformers` + `accelerate`
- `vllm` — for fast rollout generation (Unsloth handles this automatically on CUDA)
- `datasets` — for prompt dataset

**Model to Train:**
- **SmolLM3-3B** is the primary candidate — smallest, fastest, tool calling format already
  understood, Unsloth likely has a compatible checkpoint
- If Unsloth supports it, Qwen3-4B is a backup (same Hermes tool calling, native reasoning)
- Verify Unsloth has a compatible model card before committing

### Architecture

```
Training Loop (Colab + Unsloth)
    |
    ├── Unsloth loads 4-bit quantized model (SmolLM3-3B or Qwen3-4B)
    |
    ├── GRPOTrainer generates group_size candidate responses per prompt
    |       Each response = model's ADHD coaching attempt with optional tool call
    |
    ├── Reward function (our custom code) scores each response:
    |       1. Parse <tool_call> XML from raw output
    |       2. POST to adhd_env /step: {"action": {"tool_calls": [...], "message": "..."}}
    |       3. Return total_score float (0.0-1.0)
    |
    └── GRPO updates model weights toward higher-scoring responses
```

### New Code Needed

**Option A: Add `training/` directory to this repo** (recommended)
```
adhd-coach/
  training/
    train_grpo.py       # Unsloth + GRPOTrainer setup, reward fn calling adhd_env
    dataset.py          # Build HF Dataset of ADHD coaching prompts for training
    config.py           # Hyperparams: lr, batch_size, group_size, max_new_tokens, etc.
    README.md           # How to run in Colab: clone repo, pip install unsloth, run script
```

The benchmark.py script stays as-is — training is additive, not a replacement.
- `benchmark.py` = evaluation (score models, no weight updates)
- `train_grpo.py` = training (update weights to improve scores)

Run benchmark before and after training to show improvement.

### Reward Function Design

The reward function passed to GRPOTrainer must:
1. Accept a list of raw model output strings (one per rollout in the group)
2. For each output: parse tool calls using existing `parse_tool_calls_smollm3()` logic
3. Call adhd_env `/step` with parsed tool_calls + message for each output
4. Return a list of floats (scores 0.0-1.0)

Episode management:
- Call `/reset` once per prompt group to get a consistent scenario+state
- Score all group_size rollouts against that same scenario+state
- This mirrors how the benchmark works, keeping evaluation consistent

### Key GRPO Hyperparameter Decisions

- `group_size`: 4-8 responses per prompt (4 is a good start; 8 gives better gradient signal)
- `learning_rate`: 1e-6 to 5e-6 (conservative — RL can destabilize the model quickly)
- `max_new_tokens`: **1024** minimum (thinking output is verbose; tool call adds more)
- `num_train_epochs`: start with 1-3 to see reward curve before committing to long run
- `torch_dtype`: `bfloat16` on CUDA A100/L4; Unsloth handles this automatically
- 4-bit quantization via Unsloth: reduces VRAM from ~12GB to ~5GB for SmolLM3-3B

### Suggested First Steps When Ready

1. Complete Task 1 (reasoning fix + OLMo replacement) and get clean baseline benchmark scores
2. Open a fresh Colab notebook, install Unsloth, load SmolLM3-3B in 4-bit
3. Write and test the reward function in isolation:
   - Manually generate 4 responses (good/bad mix)
   - Call adhd_env and print the scores
   - Verify the score range and distribution look reasonable
4. Wire reward function into GRPOTrainer via Unsloth's GRPO example as template
5. Run a 20-step smoke test — just verify reward goes up, don't wait for convergence
6. If smoke test works, run a full training run (100-500 steps)
7. Save fine-tuned model to HF Hub
8. Run `benchmark.py` comparing baseline vs fine-tuned model

### References
- Unsloth GRPO notebooks: https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks
- TRL GRPOTrainer docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
- TRL OpenEnv integration: https://huggingface.co/docs/trl/main/en/openenv
- Existing planning docs: `docs/hackathon-2026/learning/the-basics-of-rl-and-grpo.md`
- GRPO group size notes: `docs/hackathon-2026/5-choosing-grpo-group-size.md`
- Colab GPU preference notes: `docs/hackathon-2026/6-colab-gpu-preference.md`

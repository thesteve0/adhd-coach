# Colab GPU Selection Guide for GRPO Training

**Purpose**: Choose the right GPU for training SmolLM3-3B with GRPO + vLLM during the OpenEnv hackathon.

**Context**: Colab offers multiple GPU tiers with different pricing. Since we're using vLLM for generation during GRPO training, GPU choice significantly impacts iteration speed and cost-effectiveness.

---

## Memory Requirements Analysis

**SmolLM3-3B with GRPO + vLLM:**
- Model weights (BF16): ~6GB
- vLLM KV cache (12-16 generations, prefix caching): ~3-4GB
- Training overhead (gradients, optimizer with offloading): ~6-8GB
- **Total: 15-18GB comfortable range**

---

## GPU Comparison for Your Use Case

### **Tier 1: L4 (Recommended)**
**Specs:** 24GB VRAM, Ada Lovelace, 242 TFLOPS FP16, 300 GB/s bandwidth

**Pros:**
- 24GB provides excellent headroom for experimentation
- Modern architecture with great vLLM support
- Good memory bandwidth for generation workloads
- **Best cost-performance for 3B models**

**Cons:**
- Availability varies (pricing 2.4-4.8 units suggests surge pricing)
- May be greyed out during peak times

**Cost:** $1.68-$3.36 per 7-hour training run

**Verdict:** **Primary choice if available at 2.4 units/hr**

---

### **Tier 2: T4 (Budget Baseline)**
**Specs:** 16GB VRAM, Turing, 65 TFLOPS FP16, 320 GB/s bandwidth

**Pros:**
- Always available
- Cheapest option
- 16GB sufficient with gradient checkpointing
- Proven to work with SmolLM3 GRPO

**Cons:**
- Slowest generation speed (~40-50 tokens/sec vs 80-120 on L4)
- 7-hour estimate may become 10-12 hours
- Tight memory means less room for larger group sizes

**Cost:** $1.40-$2.40 per training run

**Verdict:** **Use for initial setup/debugging, then upgrade**

---

### **Tier 3: V100 (High-Bandwidth Option)**
**Specs:** 16GB or 32GB VRAM, Volta, 125 TFLOPS FP16, **900 GB/s HBM2**

**Pros:**
- **Excellent memory bandwidth** (best for vLLM generation)
- Well-tested for ML workloads
- 32GB variant gives lots of headroom
- Good vLLM support

**Cons:**
- 16GB variant is tight (verify which you'd get)
- Older architecture (2017)
- Higher cost than T4

**Cost:** $3.43 per 7-hour run

**Verdict:** **Good choice if you need faster iteration** (especially 32GB variant)

---

### **Tier 4: A100 (Overkill but Fast)**
**Specs:** 40GB VRAM, Ampere, 312 TFLOPS FP16, 1555 GB/s HBM2e

**Pros:**
- **Fastest vLLM generation** (150-200 tokens/sec)
- 40GB allows parallel experiments or huge group sizes
- Best-in-class support for all frameworks
- Could run 2-3 experiments simultaneously

**Cons:**
- Expensive for a 3B model
- 7-hour run may compress to 4-5 hours (still billed hourly)
- Diminishing returns vs V100/L4

**Cost:** $8.26-$10.50 per run

**Verdict:** **Only if you need very fast iteration or parallel experiments**

---

### **H100 (Skip)**
**Cost:** $12.60+ per run

**Verdict:** **Not cost-effective for SmolLM3-3B** (designed for 70B+ models)

---

### **TPUs (v5e-1, v6e-1) (Skip)**

**Verdict:** **vLLM has limited TPU support** — stick with CUDA GPUs

---

## Recommended Hackathon Strategy

### **Saturday (Day 1):**

**9 AM - 11 AM: Setup & Debugging (T4)**
- Get environment working, test reward function
- First training run with `num_generations=4` (fast test)
- **Cost: ~$0.40**

**11 AM - 6 PM: First Real Training Run (L4 or V100)**
- Switch to L4 (if available) or V100
- Full 10K steps, `num_generations=12`
- **Cost: $1.68-$3.43**

---

### **Sunday (Day 2):**

**Morning: Evaluation & Iteration**
- If results good: continue with L4/V100
- If need faster iteration: upgrade to A100 for hyperparameter sweeps

**Afternoon: Final Training + Demo Prep**
- Polish best model
- Create demo materials

**Budget: $10-15 total** should cover all experimentation

---

## Quick Reference Table

| GPU | VRAM | Cost/7hrs | Gen Speed | Best For | Availability |
|-----|------|-----------|-----------|----------|--------------|
| **T4** | 16GB | $1.40 | Slow (40-50 tok/s) | Setup, debugging | ✅ Always |
| **L4** | 24GB | **$1.68** | Fast (80-120 tok/s) | **Main training** | ⚠️ Variable |
| **V100** | 16-32GB | $3.43 | Good (60-90 tok/s) | Fast iteration | ⚠️ Scarce |
| **A100** | 40GB | $8.26 | Fastest (150-200 tok/s) | Parallel experiments | ⚠️ Scarce |
| **H100** | 80GB | $12.60+ | Overkill | Skip | ⚠️ Rare |

---

## Final Recommendation

### **Primary Strategy:**
1. **Start with T4** to validate your setup (~1-2 hours)
2. **Switch to L4** for main training if available at 2.4 units/hr
3. **Fallback to V100** if L4 unavailable or priced at 4.8 units/hr
4. **Consider A100** only if you're running multiple experiments in parallel on Day 2

### **Cost Estimate:**
- **Conservative** (T4 → V100): ~$5-6 total
- **Optimal** (T4 → L4): ~$3-4 total
- **Aggressive** (T4 → A100): ~$10-12 total

### **Memory Configuration:**

All GPUs listed have enough VRAM for SmolLM3-3B with `num_generations=12-16`. The T4's 16GB is tight but workable with:

```python
# T4 configuration (16GB)
config = GRPOConfig(
    num_generations=12,
    per_device_train_batch_size=1,  # Use 1 on T4
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    max_new_tokens=100,
    learning_rate=3e-6,
    kl_coef=0.001,
)

# L4/V100/A100 configuration (24GB+)
config = GRPOConfig(
    num_generations=12,
    per_device_train_batch_size=2,  # Use 2 on larger GPUs
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    max_new_tokens=100,
    learning_rate=3e-6,
    kl_coef=0.001,
)
```

---

## TL;DR

**L4 at 2.4 units/hr is your sweet spot. Start with T4, upgrade once your code works.**

- **T4**: Setup and debugging (always available, cheap)
- **L4**: Main training runs (best value if available)
- **V100**: Fallback if L4 unavailable
- **A100**: Only for aggressive iteration or parallel experiments
- **H100/TPU**: Skip for this project

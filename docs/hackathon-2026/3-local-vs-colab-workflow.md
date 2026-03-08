# Local Development vs Colab Training Workflow

**Purpose**: Develop your OpenEnv environment and training code locally as Python scripts, then run training in Colab with minimal changes.

**Key principle**: Avoid notebooks for development. Structure everything as importable Python modules that work locally and in Colab.

---

## Local Laptop (ROCm Devcontainer) vs Colab

### **Develop Locally:**

1. **Environment logic** (100% local development)
   - `reset()`, `step()`, reward function
   - State management, scenario generation
   - Run with `uvicorn` for testing
   - **No GPU needed** - pure Python logic

2. **Training script skeleton** (develop structure locally)
   - GRPOConfig setup
   - Custom rollout function
   - Reward extraction logic
   - Data loading/preprocessing
   - **Test imports and structure** (don't run actual training)

3. **Evaluation scripts** (develop locally, run in Colab)
   - Model inference testing
   - Reward curve plotting
   - Response quality analysis

4. **Utilities** (develop and test locally)
   - Logging, checkpoint management
   - Config file parsing
   - Helper functions

### **Run in Colab Only:**

1. **Actual GRPO training** (requires GPU + vLLM)
2. **Model loading** (HF Hub downloads are faster on Colab)
3. **Final evaluation runs** (if you need vLLM inference speed)

---

## Recommended Project Structure

```
openenv-hackathon-prep/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── custom_env.py         # Environment class (develop locally)
│   │   ├── scenarios.py          # Scenario generation (develop locally)
│   │   └── reward.py             # Reward function (develop locally)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py             # GRPOConfig (develop locally, run in Colab)
│   │   ├── rollout.py            # Custom rollout function (develop locally)
│   │   └── train.py              # Main training script (develop locally, run in Colab)
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate.py           # Evaluation logic (develop locally, run in Colab)
│       └── plotting.py           # Visualization (run anywhere)
│
├── scripts/
│   ├── test_environment_local.py # Test env with uvicorn (run locally)
│   ├── train_colab.py            # Entry point for Colab (run in Colab)
│   └── evaluate_model.py         # Model evaluation (run in Colab)
│
├── configs/
│   └── grpo_config.yaml          # Hyperparameters (shared)
│
├── pyproject.toml                # Dependencies (shared)
└── README.md
```

---

## Local Development Workflow

### **Step 1: Develop Environment Locally**

```bash
# In devcontainer terminal
cd src/environment

# Edit custom_env.py, reward.py, scenarios.py
# Test the server locally
python -m uvicorn custom_env:app --host 0.0.0.0 --port 8001

# In another terminal, test the client
python scripts/test_environment_local.py
```

**scripts/test_environment_local.py:**
```python
from src.environment.custom_env import CustomEnv, CustomAction

# Test locally
client = CustomEnv(base_url="http://localhost:8001")
result = client.reset()
print(f"Initial observation: {result.observation}")

# Test some responses
test_responses = [
    "Response that should score low",
    "Response that should score high",
]

for response in test_responses:
    step_result = client.step(CustomAction(message=response))
    print(f"Response: {response}")
    print(f"Reward: {step_result.reward}\n")
```

### **Step 2: Develop Training Script Locally (Don't Run)**

```bash
# Edit src/training/train.py
# Test imports and structure (don't run training)
python -c "from src.training.train import create_trainer; print('Imports OK')"
```

**src/training/train.py:**
```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.environment.custom_env import CustomEnv, CustomAction

def create_config(use_vllm=True, num_generations=12):
    """Create GRPO config - can tweak locally without running"""
    return GRPOConfig(
        output_dir="./results",
        num_generations=num_generations,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_new_tokens=100,
        learning_rate=3e-6,
        num_train_epochs=1,
        max_steps=10000,
        use_vllm=use_vllm,
        logging_steps=10,
        save_steps=500,
    )

def create_trainer(model_name="HuggingFaceTB/SmolLM3-3B", env_url=None):
    """Create trainer - develop locally, run in Colab"""
    config = create_config()

    # Load model (will only work in Colab with GPU)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Environment client
    env = CustomEnv(base_url=env_url)

    # Custom rollout function
    def rollout_func(prompts, trainer):
        # Your rollout logic here
        pass

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        rollout_func=rollout_func,
    )

    return trainer

def main():
    """Entry point - run this in Colab"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B")
    args = parser.parse_args()

    trainer = create_trainer(args.model, args.env_url)
    trainer.train()
    trainer.save_model("./final_model")

if __name__ == "__main__":
    main()
```

### **Step 3: Push to GitHub**

```bash
git add src/ scripts/ configs/
git commit -m "Add training script and environment"
git push
```

---

## Running in Colab (Without Notebooks!)

### **Option 1: Clone Repo and Run Script**

Create a minimal notebook in Colab with just:

```python
# Cell 1: Setup
!git clone https://github.com/yourusername/openenv-hackathon-prep.git
%cd openenv-hackathon-prep

!pip install -e .  # Installs from pyproject.toml

# Cell 2: Run training script
!python src/training/train.py \
    --env-url https://your-username-custom-env.hf.space \
    --model HuggingFaceTB/SmolLM3-3B
```

### **Option 2: Use Colab's Code Execution Feature**

Colab can run `.py` files directly:

1. Upload `src/training/train.py` to Colab files panel
2. Run: `!python train.py --env-url ...`

### **Option 3: Develop Locally, Copy-Paste Code Cells**

Structure your script with clear sections:

```python
# ===== SETUP =====
# (Copy this to Colab cell 1)

# ===== TRAINING =====
# (Copy this to Colab cell 2)

# ===== EVALUATION =====
# (Copy this to Colab cell 3)
```

---

## Sync Strategy: Git-Based Workflow

### **Local Development Loop:**

```bash
# 1. Make changes locally
vim src/environment/reward.py

# 2. Test locally
python scripts/test_environment_local.py

# 3. Commit and push
git add src/environment/reward.py
git commit -m "Improve reward function"
git push

# 4. In Colab, pull changes
# !git pull origin main
# !python src/training/train.py --env-url ...
```

### **Quick Iteration:**

For rapid changes during the hackathon:

```bash
# Local: Edit code
# Then in Colab:
!git pull && python src/training/train.py --env-url ...
```

---

## Recommended pyproject.toml

```toml
[project]
name = "openenv-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "openenv-core>=0.2.1",
    "trl>=0.8.0",
    "transformers>=4.40.0",
    "accelerate>=0.28.0",
    "torch>=2.0.0",
    "pydantic>=2.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
# Protect ROCm installation locally
override-dependencies = [
    "torch==2.9.1+rocm6.2",
]
```

Then in Colab:
```bash
pip install -e .  # Installs all dependencies
```

---

## What to Test Locally vs Colab

| Component | Local Test | Colab Test |
|-----------|------------|------------|
| Environment logic | ✅ Full testing with uvicorn | ❌ Not needed |
| Reward function | ✅ Unit tests with hardcoded responses | ✅ Validate with model |
| Training imports | ✅ Check imports work | ❌ Not needed |
| Model loading | ❌ Skip (no vLLM on ROCm) | ✅ Full test |
| GRPO training | ❌ Skip (no GPU) | ✅ Full runs |
| Evaluation plots | ✅ Can test logic | ✅ Run on real results |

---

## Example: Testing Everything Locally First

```bash
# 1. Test environment
python -m uvicorn src.environment.custom_env:app --reload

# 2. In another terminal, test client
python scripts/test_environment_local.py

# 3. Test that training script imports work
python -c "from src.training.train import create_config; print(create_config())"

# 4. Push to git
git add . && git commit -m "Ready for Colab" && git push

# 5. In Colab: git clone + run
```

---

## Echo Environment Example (Pre-Hackathon Testing)

You can practice this workflow with `echo_env` before the hackathon:

```bash
# Install echo_env locally
uv add "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"

# Test with remote HF Space
python
>>> from echo_env import EchoEnv, EchoAction
>>> client = EchoEnv(base_url="https://openenv-echo-env.hf.space")
>>> result = client.reset()
>>> print(result.observation)

# Or run server locally
python -m uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8001

# Then test with local URL
>>> client = EchoEnv(base_url="http://localhost:8001")
```

Replace `EchoEnv` with your custom environment during the hackathon.

---

## TL;DR

**Local (Laptop):**
- Develop environment logic (100% functional locally)
- Write training scripts as `.py` files (structure only)
- Test imports, reward function, scenarios
- Use uvicorn to validate environment
- Push to GitHub

**Colab:**
- Clone repo
- Install dependencies (`pip install -e .`)
- Run training script: `python src/training/train.py --env-url ...`
- Pull updates with `git pull` during iteration

**Key insight:** Structure everything as importable Python modules, not notebook cells. Colab can run `.py` files just fine—you just need one minimal notebook to clone and execute.

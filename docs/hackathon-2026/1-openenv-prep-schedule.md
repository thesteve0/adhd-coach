# OpenEnv Hackathon Prep Schedule
**Hackathon**: March 7-8, Shack15, San Francisco
**Coding deadline**: Sunday March 8, 1:00 PM

---

## Thursday Night (10:30 PM → Midnight)

- [ ] Read [SmolLM3 Introduction](https://huggingface.co/learn/smol-course/en/unit1/1) — understand the model you'll be training (~20 min)
- [ ] Read [HF TRL + OpenEnv integration docs](https://huggingface.co/docs/trl/en/openenv) — focus on the Wordle example end-to-end (~30 min)
- [ ] Get ROCm devcontainer running if it's straightforward — stop by 11:30 if hitting walls

---

## Friday Morning (9 AM → Noon)

- [ ] Read [Spinning Up RL — Key Concepts and Terminology](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) (~45 min, this section only)
- [ ] Read [SmolLM GRPO Fine-tuning Guide](https://huggingface.co/blog/prithivMLmods/smollm-grpo-ft) — this covers GRPOConfig, GRPOTrainer, and reward functions (~45 min)
- [ ] Skim [SmolLM3 SFT with TRL](https://huggingface.co/learn/smol-course/en/unit1/3) — understand the training script structure (~20 min)
- [ ] Read (don't run) [BlackJack example code](https://github.com/meta-pytorch/OpenEnv) in `examples/grpo_blackjack/` — focus on the structure of `reset()` and `step()`

---

## Friday Afternoon (1 PM → 5 PM)

- [ ] Install and run echo_env locally:
  ```bash
  # Install packages (using uv in devcontainer)
  uv add "openenv-echo-env @ git+https://huggingface.co/spaces/openenv/echo_env"

  # This installs both openenv-core 0.2.1 and openenv-echo-env
  # Import name is: from echo_env import EchoEnv, EchoAction
  ```

- [ ] Test echo_env locally (three options):

  **Option 1: Connect to remote HF Space (easiest, no setup)**
  ```python
  from echo_env import EchoEnv, EchoAction

  client = EchoEnv(base_url="https://openenv-echo-env.hf.space")
  result = client.reset()
  print(result.observation)

  step_result = client.step(EchoAction(message="Hello!"))
  print(f"Reward: {step_result.reward}")
  ```

  **Option 2: Run local server (for testing server deployment)**
  ```bash
  # Terminal 1: Start server
  python -m uvicorn echo_env.server.app:app --host 0.0.0.0 --port 8001

  # Terminal 2: Test with client
  # Use EchoEnv(base_url="http://0.0.0.0:8001")
  ```

  **Key API notes:**
  - `reset()` returns `StepResult` with `.observation`, `.reward`, `.done`
  - `step(action)` returns `StepResult` with same structure
  - Actions are environment-specific: `EchoAction(message="text")`

  **Understanding Local vs Hackathon Deployment:**
  - **Local uvicorn** (what you just did): Tests your environment logic
  - **HF Spaces** (hackathon): Runs uvicorn automatically, gives you a public URL
  - **Colab training**: Connects to your HF Space URL, not localhost

  ```
  Local Dev:          devcontainer → http://0.0.0.0:8001 (testing)
  Hackathon:          HF Space → https://your-env.hf.space → Colab GRPO trainer
  ```

- [ ] Deploy echo_env to [HF Spaces](https://huggingface.co/spaces) — debug until it works
- [ ] Connect a Colab notebook to your running echo_env Space and confirm the connection works end-to-end

---

## Friday Evening (5 PM → 7 PM)

- [ ] Run your reward function rubric against 4-5 hardcoded responses locally — confirm scoring makes sense
- [ ] Pack for SF
- [ ] Early bed

---

## Saturday Drive (7:30 AM → 9:00 AM)

- [ ] Audio: Search Spotify/YouTube for "Latent Space podcast GRPO" or play Yannic Kilcher GRPO video audio-only
- [ ] Arrive Shack15 — you already know how HF Spaces works 🎯

---

## Key Resources

| Resource | URL |
|----------|-----|
| SmolLM3-3B Model | https://huggingface.co/HuggingFaceTB/SmolLM3-3B |
| Smol Course | https://huggingface.co/learn/smol-course/en/unit1/1 |
| SmolLM GRPO Guide | https://huggingface.co/blog/prithivMLmods/smollm-grpo-ft |
| OpenEnv GitHub | https://github.com/meta-pytorch/OpenEnv |
| HF TRL OpenEnv Docs | https://huggingface.co/docs/trl/en/openenv |
| HF Spaces | https://huggingface.co/spaces |
| Hackathon Submission | https://cerebralvalley.ai/e/openenv-hack |

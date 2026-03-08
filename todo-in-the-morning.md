# Morning TODO

## Blocked: HF Spaces Deployment
- Space is flaky — builds succeed but container often fails to start silently
- Questions for HF team are in `QUESTIONS_FOR_HF.md`
- Continue iterating locally until we get answers
- Last working deploy was v2 (state tracking + rubric scoring confirmed working on HF)

## Ready to Test: LLM Leaderboard Script
- `test_with_model.py` is written but untested
- Needs `export HF_TOKEN=hf_...` then: `.venv/bin/python test_with_model.py`
- Tests 3 models (SmolLM3-3B, Qwen3.5-9B, OLMo-3-7B) against 4 scenarios
- Runs against local environment (no HF Space needed)
- Install openai first: `.venv/bin/pip install openai` or `uv sync --extra dev`
- Run single model: `.venv/bin/python test_with_model.py`
- Run leaderboard: `.venv/bin/python test_with_model.py --all`

## TODO: "Movement First" Rubric Criterion
- Brainstorm is in `TODO_movement_first_rubric.md`
- When user is slouching / sitting long / late evening, reward responses that prioritize body movement BEFORE task work
- First-cut keyword approach is sketched out, ready to implement
- Would be a new scoring function in `reward.py` with heavy weight

## TODO: Improve Test Coverage
- Add HTTP tests against local server for V2 scoring
- Test non-ADHD scenarios through the full environment (not just rubric unit tests)
- Test edge cases: empty messages, multiple tool calls, unknown tools

## Local Dev Commands
```bash
# Run tests
cd /workspaces/adhd-coach/adhd_env && .venv/bin/python test_environment.py

# Start local server
.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Test against local server
.venv/bin/python test_environment.py --http

# Deploy (once HF is sorted out)
.venv/bin/openenv push --repo-id TheSteve0/adhd-env --no-interface
```

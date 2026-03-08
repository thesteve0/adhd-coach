# TODO: "Movement First" Rubric Criterion

## The Ideal Response Pattern

When user state indicates physical distress (slouching, long sitting, late evening),
the OPTIMAL coaching response prioritizes body movement BEFORE task work:

> "Before you do anything else, get up and move your body. Maybe get a drink of water,
> go outside and touch grass, go split some wood. When you come back I will have some
> questions for you."

Then it calls `adhd_coach_tool` to set up the follow-up interaction for when they return.

This should be rewarded very heavily — it's the best possible ADHD coaching response
for these states.

## State Triggers (any of these)

- `position_in_chair == "slouching"`
- `minutes_since_last_stood >= 60`
- Late evening: hour >= 20

## What Makes This Response Pattern Unique

It has THREE parts, all present together:
1. **Movement-first priority** — "before anything else", "first", "before you start"
2. **Physical activity suggestions** — water, outside, walk, fresh air, move your body, stretch
3. **Promise of return/continuation** — "when you come back", "after that we'll", "then I'll help"

Plus: calls `adhd_coach_tool` (to prepare the follow-up)

## Brainstorm: First-Cut Keyword Approach

Could score this without an LLM judge by checking for all 3 categories:

```python
def score_movement_first(action, user_state) -> float:
    """Heavy bonus when response prioritizes movement before task work."""
    # Only triggers when state warrants it
    needs_movement = (
        user_state.get("position_in_chair") == "slouching"
        or user_state.get("minutes_since_last_stood", 0) >= 60
        or int(user_state.get("time_of_day", "12:00").split(":")[0]) >= 20
    )
    if not needs_movement:
        return 0.0  # not applicable, no bonus

    msg = action.message.lower()

    # Category 1: Prioritizes movement BEFORE task
    priority_words = ["before", "first", "before anything", "step away", "stop"]

    # Category 2: Physical activity
    activity_words = ["water", "outside", "walk", "fresh air", "move", "body",
                      "stretch", "drink", "grass", "sunshine", "exercise"]

    # Category 3: Promise to continue after
    return_words = ["come back", "when you return", "after that", "then we",
                    "then i", "ready", "waiting", "here for you"]

    has_priority = any(w in msg for w in priority_words)
    has_activity = any(w in msg for w in activity_words)
    has_return = any(w in msg for w in return_words)

    if has_priority and has_activity and has_return:
        return 1.0   # full bonus — all 3 parts present
    elif has_priority and has_activity:
        return 0.6   # good — movement first but no return promise
    elif has_activity:
        return 0.3   # mentions movement but doesn't prioritize it
    return 0.0
```

### How to integrate into rubric weights

Option A: Add as 4th criterion, rebalance weights:
- tool_calling: 30%, state_awareness: 20%, adhd_relevance: 20%, movement_first: 30%

Option B: Make it a multiplier/bonus on top of existing score:
- If movement_first triggers fully, multiply final score by 1.5 (before clamp)

Option C: Replace state_awareness with this (it's a superset):
- This IS state awareness, just the most important kind

## Limitations of Keyword Approach

- Can't tell if the response ACTUALLY prioritizes movement vs just mentioning it
- "Don't walk away from your task" would false-positive on "walk" and "away"
- Can't evaluate the QUALITY or tone of the suggestion
- Can't tell if the return promise is genuine coaching setup vs throwaway

## Future: LLM-as-Judge

For a proper implementation, we'd want an LLM judge that evaluates:
1. Does the response prioritize physical wellbeing over task completion?
2. Does it give concrete physical activity suggestions (not just "take a break")?
3. Does it promise meaningful follow-up (not just dismissing the user)?
4. Is the tone encouraging rather than prescriptive?

Could use a small model (SmolLM3-3B) as judge with a rubric prompt.
Trade-off: slower scoring, but much more accurate for this nuanced criterion.

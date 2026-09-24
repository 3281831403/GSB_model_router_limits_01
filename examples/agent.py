"""Agent patterns: route every LLM call an agent makes, then call the chosen model via OpenRouter.

Run from the repo root: python3 examples/agent.py
"""

import os

from basic import load_env

from model_router import Limits, NoModelFitsError, Router, RouterError
from model_router._http import request_json

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = ["anthropic/claude-opus-5.5", "openai/gpt-6-luna", "google/gemini-3.5-flash-lite"]
FALLBACK = "openai/gpt-6-luna"  # used when Jev is unreachable or rate limited


def chat(api_key, model, messages, max_tokens):
    res = request_json(CHAT_URL, api_key, {"model": model, "messages": messages, "max_tokens": max_tokens})
    choice = res["choices"][0]
    # Reasoning models can spend the whole max_tokens thinking and return no content.
    return (
        choice["message"].get("content") or f"<empty reply, finish_reason={choice.get('finish_reason')}>"
    ).strip()


def pick(router, prompt, limits):
    """Route, but never let a Jev outage stall the agent."""
    try:
        return router.route(prompt, limits)
    except NoModelFitsError:
        raise  # no model can hold this input: the agent must shrink it, a fallback won't help
    except RouterError as e:
        print(f"  [router unavailable, using {FALLBACK}: {e}]")
        return FALLBACK


def plan_and_execute(router, api_key, goal, budget_usd=0.05):
    """Example 1: planner -> per-subtask workers -> summarizer, each step routed separately,
    with a total cost budget spread across the remaining steps."""
    print(f"\n== Plan and execute: {goal}")
    spent = 0.0
    by_id = {m.id: m for m in router.models}

    def step(prompt, output_tokens, steps_left):
        nonlocal spent
        limits = Limits(output_tokens=output_tokens, max_cost_usd=(budget_usd - spent) / steps_left)
        model = pick(router, prompt, limits)
        answer = chat(api_key, model, [{"role": "user", "content": prompt}], output_tokens)
        if model in by_id:  # ponytail: estimated cost; read usage from the response if you need exact spend
            spent += by_id[model].cost(len(prompt) // 4 + 1, output_tokens)
        return model, answer

    model, plan = step(
        f"Break this goal into exactly 3 short subtasks, one per line, no numbering:\n{goal}",
        200,
        steps_left=5,
    )
    subtasks = [s.strip("-• ").strip() for s in plan.splitlines() if s.strip()][:3]
    print(f"  plan    [{model}]")

    results = []
    for i, sub in enumerate(subtasks):
        model, out = step(f"Goal: {goal}\nDo this subtask concisely:\n{sub}", 400, steps_left=4 - i)
        results.append(out)
        print(f"  subtask [{model}] {sub}")

    model, summary = step(
        f"Combine these results into a short final answer for the goal '{goal}':\n\n" + "\n\n".join(results),
        300,
        steps_left=1,
    )
    print(f"  summary [{model}]\n\n{summary}\n\n  estimated spend ${spent:.4f} of ${budget_usd}")


def chat_session(router, api_key, turns):
    """Example 2: multi-turn chat. Route on the whole transcript, so the context limit
    tracks the growing history and a harder turn can switch to a stronger model."""
    print("\n== Chat session")
    messages = []
    for user_msg in turns:
        messages.append({"role": "user", "content": user_msg})
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        model = pick(router, transcript, Limits(output_tokens=2000))
        reply = chat(api_key, model, messages, 2000)
        messages.append({"role": "assistant", "content": reply})
        print(f"  [{model}] user: {user_msg}\n    -> {reply[:200]}")


def main():
    load_env()
    api_key = os.environ["OPENROUTER_API_KEY"]  # this demo also calls the chosen models through OpenRouter
    router = Router(openrouter_api_key=api_key, models=MODELS)

    plan_and_execute(
        router, api_key, "Write a launch announcement for a CLI tool that routes prompts to the best LLM"
    )
    chat_session(
        router,
        api_key,
        [
            "Hi! What's the capital of Australia?",
            "Now write a Python function that finds the longest palindromic substring "
            "in O(n) time and explain it.",
        ],
    )


if __name__ == "__main__":
    main()

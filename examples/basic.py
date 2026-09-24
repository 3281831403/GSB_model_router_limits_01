import os
from pathlib import Path

from model_router import Limits, NoModelFitsError, Router


def load_env(path=".env"):
    # ponytail: minimal .env parser, swap for python-dotenv if you need quoting/export syntax
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def routing_key():
    """The library takes keys as arguments; this demo just reads them from .env for convenience."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return {"openrouter_api_key": os.environ["OPENROUTER_API_KEY"]}
    jev = os.environ.get("JEV_API_KEY") or os.environ.get("jev_api_key")
    if jev:
        return {"jev_api_key": jev}
    raise SystemExit("Set OPENROUTER_API_KEY or JEV_API_KEY in .env")


def main():
    load_env()

    # Provider names only: the router picks each provider's newest models.
    by_provider = Router(**routing_key(), providers=["openai", "anthropic"], models_per_provider=3)
    print("candidates from providers:", [m.id for m in by_provider.models])

    # Exact models.
    router = Router(
        **routing_key(),
        models=["anthropic/claude-opus-5.5", "openai/gpt-6-luna", "google/gemini-3.5-flash-lite"],
    )
    for task in [
        "Translate 'good morning' into French.",
        "Design a lock-free concurrent hash map in Rust and prove it is linearizable.",
    ]:
        print(router.route(task), "<-", task)

    print(router.route("Summarize this paragraph.", Limits(max_cost_usd=0.001)), "<- cost-capped")

    try:
        router.route("x" * 5_000_000)
    except NoModelFitsError as e:
        print("no fit:", e)


if __name__ == "__main__":
    main()

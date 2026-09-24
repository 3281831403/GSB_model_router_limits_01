from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_length: int
    max_output_tokens: int | None  # None = provider sets no separate cap
    prompt_price: float  # USD per input token
    completion_price: float  # USD per output token
    description: str = ""
    created: int = 0  # unix time the model was listed; used to order a provider's models newest first
    expires: str | None = None  # retirement date if OpenRouter has scheduled one
    context_length_known: bool = True  # False = upstream gave no context_length; 0 is a placeholder, not a real limit

    def cost(self, in_tokens, out_tokens):
        return in_tokens * self.prompt_price + out_tokens * self.completion_price

    @classmethod
    def from_openrouter(cls, raw):
        pricing = raw.get("pricing") or {}
        top = raw.get("top_provider") or {}
        context_known = raw.get("context_length") is not None
        return cls(
            id=raw["id"],
            context_length=int(raw.get("context_length") or 0),
            max_output_tokens=top.get("max_completion_tokens"),
            # OpenRouter uses negative prices for "variable"; treat as unknown (0) rather than a discount.
            prompt_price=max(float(pricing.get("prompt") or 0), 0.0),
            completion_price=max(float(pricing.get("completion") or 0), 0.0),
            description=(raw.get("description") or "").replace("\n", " "),
            created=int(raw.get("created") or 0),
            expires=raw.get("expiration_date"),
            context_length_known=context_known,
        )


@dataclass(frozen=True)
class Limits:
    output_tokens: int = 1024
    max_cost_usd: float | None = None

    def __post_init__(self):
        if self.output_tokens <= 0:
            raise ValueError(
                f"output_tokens must be a positive integer, got {self.output_tokens!r}"
            )
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError(
                f"max_cost_usd must be None or >= 0, got {self.max_cost_usd!r}"
            )

"""List prices for the hosted services, used to estimate what each step cost.

These are estimates from published rates, not invoices: check the Anthropic and fal.ai
dashboards for what was actually billed. Models that run on this Mac cost nothing."""

from __future__ import annotations

from .config import settings
from .hosted import is_hosted

# US dollars per million tokens (input, output), matched by model-id prefix.
# Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-11-19).
CLAUDE_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4": (5.00, 25.00),
    "claude-fable-5-2": (8.00, 40.00),
    "claude-fable-5": (10.00, 50.00),
}
# Haiku 5.5 charges more for prompts over 100K tokens.
HAIKU_LONG_PROMPT = (100_000, (0.50, 2.50))

# US dollars per megapixel of output. Source: each model's page on fal.ai (checked 2026-11-19).
FAL_PRICES: dict[str, float] = {
    "fal-ai/flux-2/klein/4b": 0.005,
    "fal-ai/flux-2/klein/4b/base": 0.009,
    "fal-ai/qwen-image-edit-2511": 0.03,
}


def claude_rates(model: str, input_tokens: int = 0) -> tuple[float, float] | None:
    """(input, output) dollars per million tokens, or None for a model with no known price."""
    if model.startswith("claude-haiku-5-5") and input_tokens > HAIKU_LONG_PROMPT[0]:
        return HAIKU_LONG_PROMPT[1]
    # Longest prefix first, so claude-sonnet-5-5 is not priced as claude-sonnet-5.
    for prefix in sorted(CLAUDE_PRICES, key=len, reverse=True):
        if model.startswith(prefix):
            return CLAUDE_PRICES[prefix]
    return None


def llm_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    """One Claude call. `usd` is None when the model's price is not in the table."""
    rates = claude_rates(model, input_tokens)
    usd = None if rates is None else (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
    return {"model": model, "input_tokens": input_tokens, "output_tokens": output_tokens, "usd": usd}


def render_model(is_edit: bool) -> str:
    """The model that draws a new image, or edits an existing one."""
    if settings.renderer == "mock":
        return "mock"
    return (settings.edit_model or settings.image_model) if is_edit else settings.image_model


def image_cost(model: str, width: int, height: int) -> dict:
    """One render. Local models are free; `usd` is None for a hosted model with no known price."""
    megapixels = width * height / 1_000_000
    if not is_hosted(model):
        return {"model": model, "where": "local", "megapixels": megapixels, "usd": 0.0}
    rate = FAL_PRICES.get(model)
    usd = None if rate is None else megapixels * rate
    return {"model": model, "where": "fal.ai", "megapixels": megapixels, "usd": usd}


def rates() -> dict:
    """The rates in force for the current settings, for the cost panel."""
    claude = claude_rates(settings.claude_model)

    def image(is_edit: bool) -> dict:
        model = render_model(is_edit)
        hosted = is_hosted(model)
        return {
            "model": model,
            "where": "fal.ai" if hosted else "local",
            "usd_per_megapixel": FAL_PRICES.get(model) if hosted else 0.0,
        }

    return {
        "llm": {
            "model": settings.claude_model,
            "usd_per_mtok_input": claude[0] if claude else None,
            "usd_per_mtok_output": claude[1] if claude else None,
        },
        "generate": image(False),
        "edit": image(True),
    }

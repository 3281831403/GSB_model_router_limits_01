class RouterError(Exception):
    """Base error for the model router."""


class UnknownModelError(RouterError):
    """A configured model id is not in the OpenRouter catalog."""


class NoModelFitsError(RouterError):
    """Every configured model is ruled out by the task's limits."""

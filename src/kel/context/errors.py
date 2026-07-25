from kel.models.errors import KelError


class LoopError(KelError):
    """Base class for Loop guardrail failures."""


class LoopBudgetExceededError(LoopError):
    """Raised when a loop exceeds its max_iterations or max_wall_seconds."""


class StuckLoopError(LoopError):
    """Raised when the same action signature repeats enough times in a
    recent window to look like no-progress rather than legitimate retry."""

    def __init__(self, message: str, *, signature: str, repeat_count: int):
        super().__init__(message)
        self.signature = signature
        self.repeat_count = repeat_count

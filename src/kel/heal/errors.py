from kel.heal.types import HealAttempt
from kel.models.errors import KelError


class HealExhaustedError(KelError):
    """Raised when healing gives up — either the diagnosis escalated to a
    human, or max_attempts ran out. Always carries the full diagnostic
    trail (DESIGN.md 3.15: "fail loudly ... never silently")."""

    def __init__(self, message: str, *, attempts: list[HealAttempt]):
        super().__init__(message)
        self.attempts = attempts

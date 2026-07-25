from kel.models.errors import KelError


class BudgetExceededError(KelError):
    """Raised when a call would exceed (or already put the tracker over) a
    Budget limit. `dimension` identifies which limit tripped, so callers
    (e.g. kel.heal) can decide policy without parsing the message string."""

    def __init__(self, message: str, *, dimension: str):
        super().__init__(message)
        self.dimension = dimension

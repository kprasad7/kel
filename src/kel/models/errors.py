"""Provider-agnostic error hierarchy. Adapters translate provider SDK
exceptions into these at the boundary, so callers never need to catch
per-provider exception types."""


class KelError(Exception):
    """Base class for all kel errors."""


class ProviderError(KelError):
    """A provider call failed. Wraps the original exception in __cause__."""

    def __init__(self, message: str, *, provider: str, retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class AuthenticationError(ProviderError):
    def __init__(self, message: str, *, provider: str):
        super().__init__(message, provider=provider, retryable=False)


class RateLimitError(ProviderError):
    def __init__(self, message: str, *, provider: str, retry_after: float | None = None):
        super().__init__(message, provider=provider, retryable=True)
        self.retry_after = retry_after


class ModelNotFoundError(KelError):
    """Raised by the registry when a model spec doesn't resolve to a known provider/model."""

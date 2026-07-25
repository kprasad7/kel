from kel.models.errors import KelError


class EmptyModelResponseError(KelError):
    """Raised when a model turn comes back with no content at all (no
    text, no tool calls) and isn't a tool-use turn — e.g. truncated by a
    provider-side error or a response that got cut off mid-stream.

    Agent refuses to store a turn like this into shared memory: an empty
    assistant message left in `Message` history doesn't just look wrong,
    it corrupts every later turn in the same session (some providers
    reject a request whose history contains an empty assistant message
    outright; others just get confused by it), and `Agent.memory` persists
    across every `run()` call with no automatic reset. Raising here forces
    the caller to see the failure immediately, on the turn that actually
    caused it, instead of a much more confusing failure two questions
    later.
    """

    def __init__(self, message: str, *, stop_reason: str):
        super().__init__(message)
        self.stop_reason = stop_reason

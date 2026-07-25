"""File-backed implementation of kel.runtime.checkpoint.CheckpointStore —
the durable counterpart to InMemoryCheckpointStore, so a graph run's state
survives a process restart.

Uses pickle, not JSON: Checkpoint.state can hold arbitrary Python/pydantic
objects (Message, ModelResponse, ...), and forcing a JSON-safe schema for
every possible node's state would be real, unneeded work for v1. Known
tradeoff: Python-only, not a cross-language artifact format. Also known:
this rewrites the whole per-run file on every save, which is fine for
local dev / moderate-length runs and not optimized for very long-running
graphs — a future version could append-only instead.

Security note: plain `pickle.loads` on untrusted data is a known
deserialization-of-untrusted-data RCE vector — the same vulnerability
class behind several disclosed CVEs in LLM orchestration frameworks that
pickle agent state by default. Loading here goes through a restricted
unpickler (`_RestrictedUnpickler`) that only permits
kel's own types plus a small builtin allowlist, so a tampered/malicious
checkpoint file can't smuggle in an arbitrary class to instantiate. This
does not make loading checkpoints from a truly untrusted source safe in
general — treat checkpoint files the way you'd treat any other local
application data, not as a sandboxed format."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

from kel.runtime.checkpoint import Checkpoint

_ALLOWED_MODULE_PREFIXES = ("kel.", "pydantic", "pydantic_core")
_ALLOWED_BUILTINS = {
    "builtins": {
        "dict",
        "list",
        "tuple",
        "set",
        "frozenset",
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "object",
    },
    "collections": {"OrderedDict", "defaultdict"},
    "datetime": {"datetime", "date", "time", "timedelta", "timezone"},
    "uuid": {"UUID"},
    "enum": {"Enum", "EnumMeta"},
}


class _RestrictedUnpickler(pickle.Unpickler):
    """Only allows constructing kel's own classes (Checkpoint, Message,
    ModelResponse, ...) and a small set of safe builtins/stdlib types —
    rejects everything else, so a malicious checkpoint file can't smuggle
    in an arbitrary class to instantiate on load."""

    def find_class(self, module: str, name: str):
        if module.startswith(_ALLOWED_MODULE_PREFIXES):
            return super().find_class(module, name)
        if name in _ALLOWED_BUILTINS.get(module, ()):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"refusing to unpickle {module}.{name} — only kel's own types and a small "
            "builtin allowlist are permitted when loading checkpoint files"
        )


def _safe_loads(data: bytes) -> list[Checkpoint]:
    return _RestrictedUnpickler(io.BytesIO(data)).load()


class FileCheckpointStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.pkl"

    def save(self, checkpoint: Checkpoint) -> None:
        history = self.history(checkpoint.run_id)
        history.append(checkpoint)
        self._path(checkpoint.run_id).write_bytes(pickle.dumps(history))

    def load_latest(self, run_id: str) -> Checkpoint | None:
        history = self.history(run_id)
        return history[-1] if history else None

    def history(self, run_id: str) -> list[Checkpoint]:
        path = self._path(run_id)
        if not path.exists():
            return []
        return _safe_loads(path.read_bytes())

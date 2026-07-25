"""Zero-dependency embedder for local dev/testing without an API key.
Real production use should pass a real `kel.models.EmbeddingModel`
(`kel.get_embedding_model("openai:text-embedding-3-small")` or
`"cohere:embed-v4.0"`) via `embedder_from_model` below instead."""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable

from kel.models.base import EmbeddingModel


def embedder_from_model(model: EmbeddingModel) -> Callable[[str], list[float]]:
    """Adapts a batch-oriented `EmbeddingModel.embed(texts) -> vectors` into
    the single-text `Callable[[str], list[float]]` shape `Retriever`
    expects. Known inefficiency: this makes one API call per text since
    `Retriever` embeds one chunk/query at a time — fine for query-time
    embedding, wasteful for bulk ingestion of many chunks. Call
    `model.embed(texts)` directly with a full batch for bulk ingestion
    instead of going through this per-text bridge."""

    def embed_one(text: str) -> list[float]:
        return model.embed([text])[0]

    return embed_one


class NaiveHashEmbedder:
    """Deterministic bag-of-words hash embedding. Captures crude lexical
    overlap, nothing semantic — good enough to exercise the retrieval
    pipeline locally, not good enough for production relevance.

    Uses zlib.crc32, not Python's builtin `hash()` — string hashing is
    randomized per-process by default (PYTHONHASHSEED), so `hash()` would
    make this "deterministic" embedder actually vary run to run."""

    def __init__(self, dims: int = 64):
        self.dims = dims

    def __call__(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in text.lower().split():
            vec[zlib.crc32(word.encode("utf-8")) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec

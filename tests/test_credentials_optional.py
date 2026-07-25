"""Contract tests: none of kel's adapters should *require* an explicit
credential argument. This matters concretely for EC2 (instance profile
role), EKS (IRSA / pod identity), ECS (task role), and self-hosted
in-cluster services (Weaviate/Chroma with anonymous access) — all of these
resolve credentials ambiently, and forcing an explicit key/token/password
argument would break every one of them. Checked at the signature level so
a future change that accidentally makes a credential required fails a test
immediately, not just in some end-user's cluster."""

from __future__ import annotations

import inspect

import pytest


def _required_params(cls) -> list[str]:
    sig = inspect.signature(cls.__init__)
    return [
        name
        for name, param in sig.parameters.items()
        if name not in ("self",)
        and param.default is inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]


@pytest.mark.parametrize(
    "import_path,cls_name,expected_required",
    [
        ("kel.models.providers.anthropic", "AnthropicChatModel", ["model_id"]),
        ("kel.models.providers.openai", "OpenAIChatModel", ["model_id"]),
        ("kel.models.providers.cohere", "CohereChatModel", ["model_id"]),
        ("kel.models.providers.gemini", "GeminiChatModel", ["model_id"]),
        ("kel.models.providers.mistral", "MistralChatModel", ["model_id"]),
        ("kel.storage.s3", "S3BlobStore", ["bucket"]),
        ("kel.retrieval.weaviate_store", "WeaviateVectorStore", ["collection_name"]),
        ("kel.retrieval.chroma_store", "ChromaVectorStore", ["collection_name"]),
        ("kel.retrieval.pgvector_store", "PgVectorStore", ["table_name"]),
    ],
)
def test_adapter_never_requires_an_explicit_credential_argument(import_path, cls_name, expected_required):
    module = __import__(import_path, fromlist=[cls_name])
    cls = getattr(module, cls_name)
    assert _required_params(cls) == expected_required


def test_pinecone_only_requires_index_name_not_a_key():
    # Pinecone is SaaS-only with no ambient-credential path (unlike S3/RDS) —
    # it still shouldn't force api_key as a *required* positional/keyword
    # argument, since the key can come from PINECONE_API_KEY instead.
    from kel.retrieval.pinecone_store import PineconeVectorStore

    assert _required_params(PineconeVectorStore) == ["index_name"]


def test_mistral_api_key_falls_back_to_env_not_forced(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    from kel.models.providers.mistral import MistralChatModel

    model = MistralChatModel("mistral-large-latest")  # no api_key passed at all
    assert model._api_key is None  # doesn't crash, just has nothing — resolved lazily on first real client use

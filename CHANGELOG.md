# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-07-26

### Added
- `Memory` now recalls prior turns automatically: if `working` isn't given explicitly, working memory is seeded from `episodic`'s existing transcript for `session_id`. Pass the same durable `episodic` store (e.g. `FileEpisodicStore`) and `session_id` across process restarts or per-interaction script reruns (Streamlit-style apps) and the conversation resumes instead of silently starting over. `InMemoryEpisodicStore` (the default) doesn't outlive the process, so default behavior is unchanged unless a durable store is supplied.
- `Agent` now runs every tool call from one model turn concurrently (`ThreadPoolExecutor` for `run`/`run_stream`, `asyncio` for `arun`/`arun_stream`) instead of one at a time in a `for` loop — a model requesting N independent tools (e.g. N web searches) previously paid their latencies serially. `run_stream`/`arun_stream` report each `ToolResultEvent` as its call actually finishes (`as_completed`), not in submission order.
- `SQLiteEpisodicStore` (`kel.memory`): a durable, single-file conversation-history backend — the SQLite counterpart to `FileEpisodicStore`, usable from multiple *processes* sharing one file (unlike `InMemoryEpisodicStore`, which never leaves the process). Same zero-new-dependency pattern as `kel.caching.SQLiteCache`.
- `filter` parameter on `VectorStore.query()`/`keyword_query()` and `Retriever.retrieve()`/`retrieve_hybrid()`, implemented across all six adapters (InMemory, Qdrant, Pinecone, Weaviate, Chroma, pgvector) — scope a search to metadata that matches every given key/value, e.g. `retriever.retrieve(query, filter={"user_id": "u1"})`, without maintaining separate collections per filter dimension.
- `approve_tool_call` hook on `Agent` (`kel.agents.ApprovalHook`): an injected `(name, input) -> bool` gate checked before any tool call runs, across every run variant. Return `False` to reject a call before it executes — e.g. pause on a specific tool and require a yes/no. Unset (the default) approves every call, matching prior behavior.
- `load_html`/`load_csv`/`load_csv_rows` document loaders (`kel.retrieval`), stdlib-only (`html.parser`, `csv`) — no new dependency. `load_html` reuses `kel.tools.web_fetch`'s text-extraction logic rather than a second implementation.
- `LLMReranker` (`kel.retrieval.reranker`): an optional second, more expensive relevance pass over a first-stage retriever's candidates, using a `ChatModel` via `generate_structured` (one call scores every candidate at once, not one call per candidate) — no new dependency, unlike a cross-encoder model. Injected via `Retriever(store, embedder, reranker=...)`; retrieval is unchanged when no reranker is given.

### Changed
- Replaced the CLI's ASCII banner with a cleaner, properly-aligned "KEL" rendering (pure ASCII, verified safe under `cp1252` and other narrow console encodings).

### Performance
- `ContextWindow.tokens_used` (i.e. `Memory.working`, which every `Agent.run()` turn appends to) recomputed the token count over the *entire* message history on every single `add`/`extend` call — an n-turn session cost O(n^2) instead of O(n). Now maintains a running total incrementally, recomputing in full only on the (already rare) eviction-overflow path.
- `sliding_window_eviction` recomputed the total from scratch on every popped message and used `list.pop(0)` (itself O(n)) — evicting k of n messages cost O(k*n). Switched to a `deque` (O(1) pop-from-front) with a running total decremented per pop, making eviction O(n) total regardless of how many messages are dropped.

### Fixed
- `python -m kel.sdk.cli` silently did nothing (exit 0, no output) regardless of subcommand — `cli.py` had no `if __name__ == "__main__":` guard, so `main()`/`run_cli()` were defined but never invoked. Only the installed `kel` console-script actually worked. Fixed; `python -m kel.sdk.cli` now runs correctly (a separate, harmless `RuntimeWarning` about the module being imported twice remains — an inherent quirk of `python -m` combined with `kel.sdk`'s eager re-export of `cli`, not a functional issue).
- `make_summarization_eviction()`'s fallback discarded the summary message first when `[summary, *recent]` was still over budget, because `sliding_window_eviction` evicts from the front of the list and the summary sat at the front — silently defeating the point of summarizing under a tight budget. Now protects the summary and evicts `recent`'s oldest messages first, falling back to plain sliding-window eviction only if the summary alone doesn't fit.
- `WeaviateVectorStore.query()`/`_ensure_collection()` unconditionally imported `weaviate.classes.query.MetadataQuery`/`weaviate.classes.config.Configure`, so even a fully injected fake client still required the real `weaviate-client` package installed — unlike Chroma/pgvector/Pinecone, which are fully fakeable via dependency injection with no vendor package installed at all. Now only constructs those real vendor types when talking to a real client (verified end-to-end with `weaviate-client` actually uninstalled).
- `ChromaVectorStore` never set an explicit distance metric, so Chroma silently defaulted to `l2` (squared Euclidean) instead of cosine — unlike Qdrant, Pinecone, and pgvector, which all use or require cosine. `query()`'s `score = 1.0 - distance` produced wrong, unbounded values under that default. Now creates the collection with `metadata={"hnsw:space": "cosine"}` explicitly, matching every other `VectorStore` adapter's score semantics.

## [1.0.2] - 2026-07-26

### Fixed
- `QdrantVectorStore.query()` called `search()`, which qdrant-client removed in favor of `query_points()` — fixed to use `query_points()`, matching the currently supported client API.
- `Agent` never forwarded `max_tokens`/`temperature` to the model on any of `run`/`arun`/`run_stream`/`arun_stream`, silently locking every agent to each provider adapter's hardcoded default (e.g. 1024 tokens). Added `max_tokens`/`temperature` constructor params on `Agent`, forwarded to every generation call only when explicitly set (so provider defaults are preserved when unset).
- `Agent` stored a model's response into shared memory unconditionally, even when a turn came back with no text and no tool calls (e.g. truncated by a provider-side error). Since `Agent.memory` persists across every `run()` call, that empty turn corrupted every later question in the session. Added `EmptyModelResponseError` (`kel.agents.errors`), raised before the malformed turn is stored, on `run`/`arun`/`run_stream`/`arun_stream`.

## [1.0.1] - 2026-07-26

### Changed
- Tightened the PyPI/GitHub-facing description and README copy for a more professional presentation.
- Added an explicit `GitHub` project URL and a PyPI version badge.
- Converted README's relative doc links (`DESIGN.md`, `USAGE.md`, `LICENSE`, workflow files) to absolute GitHub URLs so they resolve correctly when PyPI renders the README standalone.

## [1.0.0] - 2026-07-26

### Added
- PEP 561 `py.typed` marker and dynamic single-sourced versioning.
- PyPI metadata: classifiers, keywords, `project.urls`.
- Ruff lint and mypy type-check jobs in CI.
- ASCII banner shown by the `kel` CLI (`kel --version` / bare `kel`).
- CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY docs and GitHub issue/PR templates.

### Changed
- Published to PyPI as `pykel` (the `kel` name was already taken by an unrelated package); `import kel` and the `kel` CLI command are unaffected.

## [0.1.0] - 2026-07-24

### Added
- Core model gateway with Anthropic, OpenAI, Cohere, Gemini, and Mistral adapters — all credentials-optional by design (env vars / ambient cloud credentials work with no explicit key).
- Observability (tracer, spans, sinks), budget tracking, rate limiting, and response caching.
- Context window management and memory (episodic + semantic) subsystems.
- Retrieval: Qdrant, Pinecone, Weaviate, Chroma, and pgvector vector stores; recursive text splitter; PDF loader.
- Agent runtime with graph-based execution, streaming, human-in-the-loop interrupts, and structured output.
- Tooling: web fetch (SSRF-hardened), web search, SQL, and shell tools.
- Testing utilities: cassette recording/replay, LLM-graded evals, span/trace assertions.
- Storage: file/S3 blob storage, checkpointing (hardened against unsafe deserialization).
- Lightweight, stdlib-only monitoring dashboard with live-refreshing metrics.
- `.md`-based agent specs, CLI (`run`/`eval`/`trace`), and `serve()` HTTP runtime.
- DevSecOps pipeline: Trivy, pip-audit, and Bandit scanning; Dependabot.

[Unreleased]: https://github.com/kprasad7/kel/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/kprasad7/kel/releases/tag/v1.1.0
[1.0.2]: https://github.com/kprasad7/kel/releases/tag/v1.0.2
[1.0.1]: https://github.com/kprasad7/kel/releases/tag/v1.0.1
[1.0.0]: https://github.com/kprasad7/kel/releases/tag/v1.0.0
[0.1.0]: https://github.com/kprasad7/kel/releases/tag/v0.1.0

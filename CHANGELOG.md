# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance
- `ContextWindow.tokens_used` (i.e. `Memory.working`, which every `Agent.run()` turn appends to) recomputed the token count over the *entire* message history on every single `add`/`extend` call — an n-turn session cost O(n^2) instead of O(n). Now maintains a running total incrementally, recomputing in full only on the (already rare) eviction-overflow path.
- `sliding_window_eviction` recomputed the total from scratch on every popped message and used `list.pop(0)` (itself O(n)) — evicting k of n messages cost O(k*n). Switched to a `deque` (O(1) pop-from-front) with a running total decremented per pop, making eviction O(n) total regardless of how many messages are dropped.

### Fixed
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

[Unreleased]: https://github.com/kprasad7/kel/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/kprasad7/kel/releases/tag/v1.0.2
[1.0.1]: https://github.com/kprasad7/kel/releases/tag/v1.0.1
[1.0.0]: https://github.com/kprasad7/kel/releases/tag/v1.0.0
[0.1.0]: https://github.com/kprasad7/kel/releases/tag/v0.1.0

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.7.2] - 2026-07-26

### Added
- `FalMediaModel(..., cost_estimator=)` (`kel.media`): reserves an *estimated* cost against a `BudgetTracker` from the request arguments **before** the real network call, instead of only charging after a (possibly expensive, minutes-long) generation already completed. `cost_usd=`'s post-hoc charging still works unchanged when no `cost_estimator` is given; passing both isn't double-charged — `cost_estimator` takes over entirely for that call. Same "conservative, not exact" tradeoff `kel.ratelimit` already documents for its own up-front token reservation. Closes the gap where a single expensive call could still blow past a budget cap before it was ever charged.

## [1.7.1] - 2026-07-26

### Added
- `kel.media`'s `FalMediaModel` now translates fal.ai errors into kel's own `ProviderError`/`AuthenticationError`/`RateLimitError` hierarchy (best-effort, from an HTTP status code on the raised exception) instead of leaking a raw vendor exception — the same translation every chat-model provider adapter already does.
- `FalMediaModel.submit()`/`FalJobHandle` (`kel.media`): the queue-based submit-then-poll flow fal's own docs recommend for slow generations (video especially can take minutes) instead of blocking on `generate()`.
- `FalMediaModel(..., budget=, cost_usd=)`: meters media generation cost against a `kel.budget.BudgetTracker`, the same way `kel.get_model(budget=...)` meters chat spend — `cost_usd` can be a fixed float or a function computing cost from the actual response (e.g. reported duration), since fal has no fixed per-token pricing table the way chat models do. Directly addresses the "surprise bill" pattern real fal.ai users report.

## [1.7.0] - 2026-07-26

### Added
- `kel.media` (`pip install "pykel[fal]"`): a new generic gateway for third-party generative-media APIs — image generation, video generation, text-to-speech, and lipsync — via `get_image_model()`/`get_video_model()`/`get_audio_model()`/`get_lipsync_model()`, all resolving `"provider:model"` specs the same way `kel.get_model` does. One built-in provider (fal.ai) covers all four media types through a single generic `FalMediaModel` class, since fal's platform already exposes every media type through the same request shape; `register_media_provider` keeps the registry open for another vendor. `FalTTSProvider`/`FalSTTProvider` also close `kel.realtime`'s "wire up your own vendor" gap with a real, concrete fal.ai-backed implementation of its `TTSProvider`/`STTProvider` Protocols. Same dependency-injection (`client=`) testing pattern as every other adapter — not exercised against a live fal.ai key, implemented against fal's documented SDK shape and verified against injected fakes.

## [1.6.0] - 2026-07-26

### Added
- `create_fastapi_app()`/`add_agent_routes()` (`kel.sdk`) and `serve_websocket()` now accept a zero-arg `Agent` factory in addition to a single `Agent` instance. This closes the real production-scale gap both adapters' docstrings previously just documented as a limitation: a single shared `Agent` means every caller reads/writes the same conversation history. Passing a factory instead makes the FastAPI adapter look up `payload["session_id"]` in a lazily-built, TTL-evicted per-session registry (`session_ttl_seconds`, default 1 hour idle), and makes the WebSocket adapter build a fresh `Agent` per connection (a connection already being a natural session boundary). Passing a plain `Agent` still works exactly as before — this is additive, not a breaking change.

## [1.5.1] - 2026-07-26

### Fixed
- **Concurrent calls to `run()`/`arun()`/`run_stream()`/`arun_stream()` on the same `Agent` instance could interleave their `memory.remember_turn()` writes, silently scrambling conversation history** — reproduced as 5 concurrent `run()` calls producing 5 user messages back-to-back, then 5 assistant messages, instead of 5 alternating pairs. This became much more likely to hit in practice now that `kel.sdk.serve`/`serve_websocket`/`fastapi_adapter` all naturally invite multiple concurrent callers against one shared `Agent`. Fixed with a per-instance lock (`threading.Lock` for the sync methods, `asyncio.Lock` for the async ones) serializing calls on one `Agent`; different `Agent` instances are unaffected and still run fully concurrently. Documented in `serve`/`serve_websocket`/`fastapi_adapter` that sharing one `Agent` across many callers still means they share one conversation (no longer corrupted, but still one history) — construct one `Agent` per session for real multi-user serving.
- `Graph.set_fallback()` pointing at a node that was never registered (e.g. a typo) used to only surface as a confusing raw `KeyError` deep inside the executor, and only once the fallback path was actually taken at runtime. `Graph.validate()` now catches this up front, the same way it already validates `entry`.

## [1.5.0] - 2026-07-26

### Added
- `reflect_and_retry()`/`areflect_and_retry()` (`kel.agents`): a ready-made generator/critic reflection loop — a downstream critic's feedback is fed back to the agent as the next turn's input, and it retries up to `max_attempts`. The same "reverse feedback" pattern already expressible as a cyclic `Graph` (`agent_node()` + a conditional edge routing back), packaged as a helper for the common one-generator/one-critic case instead of hand-wiring a `Graph` every time.
- `max_workers` on `run_graph()`/`resume_graph()`/`fork_from_checkpoint()` (`kel.runtime`): the per-layer concurrency ceiling was hardcoded at 8 regardless of how wide a layer's fan-out actually was — a real throughput cap on large multi-agent flows with many parallel branches. Now configurable (default unchanged at 8); verified with a deterministic barrier-based test that the cap is actually enforced, not just accepted and ignored.

## [1.4.0] - 2026-07-26

### Added
- `create_fastapi_app()`/`add_agent_routes()` (`kel.sdk`, `pip install kel[fastapi]`): a FastAPI adapter for `Agent` — `POST /invoke` (JSON, matching `kel.sdk.serve`'s contract) and `POST /stream` (Server-Sent Events, one per `run_stream()` event), the "real ASGI deploy story" upgrade `kel.sdk.serve`'s own docstring flagged as reasonable future work, in the same spirit as LangChain's LangServe. Uses `Agent.arun()`/`arun_stream()` so a slow model call doesn't block the ASGI event loop. Verified end-to-end with FastAPI's real `TestClient`, including actual SSE streaming.

## [1.3.0] - 2026-07-26

### Added
- `Graph.set_fallback()` (`kel.runtime`): a node that raises now routes to a designated fallback node instead of crashing the whole run, with the error captured into `state["__error__"]` for the fallback to actually inspect. Opt-in per node — a node without a fallback registered still raises, same as before.
- `HallucinationChecker` (`kel.agents`): a second, more expensive pass that checks whether a response's claims are actually supported by given source material (RAG chunks, tool results, etc.), using the same one-call structured-output pattern `LLMReranker` uses. Not wired into `Agent` automatically — run it yourself after `agent.run(...)`.
- `ttl_seconds` on `SemanticMemory.remember()`: an opt-in expiration per fact (default: never expires). `search()` never returns an expired fact; `forget_expired()` purges them from storage. One parameter instead of a forced multi-tier memory taxonomy.
- `Notifier`/`WebhookNotifier`/`notify_interrupt()` (`kel.runtime`): `Interrupt`/`resume_graph()`/`CheckpointStore` already let a run pause and resume arbitrarily later (state is just data — persist it for weeks if needed); this closes the other half, telling a human it's actually waiting on them. `WebhookNotifier` is a zero-new-required-dependency stdlib `urllib` POST, compatible with Slack incoming webhooks, PagerDuty, and most custom notification endpoints.

## [1.2.0] - 2026-07-26

### Added
- `context_selector` on `sequential_pipeline` and `results_selector` on `run_supervisor`: injected filters that scope what each agent/the supervisor actually sees from shared state, instead of always seeing every upstream output or every accumulated result — closes the "shared context bus floods downstream agents" complaint for pipelines with many agents.
- `fork_from_checkpoint()` (`kel.runtime`): real state time-travel — rewind to an arbitrary historical `Checkpoint` (not just an `Interrupt`'s pause point, which is all `resume_graph` exposed) and continue forward as a new branch, optionally patching state first. This is what `Checkpoint`'s own docstring already promised ("a run can pause/resume/branch from any point") but `run_graph`/`resume_graph` alone never actually exposed.
- `agent_node()` (`kel.agents`): wraps any `Agent` as a `kel.runtime.Graph` node function, so agents compose into fully dynamic, cyclic, multi-directional graphs via `Graph` directly (which already supports conditional edges and cycles) instead of being limited to the four fixed orchestration shapes (sequential/supervisor/parallel/swarm).
- `MCPToolset`/`mcp_tools_from_server()` (`kel.tools`, `pip install kel[mcp]`): connects to a Model Context Protocol server and exposes every tool it advertises as a `kel.agents.Tool` in one call, instead of hand-writing a custom integration adapter per server.
- `serve_websocket()`/`KelWebSocketServer` (`kel.sdk`, `pip install kel[websockets]`): a working WebSocket endpoint that streams `Agent.run_stream()` events to a connected client — the concrete, ready-made piece behind the common "stream an agent's response over a socket" case, versus needing to hand-write the async network loop yourself. Full bidirectional voice/video realtime orchestration remains `kel.realtime`'s documented interfaces-only scope, a real vendor SDK's job.

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

[Unreleased]: https://github.com/kprasad7/kel/compare/v1.7.2...HEAD
[1.7.2]: https://github.com/kprasad7/kel/releases/tag/v1.7.2
[1.7.1]: https://github.com/kprasad7/kel/releases/tag/v1.7.1
[1.7.0]: https://github.com/kprasad7/kel/releases/tag/v1.7.0
[1.6.0]: https://github.com/kprasad7/kel/releases/tag/v1.6.0
[1.5.1]: https://github.com/kprasad7/kel/releases/tag/v1.5.1
[1.5.0]: https://github.com/kprasad7/kel/releases/tag/v1.5.0
[1.4.0]: https://github.com/kprasad7/kel/releases/tag/v1.4.0
[1.3.0]: https://github.com/kprasad7/kel/releases/tag/v1.3.0
[1.2.0]: https://github.com/kprasad7/kel/releases/tag/v1.2.0
[1.1.0]: https://github.com/kprasad7/kel/releases/tag/v1.1.0
[1.0.2]: https://github.com/kprasad7/kel/releases/tag/v1.0.2
[1.0.1]: https://github.com/kprasad7/kel/releases/tag/v1.0.1
[1.0.0]: https://github.com/kprasad7/kel/releases/tag/v1.0.0
[0.1.0]: https://github.com/kprasad7/kel/releases/tag/v0.1.0

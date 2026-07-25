# kel — Universal Agentic OS

## 1. Why

LangChain's core pain points this project targets directly:

- **Opaque execution** — no first-class tracing/metrics; you bolt on LangSmith or nothing.
- **Leaky abstractions** — chains/agents wrap so much that debugging means reading library internals.
- **Context handling is an afterthought** — no native long-context management, summarization, or loop-safe state.
- **Vector store lock-in friction** — swapping stores means rewriting retriever code.
- **RAG is one-size-fits-all** — naive top-k similarity, no native support for hybrid/graph/agentic RAG.
- **Text-first** — audio/video/image flows are bolted on via loaders, not native citizens.
- **No prompt lifecycle** — prompts live as inline strings or brittle templates, no versioning/eval loop.
- **No memory architecture** — "memory" classes are toy buffers, not a real hierarchy (working/episodic/semantic).
- **No token/cost governance** — no built-in budget enforcement, truncation policy, or spend tracking.
- **No testing story** — no way to regression-test agent behavior deterministically.
- **Model API sprawl** — every provider needs custom glue for auth, streaming, tool-calling schema quirks.

kel's bet: treat these as **one system's subsystems with clean interfaces**, not a pile of independent integrations.

### 1.1 What "better than LangChain" means concretely

kel must be a **superset**, not a rewrite that drops capability. Everything LangChain offers has to have a kel equivalent, plus the fixes below. Feature-parity checklist:

| LangChain concept | kel equivalent |
|---|---|
| Chat models / LLMs wrapper | `kel.models` Model Gateway (3.1) |
| Prompt templates | `.md` prompt specs (3.6) |
| Output parsers / structured output | Gateway-level structured output (3.1), schema-validated |
| Chains / LCEL (`|` pipe) | `kel.runtime` typed DAG (3.10) |
| Agents / AgentExecutor | `kel.agents` (3.11) |
| Tools / tool-calling | Gateway-normalized tool schema (3.1) + tool registry (3.11) |
| `langchain_community.tools` (built-in tool library) | `kel.tools` — pluggable web search (Wikipedia/DuckDuckGo/Tavily/Brave/SerpAPI/Bing/Google, selected dynamically via `get_web_search_tool`) + generic URL content fetching, see USAGE.md §16 |
| Memory classes | `kel.memory` layered memory (3.3) |
| Document loaders / text splitters | `kel.retrieval` ingestion pipeline (3.4) |
| VectorStores / Retrievers | `kel.retrieval` adapter interface (3.4) |
| Callbacks / tracing | `kel.observability` (3.5) — first-class, not opt-in |
| LangGraph (graph orchestration, cycles, checkpoints) | `kel.runtime` (3.10) — cyclic DAGs with native checkpointing |
| LangSmith (tracing/eval SaaS) | `kel.observability` + `kel.testing`, self-hostable via Grafana stack |
| LangServe (deploy chains as API) | `kel.sdk` serving layer (3.13) |
| Multi-agent (supervisor/swarm patterns) | `kel.agents` (3.11), plus first-class multi-provider routing |
| Router chains / RunnableBranch | `kel.brain` central planner/router (3.14) — rules + LLM fallback in v1, learned router optional later |
| Retry/fallback wrappers (`.with_retry`, `.with_fallbacks`) | `kel.heal` self-healing layer (3.15) — diagnosis-driven, not blind retry |

### 1.2 Research: where current agent systems actually break (2026)

Findings from current production reports, folded directly into the subsystem designs below:

- **Voice agents**: LLM inference is ~70% of round-trip latency; time-to-first-token dominates perceived responsiveness. A naive RAG-in-the-loop call adds 50–300ms (vector query) on top of embedding + generation, blowing past the ~200ms budget needed for natural turn-taking; production targets for 2026 are p50 < 250–400ms, p95 < 800ms. Tool calls during inference make this worse because the whole tool schema rides in-prompt. → drives `kel.realtime`'s dual-path (fast-path direct response vs. slow-path tool/RAG) design in 3.12.
- **Video/lipsync agents**: the naive architecture round-trips audio to a GPU render server and back over WebSocket, re-encoding video twice; the fix used by leading 2026 stacks is having the render server join the call as its own media participant and publish video directly, cutting a full codec round trip. Current lipsync latency is 2–8s (D-ID, Wav2Lip, Sync Labs); the target for natural interaction is ~200ms turn-taking gap. Hidden audio-output buffering adds another 100–200ms unless mitigated with phoneme look-ahead prediction. → drives 3.12's media-participant architecture.
- **Multi-agent orchestration**: sequential chains mean one slow agent blocks everything; parallel fan-out needs explicit merge/conflict-resolution logic; and — most reported failure mode — agents lose each other's context ("agent 2 doesn't know what agent 1 decided, agent 3 invents context agent 2 never produced"). → drives 3.11's shared-context-bus requirement, not just message-passing.
- **Prompt/context engineering**: as of 2026, most working teams report agent failures are *context* failures, not model failures — stale or bloated context reads as confident wrong reasoning. The fix pattern reported across production systems: write context to external storage (don't carry everything in-band), select what's relevant dynamically, compress to fit the attention budget, and isolate each agent/subagent to a clean focused window. → this *is* `kel.context` (3.2) and `kel.memory` (3.3); it's why they're separate subsystems instead of a single "memory" bag.

## 2. Design principles

1. **Every subsystem is swappable behind a small interface.** No subsystem assumes a specific vendor.
2. **Observability is not optional.** Every execution unit emits structured events by default; Grafana/OTel is a first-class sink, not a plugin someone builds later.
3. **State is explicit.** Context, memory, and budget are objects you can inspect, serialize, and replay — not hidden in closures.
4. **.md files are the prompt/config source of truth.** Prompts, agent specs, and eval cases live as versioned Markdown with frontmatter, loaded and hot-reloaded at runtime — not Python string literals.
5. **Deterministic replay for testing.** Any agent run can be recorded and replayed against a frozen model/tool trace for regression tests.
6. **Budget-aware by construction.** Every call path carries a token/cost budget object; overruns are policy decisions (truncate, summarize, abort), not silent failures.

## 3. Core subsystems

```
┌─────────────────────────────────────────────────────────────┐
│                        kel.runtime                            │
│   (execution graph engine: agents, loops, tool calls, DAGs)  │
└───────┬─────────┬─────────┬─────────┬─────────┬──────────────┘
        │         │         │         │         │
   model gateway  memory   retrieval  observability  budget/testing
```

### 3.1 Model Gateway (`kel.models`)
- Single interface (`ChatModel`, `EmbeddingModel`) implemented per provider. **Core-maintained set, kept deliberately small: Anthropic, OpenAI, Gemini, local/Ollama or vLLM.** Everything else (Mistral, Bedrock, Azure, smaller/regional providers) ships as a documented adapter contract + a community-contrib directory, not something the core team promises to keep working forever. "Supports every model" is a maintenance treadmill that has sunk projects before it sinks this one — "supports every model *someone bothers to adapt, behind one stable interface*" is the honest, sustainable version.
- Normalizes: auth (API key / OAuth / IAM), tool-call schemas, streaming, structured output, multimodal input (text/image/audio/video), rate limits, retries.
- Provider capability matrix declared in config (e.g., which models support vision, which support prompt caching) so the router can pick/fallback automatically.

### 3.2 Context & Loop Engineering (`kel.context`)
- `ContextWindow` object: tracks tokens used, applies eviction policy (sliding window, summarization-on-overflow, hierarchical compression).
- `Loop` primitive: bounded agentic loops with explicit exit conditions, step budget, and stuck-loop detection (repeated tool calls / no-progress heuristics) — the thing LangChain agents notoriously lack.
- Long-context strategies pluggable: map-reduce, recursive summarization, retrieval-augmented context injection.

### 3.3 Memory (`kel.memory`)
- Layered: **working** (current context), **episodic** (per-session transcript store), **semantic** (long-term facts, vector-backed), **procedural** (learned prompt/tool patterns, stored as .md).
- Consolidation job: episodic → semantic summarization on a schedule or trigger, mirroring the memory file convention already used in this harness.

### 3.4 Retrieval / RAG (`kel.retrieval`)
- Vector store interface: Pinecone, Weaviate, Qdrant, Chroma, pgvector, Milvus — adapters behind one interface (`upsert`, `query`, `delete`, `hybrid_query`).
- Native RAG strategies as composable stages: naive top-k, hybrid (BM25 + vector), re-ranking, graph-RAG (entity/relation traversal), agentic RAG (retrieval as a tool the agent calls iteratively).
- Multimodal ingestion pipeline: text, image (CLIP-style embeddings), audio (transcription → embedding), video (frame sampling + transcription).

### 3.5 Observability (`kel.observability`)
- Every node in the execution graph emits OpenTelemetry spans + structured events (tokens, latency, cost, tool I/O).
- Native exporters: OTLP → Grafana/Tempo/Loki/Prometheus stack, plus a local dev dashboard.
- Trace includes full replay data (see 3.7).

### 3.6 Prompt & Agent Definition (`kel.specs`)
- Agents, prompts, and eval suites defined as `.md` files with YAML frontmatter (model, tools, budget, version).
- Hot-reloadable; versioned via git; a `kel eval` command runs an agent spec against its eval cases and diffs against the last known-good run.

### 3.7 Testing Framework (`kel.testing`)
- Record/replay: capture real model + tool responses, replay deterministically in CI.
- Golden-trace regression tests: assert on trace shape (which tools called, in what order) not just final output.
- Property-based agent tests: fuzz inputs, assert budget/loop invariants hold (never exceeds token budget, never infinite-loops).

### 3.8 Token & Cost Control (`kel.budget`)
- `Budget` object threaded through every call: max tokens, max cost, max tool calls, max wall time.
- Enforcement points at model call, loop iteration, and subagent spawn boundaries.
- Live cost tracking per session/agent, exportable to observability sink.

### 3.9 Storage & Data Transfer (`kel.storage`)
- Content-addressed local store for artifacts (traces, embeddings cache, media blobs) — avoids re-computation and enables replay.
- Async I/O throughout; zero-copy where possible for large media payloads (memory-mapped buffers, not base64-in-JSON round trips for local transfers).
- Pluggable backends: local FS, S3-compatible, for artifact persistence.

### 3.10 Runtime / Execution Graph (`kel.runtime`)
- Agents and tools are nodes in a typed DAG (not just linear chains) — supports branching, parallel fan-out, and cyclic loops with the guardrails from 3.2.
- Sync and async execution; process-level isolation for untrusted tool code.
- Native checkpointing (LangGraph-equivalent): every node transition persists state so a run can pause/resume/branch from any point.

### 3.11 Multi-Agent Orchestration (`kel.agents`)
- Patterns natively supported: supervisor/worker, swarm (peer handoff), sequential pipeline, parallel fan-out with merge — selectable per graph, not hardcoded into one "agent" abstraction.
- **Credential model is user-chosen, per agent, at graph-definition time:**
  - *Single API key, many agents*: one provider credential shared across all agents in the graph (simplest, cheapest to operate, all agents share rate limits).
  - *Multiple API keys / multiple providers*: each agent in the graph binds its own `ChatModel` instance (own key, own provider, even own vendor) — e.g. a cheap/fast model for a routing agent, a frontier model for the reasoning agent, a local model for a privacy-sensitive tool-caller. The Model Gateway (3.1) makes this a config swap, not a code change.
  - Declared in the agent's `.md` spec frontmatter (`model:`, `api_key_ref:`) so credential choice is versioned alongside the agent, not buried in environment setup.
- **Shared context bus, not just message-passing**: agents don't only exchange the last message — they read/write a structured shared-state object (the graph's checkpoint state) so a downstream agent can see what an upstream agent *decided*, not just what it said. This directly targets the "agent 3 invents context agent 2 never produced" failure mode from 1.2.
- Per-agent budget (3.8) and trace (3.5) — a slow or runaway agent is visible and boundable independently of the rest of the graph.

### 3.12 Real-Time Multimodal Agents — voice, video, lipsync (`kel.realtime`)
**Descoped from "build our own media infrastructure" to "orchestrate the best existing infrastructure."** Building a custom WebRTC media server, GPU render pipeline, and codec stack is a different company's worth of systems engineering — LiveKit, Daily, Deepgram, Cartesia, ElevenLabs, Tavus and others already do this well and compete on exactly that latency race. kel adding its own would be scope creep that delays or kills the actual differentiators (gateway, observability, budget, testing). kel's job here is orchestration, not infrastructure:
- **Voice**: full-duplex pipeline (STT → LLM → TTS) that calls out to best-in-class streaming STT/TTS providers, with kel providing the TTFT-optimized model routing (fast/cheap model for turn-taking filler or routing, frontier model for substantive answers) and the dual-path pattern below — the parts that are actually kel's job, not re-implementing speech infra.
- **Dual-path response**: a fast path answers directly from context/working memory while a slow path (RAG query, tool call) runs concurrently; the fast path can emit a natural filler ("let me check that") instead of the agent going silent during the 50–300ms+ RAG round trip. This is orchestration logic, cheap to build, high leverage.
- **Video/avatar/lipsync**: kel integrates with providers whose architecture already joins the call as a media participant (avoiding the double-encode round trip) — kel does not build its own media-participant server. If no existing provider fits a use case, that's a signal for a narrow provider-specific adapter, not a reason to build a general render pipeline in core.
- Pluggable backends: any STT/TTS/lipsync/avatar-render vendor behind the same interface pattern as the Model Gateway.
- All of this emits the same observability events as text agents (3.5) — latency is a first-class trace dimension here, not an afterthought, since it's the actual product metric for real-time agents.
- **Ships as an optional package (`kel-realtime`), not core.** Most kel users building text/tool agents should never need to install it.

### 3.13 SDK & UI (`kel.sdk`, `kel.ui`)
- **SDK**: a small, typed, batteries-included Python API — define an agent, a graph, or a full multi-agent system in a few lines; sensible defaults for model, memory, and budget so a first agent doesn't require configuring all 10 subsystems. LangServe-equivalent one-line deploy (`kel.sdk.serve(graph)`) exposing a REST/streaming API.
- **CLI**: `kel run`, `kel eval`, `kel trace`, `kel replay` — mirrors the testing/observability subsystems from the terminal.
- **UI**: local dashboard (dev mode) for live-inspecting a running graph — trace timeline, token/cost meters, memory contents, per-agent state — separate from but data-compatible with the Grafana exporters in 3.5 (Grafana for ops/production monitoring, kel UI for development/debugging).
- Scaffolding/templates (`kel init voice-agent`, `kel init rag-chatbot`, `kel init multi-agent-swarm`) so common patterns don't start from a blank graph.

### 3.14 Central Planner & Router — "the brain" (`kel.brain`)
Sits above `kel.runtime` (3.10) and `kel.agents` (3.11): decides *what runs next, in what order, and whether to fan out in parallel*, instead of that logic being scattered across hand-written graph edges. This is the piece that makes kel feel "clever" rather than a graph you have to hand-wire for every case.

- **Two-tier decision making**:
  - *Fast tier* — cheap, low-latency routing (which tool, which agent, which model tier, continue-vs-stop) that avoids burning a full LLM call on every decision. **V1 implementation is deliberately unglamorous: rules + embedding-similarity matching against past routes, no training pipeline.** A custom-trained ANN router needs a training pipeline, feature engineering, model serving, and drift monitoring — real ML infrastructure that only pays for itself at a traffic volume kel won't have on day one, and building it before there's traffic to learn from is designing for a hypothetical. The interface (`predict_route(state) -> Route`) is built so a *learned* router can be dropped in later as an optional, opt-in upgrade once a project has enough trace volume — it's a future extension point, not a v1 dependency.
  - *Slow tier* — falls back to an LLM-based planner when the fast tier's confidence is below threshold, or the task is novel. This tier does real decomposition: goal → DAG of subtasks with dependencies.
- **Parallel-to-finish scheduling**: the planner doesn't commit to a static DAG upfront. It re-plans continuously as branches complete, launches every branch whose dependencies are already satisfied, and races redundant paths — e.g. two retrieval strategies run concurrently, whichever produces a sufficient answer first wins and the other is cancelled — rather than always waiting for every branch to finish.
- **Loop control lives here**: subsumes the `Loop` primitive's stop/continue decision from 3.2. Instead of a fixed `max_iterations` or asking the LLM to self-assess every time, the brain estimates marginal value of another iteration against remaining budget (3.8) and stops when it's not worth it.
- **Budget-aware scheduling**: a 5-way parallel fan-out is a scheduling decision made against remaining token/cost/time budget, not launched blind and cancelled after the fact.
- Fully traced (3.5): every routing decision — which tier answered, confidence score, what was launched/cancelled — is a first-class event, so a bad routing call is debuggable, not a black box on top of an already-complex system.

### 3.15 Self-Healing & Error Recovery (`kel.heal`)
When a step in a multi-agent loop fails — tool error, malformed/invalid output, provider timeout or rate-limit, stuck loop caught by 3.2's detector, an agent producing an answer that fails its own eval/schema check — the system should diagnose and attempt repair itself before surfacing the failure to the user.

- **Detection**: every failure mode already surfaces as a structured event via observability (3.5) — exceptions, schema-validation failures, stuck-loop signals from 3.2, budget-near-exhaustion warnings from 3.8. `kel.heal` subscribes to these rather than needing its own separate monitoring.
- **Diagnosis via LLM**: on a failure event, kel.heal builds a diagnostic prompt from the failure's trace context (what was attempted, the error/output, recent history) and sends it to a model — using the **user's configured API/model** (their own gateway credentials from 3.1, so diagnosis quality/cost is the user's choice, e.g. a cheap model for common cases, frontier model for gnarly ones) — asking it to classify the root cause and propose a fix strategy.
- **Healing strategies** the diagnosis can select between (each is an existing kel primitive, not new machinery):
  - retry with a corrected/clarified prompt or tool arguments
  - fall back to a different model/provider for that step (3.1's gateway makes this a swap)
  - roll back to the last good checkpoint (3.10) and re-plan a different branch via the brain (3.14)
  - hand off to a different agent better suited to the failure (3.11)
  - narrow scope / reduce the request (e.g. smaller retrieval window, simpler subtask) and retry
  - escalate to human-in-the-loop when none of the above is safe or budget allows no more attempts
- **Idempotency guardrail — the most important constraint on this subsystem**: auto-healing (retry/rollback/replan) is only ever applied automatically to read-only or explicitly-marked-idempotent tool calls by default. Any tool with side effects (payments, sending messages, writes to external systems, destructive operations) must be explicitly allow-listed for autonomous retry in its spec, or self-heal always stops and escalates to a human on failure. An agent silently "fixing itself" by re-firing a non-idempotent action is a worse failure mode than the original error — this guardrail is non-negotiable, not a nice-to-have.
- **Bounded, not infinite**: healing attempts are themselves loop iterations subject to 3.2's stuck-loop detection and 3.8's budget — a max-heal-attempts ceiling per node prevents "self-healing" from becoming its own runaway loop. If exhausted, fail loudly with the full diagnostic trail attached, never silently.
- **Learns over time**: every (failure → diagnosis → fix → outcome) record feeds back into the brain's fast-tier router training data (3.14), so recurring failure patterns get routed around preemptively in future runs instead of being diagnosed fresh every time.
- Fully traced like everything else: a healed run shows the failure, the diagnosis, and the fix in the trace timeline — self-healing must stay debuggable, not paper over what happened.

## 4. Suggested build order

1. **Model Gateway** — smallest surface, immediately useful standalone, everything else depends on it.
2. **Observability** — instrument the gateway first so every later subsystem is traced from day one.
3. **Budget** — cheap to add, prevents runaway costs while building the rest.
4. **Context/Loop primitives** — needed before any real agent runtime.
5. **Memory** — working + episodic first, semantic/procedural later.
6. **Retrieval/RAG** — start with one vector store adapter (e.g. Chroma for local dev, Qdrant for prod) + naive/hybrid RAG.
7. **Prompt specs (.md)** — needed before writing real agents so they're not hardcoded strings from the start.
8. **Runtime/execution graph** — ties gateway + context + memory + retrieval together into actual agents.
9. **Multi-agent orchestration** — supervisor/swarm/parallel patterns and the shared-context bus, built on top of the runtime graph.
10. **Central planner/router ("the brain")** — starts as the rule-based fast tier + LLM slow tier; the learned ANN router only makes sense once 3.5's traces exist to train it, so it arrives after everything that generates those traces.
11. **Self-healing** — needs checkpointing (3.10), budget (3.8), and the brain (3.14) already in place to have somewhere to roll back to and something to re-plan with.
12. **SDK & CLI** — wrap the above in the ergonomic API/CLI so the project is actually usable, not just internally coherent.
13. **Testing framework** — once there's something to test.
14. **Real-time multimodal (voice/video/lipsync)** — highest complexity, most infra-dependent (media servers, streaming codecs); build last, on a stable core.
15. **UI dashboard** — once there's enough trace/state data worth visualizing.

## 5. Open questions

- Package layout: monorepo with `kel-core`, `kel-observability`, `kel-<provider>` adapters as separate installable extras, or single package with optional deps?
- Sync-first (simplicity) vs async-first (throughput) core API? (Real-time voice/video in 3.12 pushes hard toward async-first.)
- How opinionated should the default Grafana dashboards be vs. just shipping OTel and letting users build their own?
- Real-time media (3.12) likely needs a non-Python hot path eventually (WebRTC media servers, audio buffering) — does that mean a Rust/Go sidecar from day one, or Python-first with a documented migration path once latency numbers demand it?
- Multi-agent credential model (3.11): should mixed single-key/multi-key graphs be allowed in the same run, or should credential strategy be an all-or-nothing choice per graph for simplicity?

## 6. End-to-end execution flow (ASCII)

```
                                   USER / DEVELOPER
                                          |
                                          v
   ┌──────────────────────────────────────────────────────────────────────┐
   │  kel.sdk  /  CLI (kel run)  /  local UI dashboard          (3.13)     │
   │  loads agent + prompt specs from versioned .md files        (3.6)     │
   └──────────────────────────────────┬───────────────────────────────────┘
                                       v
   ┌──────────────────────────────────────────────────────────────────────┐
   │                    kel.brain — central planner/router       (3.14)    │
   │                                                                       │
   │   incoming goal/state                                                │
   │        │                                                             │
   │        v                                                             │
   │   [ fast tier: lightweight ANN router ] ──confident?──yes──┐         │
   │        │ no / novel situation                              │         │
   │        v                                                    │         │
   │   [ slow tier: LLM planner ]                                │         │
   │     goal → DAG of subtasks + dependencies                   │         │
   │        │                                                    │         │
   │        └───────────────────────┬────────────────────────────┘         │
   │                                v                                     │
   │                        route / schedule decision                     │
   │              (continuously re-planned as branches complete;          │
   │               budget-aware — 3.8 — parallel fan-out sized to what    │
   │               remaining budget allows)                               │
   └──────────────────────────────────┬───────────────────────────────────┘
                                       v
   ┌──────────────────────────────────────────────────────────────────────┐
   │            kel.runtime — checkpointed, cyclic execution DAG (3.10)    │
   │                                                                       │
   │        ┌───────────────┬───────────────┬───────────────┐            │
   │        v               v               v               v            │
   │  ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐      │
   │  │ Agent A   │   │ Agent B   │   │ Agent C   │   │ Agent N…  │      │
   │  │ (own or   │   │ (own or   │   │ (own or   │   │           │      │
   │  │ shared    │   │ shared    │   │ shared    │   │  kel.agents│      │
   │  │ API key)  │   │ API key)  │   │ API key)  │   │   (3.11)  │      │
   │  └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘      │
   │        │  all agents read/write a SHARED CONTEXT BUS (3.11)  │      │
   │        └───────────────┴───────┬───────┴───────────────┘            │
   └────────────────────────────────┼───────────────────────────────────┘
                                     v
        ┌────────────────────────────────────────────────────────────┐
        │  per-agent step: what one agent invocation actually touches │
        │                                                              │
        │   kel.context (3.2)        kel.memory (3.3)      kel.retrieval (3.4)
        │   ┌────────────────┐     ┌─────────────────┐    ┌───────────────────┐
        │   │ ContextWindow  │     │ working          │    │ vector store       │
        │   │ token budget,  │<--->│ episodic         │<-->│ adapter (hybrid /  │
        │   │ eviction,      │     │ semantic         │    │ graph / agentic    │
        │   │ Loop primitive │     │ procedural (.md) │    │ RAG, multimodal)   │
        │   └───────┬────────┘     └──────────────────┘    └────────────────────┘
        │           v                                                          │
        │   ┌──────────────────────────────────────────────────────┐          │
        │   │        kel.models — Model Gateway (3.1)               │          │
        │   │  normalizes auth / tool schema / streaming / vision   │          │
        │   └───────┬─────────────────────────────┬──────────────────┘         │
        │           v                             v                           │
        │   Anthropic · OpenAI · Gemini    local / Ollama · Bedrock · Azure    │
        └──────────────────────┬───────────────────────────────────────────────┘
                                │
              every call/step emits a structured event, always:
                                │
        ┌───────────────────────┼─────────────────────────┬─────────────────────┐
        v                       v                         v                     v
 ┌─────────────┐        ┌──────────────┐          ┌───────────────┐    ┌────────────────┐
 │kel.observ-  │        │ kel.budget   │          │  kel.heal      │    │ kel.storage    │
 │ability (3.5)│        │ (3.8)        │          │  (3.15)        │    │ (3.9)          │
 │ OTel spans →│        │ tokens/cost/ │  failure  │ diagnose via   │    │ checkpoints,   │
 │ Grafana /   │        │ time/tool-   │──event──→ │ user's LLM API │    │ traces, media  │
 │ Tempo/Loki/ │        │ call ceiling │           │ → pick strategy│    │ blobs, content-│
 │ Prometheus  │        │ per call     │           │  retry /       │    │ addressed,     │
 └─────────────┘        └──────┬───────┘           │  fallback model│    │ replayable     │
                                │ exceeded?         │  / rollback+   │    └────────────────┘
                                v                   │  replan (→brain│
                       truncate / summarize /       │  3.14) / hand- │
                       abort per policy              │  off agent /   │
                                                      │  narrow scope /│
                                                      │  escalate human│
                                                      │  (bounded by   │
                                                      │  loop guard)   │
                                                      └───────┬────────┘
                                                              │
                     healed run feeds back as training data  │
                     into kel.brain's fast-tier router  ◄─────┘
                                     │
                                     v
   ┌──────────────────────────────────────────────────────────────────────┐
   │  back to kel.brain: merge branch results, cancel losing parallel      │
   │  branches once a sufficient answer exists, decide continue-vs-stop    │
   │  (marginal value vs. remaining budget) — loop back into kel.runtime   │
   │  or finish                                                            │
   └──────────────────────────────────┬───────────────────────────────────┘
                                       v
                              final result assembled
                                       │
                                       v
                        back to kel.sdk / CLI / UI  →  USER


   ── parallel special-purpose path, same hooks apply throughout ──

   ┌──────────────────────────────────────────────────────────────────────┐
   │        kel.realtime — voice / video / lipsync agents        (3.12)    │
   │                                                                       │
   │   mic/camera in → streaming STT ─┬─→ fast path: filler/direct answer  │
   │                                  └─→ slow path: kel.brain → RAG/tools │
   │   dual-path responses merge → streaming TTS (speaks on first clause)  │
   │   avatar/lipsync render joins the call as its own media participant   │
   │   (no round-trip through a second encode/decode pass)                │
   │   → all events still flow through observability / budget / heal      │
   └──────────────────────────────────────────────────────────────────────┘

   ── offline / CI path ──

   ┌──────────────────────────────────────────────────────────────────────┐
   │  kel.testing (3.7): kel eval / kel replay                             │
   │  pulls recorded traces from kel.storage, replays deterministically    │
   │  against golden-trace assertions and budget/loop invariants           │
   └──────────────────────────────────────────────────────────────────────┘
```


## 7. Final verdict — what's good, what was bad, what got cut

A 15-subsystem framework that tries to ship everything at once is exactly how "LangChain killers" die — never shipping, or shipping shallow. Being ruthless about scope now is what makes this credible instead of a wishlist.

**Genuinely strong, keep as the actual differentiators (this is what wins):**
- **Observability-by-default (3.5)** — the single biggest real gap in LangChain. Nobody ships a framework where every span is traced from line one. This is the headline feature, not an add-on.
- **Budget/token control threaded through every call (3.8)** — real, common production pain; small to build; huge trust payoff.
- **Context/loop engineering with stuck-loop detection (3.2)** — directly matches the #1 reported 2026 failure mode (context failures, not model failures).
- **Deterministic record/replay testing (3.7)** — nobody does this well today; if kel nails it, it's a reason to switch on its own.
- **Shared-context-bus multi-agent (3.11)** — targets a specific, named, repeatedly-reported failure ("agent 3 invents context agent 2 never produced"), not a vague "multi-agent support" checkbox.
- **.md prompt/agent specs (3.6)** — low-risk, high-ergonomics, genuinely differentiated from string-literal prompts.

**Right idea, was overbuilt as originally written — descoped, not cut:**
- **The "brain" router (3.14)** — the *concept* (fast cheap routing, slow LLM planning as fallback, parallel-to-finish, budget-aware loop control) is good and stays. The *originally-stated mechanism* — a custom-trained ANN needing a training pipeline and model-serving infra — was premature ML-ops for a project with no traffic yet to train on. Cut down to: rules + embedding-similarity in v1, learned router as a strictly optional later upgrade.
- **Self-healing (3.15)** — the concept is a real differentiator (nobody else does diagnosis-driven repair). What was missing was a hard safety line: it must never auto-retry a side-effecting action (payment, message send, destructive write) without explicit opt-in per tool. Added as a non-negotiable guardrail — without it, this feature is a liability, not a selling point.
- **"Supports all models" (3.1)** — reframed from an open-ended promise (unsustainable maintenance burden) to a small core-maintained set plus a stable adapter contract for everything else.

**Actually bad as originally scoped — cut down hard:**
- **Custom real-time media infrastructure (3.12)** — building a WebRTC media server, GPU render pipeline, and codec stack to compete on voice/lipsync latency is a different, infrastructure-heavy business from an agent-orchestration framework. This was the single riskiest part of the plan: it doesn't reuse anything else in kel, requires entirely different engineering skills, and existing vendors (LiveKit, Daily, Deepgram, Tavus) already race on exactly this metric. **Cut from core, shipped as an optional adapter package (`kel-realtime`) that orchestrates existing best-in-class infra instead of rebuilding it.** kel's job stays orchestration (dual-path responses, routing, budget, observability) — the thing that's actually its differentiator.
- **Literal "kernel"/OS-level work** (from the original brief) — never made it into the design as literal kernel engineering, and stays that way. What's in 3.9 (async I/O, zero-copy for local media transfer, content-addressed storage) is legitimate systems-level care applied to a Python framework, not an actual OS kernel. Worth saying explicitly: kel is not building a kernel.
- **15 subsystems as one simultaneous v1** — the plan itself, not any single subsystem, was the biggest risk. Fixed by the build order already in §4, now read as a hard gate, not a suggestion: **v1 = Model Gateway → Observability → Budget → Context/Loop → Memory(working+episodic) → Retrieval(one adapter) → Prompt specs → Runtime graph.** Multi-agent, the brain, self-healing, SDK/UI polish, testing framework, and real-time are v2+/optional packages. Nothing after v1 ships until v1 is something people actually use.

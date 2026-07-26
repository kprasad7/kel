# kel — Usage Guide

This is a working guide to the parts of kel that exist today. For the
architecture rationale and what's still missing, see [DESIGN.md](DESIGN.md).
Every snippet below uses the real public API (checked against `src/kel/`);
none of this is aspirational.

## Install

```bash
pip install pykel               # from PyPI — the distribution name is
                                 # "pykel" (the "kel" name was already
                                 # registered), but `import kel` and the
                                 # `kel` CLI command are unaffected
pip install "pykel[anthropic]"  # + Anthropic provider
pip install "pykel[openai]"     # + OpenAI provider
pip install "pykel[cohere]"     # + Cohere provider
pip install "pykel[otel]"       # + Grafana/OTLP trace export
pip install "pykel[s3]"         # + S3-compatible blob storage
pip install "pykel[all]"        # everything
```

Working from a clone instead of PyPI:

```bash
pip install -e ".[dev]"        # core + test tooling
pip install -e ".[all]"        # everything
```

---

## 1. Model Gateway (`kel.models`)

One interface for every provider. `get_model` wraps the result with
tracing (and budget, if given) by default.

```python
from kel import get_model, Message

model = get_model("anthropic:claude-sonnet-5", api_key="sk-...")
response = model.generate([Message.user("Hello!")])
print(response.text)

# swap provider with no other code changes
model = get_model("openai:gpt-5.2", api_key="sk-...")
model = get_model("cohere:command-r-plus", api_key="...")
```

Adding a new provider means writing one adapter file and calling
`register_provider("myprovider", factory)` — see
`src/kel/models/providers/cohere.py` for a template (request/response
mapping isolated in one file, lazy-imports the vendor SDK, translates
vendor exceptions into kel's `ProviderError`/`AuthenticationError`/
`RateLimitError`).

Streaming:

```python
for event in model.stream([Message.user("Tell me a story")]):
    if hasattr(event, "text"):
        print(event.text, end="")
```

Tool calling — pass `tools=[...]`, inspect `response.tool_calls`:

```python
from kel import ToolSpec

tools = [ToolSpec(name="get_weather", description="...", input_schema={"type": "object", "properties": {"city": {"type": "string"}}})]
response = model.generate([Message.user("weather in Paris?")], tools=tools)
for call in response.tool_calls:
    print(call.name, call.input)
```

Opt out of tracing/budget wrapping when you want the raw adapter:

```python
model = get_model("anthropic:claude-sonnet-5", instrument=False)
```

**Error hierarchy** — every vendor SDK's own exception is translated
into one of these before it reaches your code, so error handling doesn't
need a different `except` clause per provider:

```python
from kel.models import AuthenticationError, RateLimitError, ProviderError, ModelNotFoundError, KelError

try:
    response = model.generate([Message.user("hi")])
except AuthenticationError:
    ...   # bad/missing API key
except RateLimitError:
    ...   # vendor 429 — safe to back off and retry
except ModelNotFoundError:
    ...   # get_model("provider:bad-model-id")
except ProviderError:
    ...   # any other vendor-side failure, not one of the above
except KelError:
    ...   # base class for everything above, if you just want one catch-all
```

---

## 2. Observability (`kel.observability`)

Every `generate()`/`stream()` call through `get_model()` is traced
automatically — nothing to opt into. Default sink prints to the console;
swap in a real backend or collect spans in memory:

```python
from kel.observability import configure, add_sink, ConsoleSink, ListSink

spans = ListSink()
add_sink(spans)          # keep console output, also capture spans in memory
# configure([ConsoleSink()])  # or replace the sink list entirely
```

Export to Grafana (via an OTel Collector → Tempo), requires `pykel[otel]`:

```python
from kel.observability.otel import configure_otlp
configure_otlp(endpoint="http://localhost:4317", service_name="my-agent")
```

Wrap a model manually (this is what `get_model` does internally):

```python
from kel.observability import InstrumentedChatModel
from kel.models.registry import get_model
raw = get_model("anthropic:claude-sonnet-5", instrument=False)
traced = InstrumentedChatModel(raw)
```

Silence tracing entirely (e.g. in a test that doesn't care about spans)
with `NullSink`, or reach the process-wide `Tracer` directly (`get_tracer`
is what `add_sink`/`configure` operate on under the hood):

```python
from kel.observability import NullSink, configure, get_tracer

configure([NullSink()])
tracer = get_tracer()
tracer.sinks   # the currently configured list of Sink instances
```

---

## 3. Budget (`kel.budget`)

```python
from kel import get_model, Message
from kel.budget import Budget

model = get_model("anthropic:claude-sonnet-5", budget=Budget(max_tokens=50_000, max_cost_usd=2.0))
model.generate([Message.user("hi")])   # raises BudgetExceededError once the limit trips
```

Share one budget across multiple models/agents by passing a `BudgetTracker` instead of a `Budget`:

```python
from kel.budget import Budget, BudgetTracker

tracker = BudgetTracker(Budget(max_cost_usd=5.0))
model_a = get_model("anthropic:claude-sonnet-5", budget=tracker)
model_b = get_model("openai:gpt-5.2", budget=tracker)   # same pool
snapshot = tracker.snapshot()   # a BudgetSnapshot: tokens_used, cost_usd_used, *_remaining
```

Checking a model's per-token pricing directly, without making a call —
`estimate_cost_usd`/`is_priced` are what `Budget` uses internally, from a
small hardcoded pricing table (unknown models cost $0.0, not a guess):

```python
from kel.budget import estimate_cost_usd, is_priced
from kel.models import Usage

is_priced("anthropic", "claude-sonnet-5")   # True — has known per-token pricing
estimate_cost_usd("anthropic", "claude-sonnet-5", Usage(input_tokens=1000, output_tokens=500))
```

---

## 4. Context & Loop (`kel.context`)

```python
from kel.context import ContextWindow, sliding_window_eviction, make_summarization_eviction
from kel.models import Message

window = ContextWindow(max_tokens=8000, policy=sliding_window_eviction)
window.add(Message.user("..."))
window.tokens_used, window.tokens_remaining, window.evicted_count
```

Summarize-on-overflow instead of dropping the oldest messages:

```python
def summarize(messages):
    return Message.assistant(f"summary of {len(messages)} earlier turns")

window = ContextWindow(max_tokens=4000, policy=make_summarization_eviction(summarize, keep_recent=4))
```

Bounded loops with stuck-loop detection:

```python
from kel.context import Loop, LoopBudgetExceededError, StuckLoopError

loop = Loop(max_iterations=10, stuck_window=4, stuck_threshold=3)
while True:
    loop.step()                       # raises LoopBudgetExceededError past max_iterations
    action = decide_next_action()
    loop.record_action(signature_of(action))   # raises StuckLoopError on repeated no-progress
    if done:
        break
```

---

## 5. Memory (`kel.memory`)

```python
from kel.memory import Memory
from kel.models import Message

memory = Memory(session_id="user-42")
memory.remember_turn(Message.user("hi"))
memory.remember_turn(Message.assistant("hello!"))

memory.working.messages          # current context window
memory.episodic.transcript("user-42")   # full session history
```

**Recalling a session across process restarts / script reruns.** By
default `Memory()` uses an `InMemoryEpisodicStore`, which doesn't outlive
the process — a fresh `Memory(session_id="user-42")` in a new process
starts empty. Pass a durable store instead (`FileEpisodicStore`,
`SQLiteEpisodicStore` — a single file usable from multiple *processes*
sharing it, not just multiple in-process reconstructions — or your own
backend) and working memory is seeded automatically from that session's
existing transcript:

```python
from kel.memory import FileEpisodicStore, Memory

episodic = FileEpisodicStore("./sessions")

# first run
memory = Memory(session_id="user-42", episodic=episodic)
memory.remember_turn(Message.user("what's the capital of France?"))
memory.remember_turn(Message.assistant("Paris"))

# a later run (new process, or a Streamlit-style script rerun) —
# same store, same session_id: working memory already has both turns
memory2 = Memory(session_id="user-42", episodic=episodic)
memory2.working.messages  # ["what's the capital of France?", "Paris"]
```

This is exactly what `Agent(..., memory=Memory(session_id=..., episodic=episodic))`
needs for a web UI that recreates the `Agent` object on every request/rerun
but wants the conversation to persist — construct `Memory` yourself with a
shared `episodic` store instead of letting `Agent` default to a fresh one.

`SQLiteEpisodicStore` is the same idea as `FileEpisodicStore`, but a
single SQL-queryable file safe to share across multiple *processes* at
once (several worker processes behind a web server, all resuming the
same sessions) — `FileEpisodicStore` is one-JSONL-file-per-session and
isn't safe for concurrent multi-process writes to the same session:

```python
from kel.memory import SQLiteEpisodicStore, Memory

episodic = SQLiteEpisodicStore("./sessions.sqlite")
memory = Memory(session_id="user-42", episodic=episodic)
memory.remember_turn(Message.user("hi"))
# any other process opening SQLiteEpisodicStore("./sessions.sqlite") with
# the same session_id sees this turn immediately
```

Semantic memory (long-term facts), procedural memory (learned patterns as `.md`):

```python
from kel.memory import SemanticMemory, ProceduralMemory

facts = SemanticMemory()
facts.remember("user prefers dark mode")                          # never expires
facts.remember("mentioned they're traveling this week", ttl_seconds=86400)  # decays after a day
facts.search("UI preferences")           # keyword search by default; never returns expired facts
facts.forget_expired()                   # reclaim storage; returns how many were purged

procedures = ProceduralMemory("./procedures")
procedures.save("retry-pattern", "# Retry\nAlways retry idempotent calls once.")
```

Consolidate a session's transcript into a semantic fact:

```python
from kel.memory import consolidate
consolidate(memory.episodic, "user-42", facts, summarize=lambda msgs: llm_summarize(msgs))
```

---

## 6. Retrieval / RAG (`kel.retrieval`)

```python
from kel.retrieval import InMemoryVectorStore, NaiveHashEmbedder, Retriever

retriever = Retriever(InMemoryVectorStore(), embedder=NaiveHashEmbedder(dims=128))
retriever.ingest(open("docs/handbook.md").read(), id_prefix="handbook")

results = retriever.retrieve("how do refunds work?", k=5)          # naive top-k
results = retriever.retrieve_hybrid("refund policy", k=5)          # vector + keyword blend
for r in results:
    print(r.score, r.chunk.text[:100])
```

`NaiveHashEmbedder` is zero-dependency and good for local dev only — pass
any `Callable[[str], list[float]]` (e.g. a real embedding API call) instead
for production relevance. `retrieve`/`retrieve_hybrid` return
`list[ScoredChunk]` — each a `.score` plus a `.chunk` (`Chunk`: `id`,
`text`, `metadata`, `embedding`).

Splitting text yourself instead of `retriever.ingest`'s default splitter
— `split_text` is fixed-size sliding-window chunking (simple, can split
mid-sentence); `recursive_split_text` (structure-aware: paragraphs, then
sentences, then words) is preferred for real documents and is what
`ingest` uses by default:

```python
from kel.retrieval import split_text, recursive_split_text

chunks = split_text(long_text, chunk_size=500, overlap=50)
chunks = recursive_split_text(long_text, chunk_size=500, overlap=50)   # prefer this for real documents
```

**Scoping a search to metadata** — every backend (InMemory, Qdrant,
Pinecone, Weaviate, Chroma, pgvector) supports `filter`, an exact-match-
on-every-key dict:

```python
retriever.retrieve("wireless mouse", k=5, filter={"user_id": "u1"})
retriever.retrieve_hybrid("wireless mouse", k=5, filter={"source": "amazon"})
```

**Reranking** — a second, more expensive relevance pass over a wider
first-stage candidate pool, using any `ChatModel` (no cross-encoder model
or new dependency needed):

```python
from kel.retrieval import LLMReranker

retriever = Retriever(store, embedder=embed, reranker=LLMReranker(get_model("anthropic:claude-sonnet-5")))
retriever.retrieve("how do refunds work?", k=5)  # overfetches, then reranks down to 5
```

**Document loaders** (`kel.retrieval`): `load_pdf`/`load_pdf_pages`
(`pip install kel[pdf]`), plus stdlib-only `load_html` and `load_csv`/
`load_csv_rows` — no extra dependency for either.

---

## 7. Agent Specs (`kel.specs`)

Define an agent as a `.md` file:

```markdown
---
name: weather-agent
model: anthropic:claude-sonnet-5
api_key_ref: ANTHROPIC_API_KEY
tools: [get_weather]
budget:
  max_tokens: 20000
---
You are a helpful weather assistant. Always cite your data source.
```

```python
from kel.specs import load_agent_spec, AgentSpecLoader

spec = load_agent_spec("agents/weather.md")
spec.model, spec.system_prompt, spec.api_key_ref

loader = AgentSpecLoader("agents/weather.md")   # hot-reloads when the file's mtime changes
spec = loader.load()
```

Eval cases, also `.md`:

```markdown
---
cases:
  - name: basic
    input: "weather in Paris?"
    expected_contains: ["Paris"]
---
```

```python
from kel.specs import load_eval_cases, run_eval_suite

cases = load_eval_cases("agents/weather.eval.md")
results = run_eval_suite(cases, respond=lambda text: agent.run(text).text)   # list[EvalResult]
for result in results:
    result.case, result.output, result.passed   # substring-match pass/fail
```

Running a single case (what `run_eval_suite` calls in a loop) directly:

```python
from kel.specs import run_eval_case

result = run_eval_case(cases[0], respond=lambda text: agent.run(text).text)
```

Grading by criteria instead of substring match — `llm_judge` is the
one-call primitive `run_llm_graded_eval_suite` (§17) is built on, useful
directly when you want a `Grade` (`passed`/`score`/`reasoning`) for a
single input/output pair without a full `EvalCase`:

```python
from kel.specs import llm_judge

grade = llm_judge(
    judge_model,
    input="what's the capital of France?",
    output=response.text,
    criteria="the answer correctly names Paris",
)
grade.passed, grade.score, grade.reasoning
```

---

## 8. Agents (`kel.agents`)

A single tool-calling agent (bounded by `kel.context.Loop` under the hood):

```python
from kel.agents import Agent, Tool
from kel import get_model

def get_weather(input: dict) -> str:
    return f"sunny in {input['city']}"

tool = Tool(name="get_weather", description="get current weather", input_schema={"type": "object", "properties": {"city": {"type": "string"}}}, fn=get_weather)

agent = Agent("weather-agent", get_model("anthropic:claude-sonnet-5"), system_prompt="Be concise.", tools=[tool])
response = agent.run("what's the weather in Paris?")
print(response.text)
```

Generation config (`max_tokens`/`temperature`) is set on the `Agent`, not
per `.run()` call, and forwarded to every model call (`run`/`arun`/
`run_stream`/`arun_stream`). Leave them unset and each provider adapter's
own default applies (e.g. 1024 tokens) — passing them overrides it:

```python
agent = Agent("weather-agent", get_model("anthropic:claude-sonnet-5"), max_tokens=4096, temperature=0.2)
```

If a model turn ever comes back with no text and no tool calls (a
truncated/degenerate response, not a legitimate empty answer), `Agent`
raises `EmptyModelResponseError` (`kel.agents.errors`) instead of storing
it into memory — an empty assistant turn left in shared history would
otherwise corrupt every later question in the same session.

When a model requests several tools in one turn, `Agent` runs them
concurrently (not one at a time) — `run`/`arun` collect all results
before continuing; `run_stream`/`arun_stream` yield each `ToolResultEvent`
as its own call actually finishes, so a UI can show real per-tool
progress, not just a report after the slowest one.

**Approving tool calls before they run** — `approve_tool_call` is an
injected `(name, input) -> bool` gate, checked once per call across every
run variant. Return `False` to reject a call before it executes:

```python
def approve(name: str, input: dict) -> bool:
    return name != "delete_file" or input.get("confirmed") is True

agent = Agent("assistant", model, tools=[delete_tool], approve_tool_call=approve)
```

A rejected call never runs — the agent sees an error tool result
("rejected by approval hook") instead, same as any other tool failure.
Leaving it unset (the default) approves every call, matching prior
behavior. `approve_tool_call`'s type is exported as `ApprovalHook`
(`kel.agents`) if you want to annotate a standalone function with it.

**Concurrent calls on one shared `Agent` instance are safe, but still
one conversation.** `run`/`arun`/`run_stream`/`arun_stream` each acquire
a per-instance lock (`threading.Lock` for the sync pair, `asyncio.Lock`
for the async pair) around the read-model-call-write-memory sequence, so
two threads/tasks calling into the *same* `Agent` at once can't interleave
and corrupt `memory.remember_turn()`. That only guarantees no corruption
— it does not give each caller a private conversation. If you want
per-user isolation (e.g. serving many users through `kel.sdk`'s adapters),
construct one `Agent` per session instead of sharing one — see §14's
per-session factory support.

Multi-agent patterns — each `Agent` can use its own model/API key, or share one:

```python
from kel.agents import sequential_pipeline, run_parallel, run_supervisor, run_swarm
from kel.runtime import run_graph

# sequential: each agent sees every upstream agent's output via shared graph state
graph = sequential_pipeline([researcher, writer, editor])
result = run_graph(graph, {"input": "write a summary of Q3 earnings"})
result.state["writer_output"]

# parallel fan-out + merge
result = run_parallel({"a": agent_a, "b": agent_b}, "task")

# supervisor delegates to named workers via a small text protocol
result = run_supervisor(supervisor, {"worker1": worker}, "big task")

# swarm: peer agents hand off to each other
result = run_swarm({"a": agent_a, "b": agent_b}, start_agent="a", task="task")
```

**Scoping what each agent sees.** With many agents in one pipeline,
every downstream agent seeing every upstream output verbatim gets noisy.
`context_selector`/`results_selector` filter the shared state before an
agent (or the supervisor) sees it:

```python
def only_from(*names):
    return lambda agent_name, state: {
        k: v for k, v in state.items() if any(k.startswith(f"{n}_") for n in names)
    }

sequential_pipeline([researcher, writer, editor], context_selector=only_from("researcher"))
# editor only sees researcher's output, not writer's intermediate draft

run_supervisor(supervisor, workers, task, results_selector=lambda results: dict(list(results.items())[-3:]))
# supervisor only sees the 3 most recent results, not the whole growing history
```

**Dynamic, cyclic multi-agent flows.** The four patterns above are fixed
shapes; for anything more dynamic (conditional branching, loops back to
an earlier agent), build directly on `kel.runtime.Graph` — which already
supports conditional edges and cycles — using `agent_node()` to wrap any
`Agent` as a Graph node function:

```python
from kel.agents import agent_node
from kel.runtime import Graph, END, run_graph

graph = Graph(entry="draft")
graph.add_node("draft", agent_node(writer))
graph.add_node("validate", agent_node(validator, input_key="draft_output"))
graph.add_node("revise", agent_node(reviser, input_key="draft_output", output_key="draft_output"))
graph.add_edge("draft", "validate")
graph.add_conditional_edges("validate", lambda s: "revise" if "FAIL" in s["validator_output"] else END)
graph.add_edge("revise", "validate")  # loop back until validation passes

result = run_graph(graph, {})
```

**Checking a response for hallucinated claims.** `HallucinationChecker`
is a second, more expensive pass — the same one-call structured-output
pattern `kel.retrieval.LLMReranker` uses — that checks whether a
response's claims are actually supported by given source material (RAG
chunks, tool results, anything). Not wired into `Agent` automatically;
run it yourself after `agent.run(...)`:

```python
from kel.agents import HallucinationChecker

checker = HallucinationChecker(model)
response = agent.run("summarize the incident report")
report = checker.check(response.text, sources=retrieved_chunks_text)

if not report.grounded:
    report.unsupported_claims  # specific claims not backed by the sources
    # retry, ask for citations, escalate to a human, etc.
```

**Reverse-feedback / reflection loops.** A downstream critic correcting
an upstream generator is already expressible as a cyclic `Graph`
(`agent_node()` + a conditional edge routing back); `reflect_and_retry`
packages that same pattern as a ready-made helper for the common
one-generator/one-critic case, so you don't hand-wire a `Graph` for it
every time:

```python
from kel.agents import reflect_and_retry

def critic(response_text: str) -> tuple[bool, str]:
    if len(response_text) < 100:
        return False, "too brief — add more supporting detail"
    return True, ""

result = reflect_and_retry(agent, "explain the incident", critic=critic, max_attempts=3)
result.response.text   # the last attempt, whether or not it was ultimately accepted
result.accepted        # False if max_attempts was reached without the critic ever accepting
result.attempts        # ReflectionResult — list of every (response, feedback) attempt, in order
```

`areflect_and_retry` is the same loop with an async agent and an async
critic — useful when the critic itself calls a model (e.g. a
`HallucinationChecker` or a judge `Agent`):

```python
from kel.agents import areflect_and_retry

async def async_critic(response_text: str) -> tuple[bool, str]:
    report = await hallucination_checker.acheck(response_text, sources=chunks)
    return report.grounded, "; ".join(report.unsupported_claims)

result = await areflect_and_retry(agent, "summarize the report", critic=async_critic, max_attempts=3)
```

Each rejected attempt's feedback is fed back to the agent as the next
turn's input — the critic (downstream) correcting the generator
(upstream), not just a linear forward pass. `critic` can be anything:
a plain rule (shown above), a `HallucinationChecker`, or another `Agent`
acting as a judge.

---

## 9. Runtime / Execution Graph (`kel.runtime`)

The lower-level primitive the `kel.agents` patterns are built on — a
cyclic DAG with real concurrent fan-out and checkpointing:

```python
from kel.runtime import Graph, END, run_graph, InMemoryCheckpointStore

graph = Graph(entry="fetch")
graph.add_node("fetch", lambda state: {"count": state.get("count", 0) + 1})
graph.add_conditional_edges("fetch", lambda state: END if state["count"] >= 3 else "fetch")

store = InMemoryCheckpointStore()
result = run_graph(graph, {}, checkpoint_store=store, run_id="run-1")
result.state, result.history
store.history("run-1")   # every node transition, for replay/debugging
```

Fan-out to multiple nodes in one layer runs them concurrently (real threads, not simulated):

```python
graph.add_conditional_edges("router", lambda state: ["path_a", "path_b"])
```

The concurrency ceiling per layer defaults to 8 — raise it for a
genuinely wide fan-out (dozens of parallel branches in a large
multi-agent flow); it's I/O-bound work (model/tool calls), so threads
scale past the CPU core count fine:

```python
run_graph(graph, {}, max_workers=32)
```

**Time travel: rewinding to an arbitrary historical checkpoint.**
`resume_graph()` only continues from an `Interrupt`'s pause point — to
rewind to any earlier step and branch off a new timeline (e.g. a 15-step
run failed at step 12; go back to step 11, patch the broken variable,
and continue):

```python
from kel.runtime import fork_from_checkpoint

checkpoints = store.history("run-1")
step_11 = next(c for c in checkpoints if c.step == 11)

forked = fork_from_checkpoint(graph, step_11, state_overrides={"retry_count": 0})
forked.run_id  # a new run_id — a branch, not a continuation of run-1
```

**Fallback routing on node failure.** A node raising an exception used
to always crash the whole run. `set_fallback` routes to a designated
node instead, with the error captured into `state["__error__"]` for the
fallback to actually inspect:

```python
graph.add_node("risky_step", call_flaky_api)
graph.add_node("recover", lambda state: {"answer": f"fallback after: {state['__error__']['error']}"})
graph.set_fallback("risky_step", "recover")
```

**Durable human-in-the-loop notifications.** `Interrupt` already lets a
run pause and `resume_graph()` continue it arbitrarily later — the
missing piece was telling a human it's waiting. `notify_interrupt` sends
one via any `Notifier` (a `WebhookNotifier` — Slack incoming webhooks,
PagerDuty, or your own endpoint — ships built in):

```python
from kel.runtime import WebhookNotifier, notify_interrupt

paused = run_graph(graph, {}, checkpoint_store=store, run_id="run-1")
if paused.interrupted:
    notify_interrupt(paused, WebhookNotifier("https://hooks.example.com/..."))
    # ... persist `paused` (or a Checkpoint) somewhere durable; whenever
    # the human actually responds, days later is fine, call resume_graph()
```

---

## 10. Brain — routing (`kel.brain`)

```python
from kel.brain import Brain, RuleRouter, EmbeddingRouter, Route

rules = RuleRouter(default="general_agent")
rules.add_rule(lambda state: "refund" in state["text"], "billing_agent", confidence=0.9)

def slow_tier(state):
    # fall back to an LLM decision when the fast tier isn't confident
    return Route(target=llm_pick_agent(state), confidence=1.0, tier="slow")

brain = Brain(fast_tier=rules.predict_route, slow_tier=slow_tier, confidence_threshold=0.6)
route = brain.route({"text": "I want a refund"})
```

Race redundant branches, take whichever finishes with a sufficient answer:

```python
from kel.brain import race_to_finish
result = race_to_finish({"vector_search": run_vector, "keyword_search": run_keyword})
result.winner, result.result   # a RaceResult
```

Deciding whether another loop iteration is worth running, given what's
left in a `BudgetTracker` — a fixed-reserve threshold check (not real
marginal-value estimation, see the module docstring), same call
signature a smarter version could fill in later:

```python
from kel.brain import should_continue

while should_continue(tracker.snapshot(), min_tokens_reserve=200, min_cost_reserve=0.01):
    step()
```

---

## 11. Self-healing (`kel.heal`)

```python
from kel.heal import Healer, make_llm_diagnoser
from kel import get_model

diagnose = make_llm_diagnoser(get_model("anthropic:claude-haiku-4-5"))  # cheap model for diagnosis
healer = Healer(diagnose=diagnose, max_attempts=3)

def call_flaky_api(diagnosis):
    return external_api.call()   # may raise

result = healer.run(call_flaky_api, idempotent=True, description="fetch account balance")
```

Every attempt is recorded on `healer.log` (a `list[HealAttempt]`); if
healing gives up entirely (max attempts exhausted, or the diagnosis
escalated to a human), `run` raises `HealExhaustedError`, which always
carries the full diagnostic trail rather than failing silently:

```python
from kel.heal import HealExhaustedError

try:
    result = healer.run(call_flaky_api, idempotent=True, description="fetch account balance")
except HealExhaustedError as exc:
    for attempt in exc.attempts:
        print(attempt.attempt, attempt.error, attempt.diagnosis.strategy, attempt.outcome)
```

**Learning from past failures**: feed a completed heal log into an
`EmbeddingRouter` (`kel.brain`) so future routing decisions can route
around error patterns a similar traffic shape has hit before:

```python
from kel.heal import feed_heal_log_into_router
from kel.brain import EmbeddingRouter

router = EmbeddingRouter(embedder=my_embedder)
feed_heal_log_into_router(healer.log, router)   # each attempt.error -> attempt.diagnosis.strategy
```

**The idempotency guardrail is not optional**: pass `idempotent=False` for
anything with side effects (payments, sends, destructive writes) — a
diagnosed "retry" is automatically downgraded to `escalate_human` and the
call is never auto-retried.

---

## 12. Testing (`kel.testing`)

Record real model calls once, replay them deterministically in CI:

```python
from kel.testing import RecordingChatModel, ReplayChatModel, Cassette

# record (run once, with a real API key)
recorder = RecordingChatModel(get_model("anthropic:claude-sonnet-5", instrument=False))
recorder.generate([Message.user("hi")])
recorder.cassette.save("tests/cassettes/basic.json")

# replay (in CI, no API key, no network)
cassette = Cassette.load("tests/cassettes/basic.json")
replay_model = ReplayChatModel(cassette)
```

Calling `replay_model.generate(...)` more times than the cassette has
recorded interactions raises `ReplayExhaustedError` — a real test failure
instead of silently reusing a stale response:

```python
from kel.testing import ReplayExhaustedError

try:
    replay_model.generate([Message.user("a question the cassette never recorded")])
except ReplayExhaustedError:
    ...  # the test made an extra call the recording didn't anticipate
```

Golden-trace assertions:

```python
from kel.testing import (
    assert_node_sequence,
    assert_nodes_visited,
    assert_no_error_spans,
    assert_span_names,
    assert_budget_never_exceeded,
)

assert_node_sequence(result.history, ["fetch", "fetch", "fetch"])   # exact order
assert_nodes_visited(result.history, {"fetch", "summarize"})        # subset, any order
assert_no_error_spans(spans.spans)          # spans from a ListSink
assert_span_names(spans.spans, ["model.generate", "tool.get_weather"])   # exact span-name sequence
assert_budget_never_exceeded(tracker)
```

---

## 13. Storage (`kel.storage`)

```python
from kel.storage import LocalBlobStore, ArtifactStore, FileCheckpointStore

blobs = LocalBlobStore("./data/blobs")
content_id = blobs.put(b"some bytes")     # id = sha256(content); re-put is free

artifacts = ArtifactStore(blobs, "./data/artifacts")
artifacts.save("trace-export", b'{"spans": []}', content_type="application/json")
artifacts.load("trace-export")

checkpoints = FileCheckpointStore("./data/checkpoints")   # drop-in for run_graph(checkpoint_store=...)
```

`FileCheckpointStore` uses pickle (state can hold arbitrary Python/pydantic
objects, not just JSON-safe values) but loads through a restricted
unpickler that only permits kel's own types plus a small builtin allowlist
— the mitigation for deserialization-of-untrusted-data attacks (the same
vulnerability class behind several disclosed CVEs in LLM orchestration
frameworks that pickle agent state by default). A tampered checkpoint
file that tries to instantiate an arbitrary class
raises `pickle.UnpicklingError` instead of executing it — see
`tests/test_storage.py::test_file_checkpoint_store_refuses_malicious_pickle_payload`
for a proof-of-concept that's actually run in CI.

S3-compatible backend (`pip install "pykel[s3]"`):

```python
from kel.storage.s3 import S3BlobStore
blobs = S3BlobStore("my-bucket", region_name="us-east-1")
```

---

## 14. SDK & CLI (`kel.sdk`)

Build an agent straight from a `.md` spec:

```python
from kel.sdk import build_agent_from_spec
agent = build_agent_from_spec("agents/weather.md")   # reads api_key_ref from env automatically
```

Serve it over HTTP (stdlib-only, no extra dependency — local/demo use, not a production deploy story):

```python
from kel.sdk import serve
with serve(agent, port=8000) as server:
    ...   # POST {"input": "..."} to http://127.0.0.1:8000/invoke -> {"text": ..., "stop_reason": ...}
```

Or stream its response over a WebSocket (`pip install "pykel[websockets]"`):

```python
from kel.sdk import serve_websocket
with serve_websocket(agent, port=8000) as server:
    ...  # connect to ws://127.0.0.1:8000/, send {"input": "..."}, receive
         # {"type": "text_delta", ...} / {"type": "tool_result", ...} /
         # {"type": "message_stop", "text": ..., "stop_reason": ...}
         # as the agent's run_stream() events arrive
```

Or mount it as real ASGI routes on FastAPI (`pip install "pykel[fastapi]"`)
— the same idea as LangChain's LangServe, but two routes you can add to
any app you already have:

```python
from kel.sdk import create_fastapi_app, add_agent_routes

app = create_fastapi_app(agent)          # a fresh app, ready for `uvicorn app:app`
# or, onto an app you already have (your own auth/middleware/other routes):
# add_agent_routes(your_app, agent, prefix="/agent")

# POST /invoke {"input": "..."} -> {"text": ..., "stop_reason": ...}
# POST /stream {"input": "..."} -> Server-Sent Events, one per run_stream() event
```

Uses `Agent.arun()`/`arun_stream()` (the async methods) under the hood,
so a slow model call doesn't block the whole ASGI event loop.

**Multi-user production serving: one `Agent` per session, not one shared
`Agent`.** Passing a single `Agent` instance (as above) means every
caller reads/writes the same conversation history — fine for a demo,
wrong for real traffic. Pass a zero-arg factory instead, and both the
FastAPI and WebSocket adapters give each session/connection its own
isolated `Agent`:

```python
from kel.agents import Agent
from kel.sdk import create_fastapi_app
from kel import get_model

def make_agent() -> Agent:
    return Agent("assistant", get_model("anthropic:claude-sonnet-5"))

app = create_fastapi_app(make_agent, session_ttl_seconds=3600)
# POST /invoke {"input": "hi", "session_id": "user-42"}
#   -> looks up (or lazily builds) a dedicated Agent for "user-42"
# a different session_id gets a different Agent/conversation;
# omitting session_id reuses one shared "default" session
# sessions idle past session_ttl_seconds (default 1h) are evicted
```

```python
from kel.sdk import serve_websocket

with serve_websocket(make_agent, port=8000) as server:
    ...   # each new WebSocket connection builds a fresh Agent from the
          # factory — a connection is already a natural session boundary,
          # so there's no session_id to pass
```

`Agent` already serializes concurrent calls on itself (see the
concurrency note in §8), so a single shared `Agent` is safe from
corruption either way — the factory is about isolating *conversations*
between users, not about thread safety.

CLI:

```bash
kel run agents/weather.md "what's the weather in Paris?"
kel eval agents/weather.md agents/weather.eval.md
kel trace tests/cassettes/basic.json
kel --version    # or bare `kel` for the banner + help
```

`python -m kel.sdk.cli ...` also works as an alternative to the installed
`kel` console-script. It prints a harmless `RuntimeWarning` about the
`kel.sdk.cli` module being imported twice (an artifact of `kel.sdk`
eagerly re-exporting `cli` for the `from kel.sdk import main` convenience
import, combined with how `python -m` executes a module) — the command
itself still runs correctly.

---

## 15. Realtime — voice/video orchestration only (`kel.realtime`)

No bundled STT/TTS/lipsync implementation — wire a real vendor SDK behind
`STTProvider`/`TTSProvider`. What's actually implemented is the dual-path
orchestration pattern:

```python
from kel.realtime import run_dual_path

def fast():
    return "let me check that"          # near-instant, e.g. from working memory

def slow():
    return agent.run(user_utterance).text   # the real (slower) answer

final_text = run_dual_path(fast, slow, on_filler=lambda text: tts.speak(text))
```

`STTProvider`/`TTSProvider` are `Protocol`s — any object with a matching
method satisfies them, no base class to inherit from. `STTResult` is
what a real `STTProvider.transcribe()` implementation returns:

```python
from kel.realtime import STTResult, STTProvider, TTSProvider

class MySTT:                                    # satisfies STTProvider structurally
    def transcribe(self, audio: bytes) -> STTResult:
        text = my_vendor_sdk.transcribe(audio)
        return STTResult(text=text, is_final=True)

class MyTTS:                                     # satisfies TTSProvider structurally
    def synthesize(self, text: str) -> bytes:
        return my_vendor_sdk.synthesize(text)
```

---

## 16. Built-in tools (`kel.tools`)

kel's built-in tool library — each function returns a
`kel.agents.Tool`, ready to hand straight to `Agent(..., tools=[...])`.

**Web search — pick whichever provider you have a key for:**

Two no-key options (`wikipedia`, `duckduckgo` — the latter documented
broken, see below) and five official, ToS-compliant API-key options
(`tavily`, `brave`, `serpapi`, `bing`, `google`). All are real search APIs,
not scrapes. Select one dynamically — same `"provider"` pattern as
`kel.get_model`:

```python
from kel.tools import get_web_search_tool

# no key needed
tools = [get_web_search_tool("wikipedia", max_results=3)]

# pick whichever key your user/deployment actually has
tools = [get_web_search_tool("tavily", api_key="tvly-...", max_results=5)]
tools = [get_web_search_tool("brave", api_key="...", max_results=5)]
tools = [get_web_search_tool("serpapi", api_key="...", max_results=5)]
tools = [get_web_search_tool("bing", api_key="...", max_results=5)]
tools = [get_web_search_tool("google", api_key="...", cx="...", max_results=5)]  # needs a Search Engine ID too
```

Or call each factory directly if you're not selecting dynamically:
`wikipedia_search_tool`, `duckduckgo_search_tool`, `tavily_search_tool`,
`brave_search_tool`, `serpapi_search_tool`, `bing_search_tool`,
`google_custom_search_tool` — all importable from `kel.tools`.

A common pattern: let the deployment's environment decide which provider
runs, without changing any agent code —

```python
import os
from kel.tools import get_web_search_tool

def build_search_tool():
    if os.environ.get("TAVILY_API_KEY"):
        return get_web_search_tool("tavily", api_key=os.environ["TAVILY_API_KEY"])
    if os.environ.get("BRAVE_API_KEY"):
        return get_web_search_tool("brave", api_key=os.environ["BRAVE_API_KEY"])
    if os.environ.get("SERPAPI_API_KEY"):
        return get_web_search_tool("serpapi", api_key=os.environ["SERPAPI_API_KEY"])
    return get_web_search_tool("wikipedia")  # always-available fallback, no key required
```

Add your own provider without touching kel's source — the registry is
open, same as `kel.models.register_provider`:

```python
from kel.tools import register_search_provider
register_search_provider("my-engine", my_tool_factory)
get_web_search_tool("my-engine", api_key="...")
```

**`duckduckgo_search_tool`: zero dependency, no API key — but DOCUMENTED
BROKEN** as of 2026-07 testing. DDG now serves an anti-bot CAPTCHA page to
non-browser requests instead of results, so this reliably returns "no
results found". Kept for reference / in case DDG's bot policy changes
again; don't rely on it.

We also tried `googlesearch-python` (blocked — Google serves an "upgrade
your browser" page to the same kind of request) and `serp-scraper`'s real
stealth-browser mode (`CaptchaError: CAPTCHA detected on Google`, even with
a full SeleniumBase UC Mode Chrome instance). Unauthenticated scraping of
major search engines does not work reliably right now, and kel does not
ship code that tries to defeat CAPTCHA/anti-bot protections — use one of
the official API-key providers above instead.

**Reading a specific page — dynamic, works on any site's HTML:**

```python
from kel.tools import fetch_url_tool

fetch = fetch_url_tool(max_chars=3000)
```

`fetch_url_tool` extracts readable text via a generic tag-based parser
(strips `<script>`/`<style>`/`<nav>`/`<header>`/`<footer>`, keeps the rest)
— it isn't hardcoded to any one site's CSS classes the way a search-result
scraper has to be, so it works across arbitrary pages. This is a normal
HTTP client fetching a page it was given, not an anti-bot workaround.

**The pattern that actually fixes shallow-snippet accuracy problems:**
search to find a page, then fetch to read it properly. Verified live:
`wikipedia_search`'s snippet-only result gave a wrong answer for "who won
the most recent Super Bowl" (misread an out-of-context snippet from an
unrelated sub-article); adding `fetch_url` to read the real source article
gave the correct, verified answer.

```python
from kel.agents import Agent
from kel.tools import wikipedia_search_tool, fetch_url_tool

agent = Agent(
    "research-agent",
    model,
    system_prompt=(
        "Use wikipedia_search to find relevant pages, and fetch_url to read "
        "the full article when the search snippet alone isn't enough to answer confidently."
    ),
    tools=[wikipedia_search_tool(max_results=3), fetch_url_tool(max_chars=3000)],
)
```

**MCP (Model Context Protocol) servers** (`pip install "pykel[mcp]"`) —
connect once, get every tool the server advertises as a `kel.agents.Tool`,
instead of hand-writing a custom integration adapter per server:

```python
from kel.tools import mcp_tools_from_server
from mcp import StdioServerParameters

tools = mcp_tools_from_server(StdioServerParameters(command="npx", args=["-y", "@some/mcp-server"]))
agent = Agent("mcp-agent", model, tools=tools)
```

Use `MCPToolset` directly (instead of the one-shot `mcp_tools_from_server`)
if you need to `.close()` the connection deterministically rather than
holding it open for the process's lifetime.

---

## 17. Feature-parity additions

Added to close specific gaps against other LLM orchestration frameworks — each is real and tested, not a stub:

```python
# Async — agenerate()/astream() are real for Anthropic/OpenAI/Cohere now
response = await model.agenerate([Message.user("hi")])
async for event in model.astream([Message.user("hi")]): ...

# Real embeddings (kel.retrieval used to only have a local-dev hash embedder)
from kel import get_embedding_model
from kel.retrieval import embedder_from_model
embedding_model = get_embedding_model("openai:text-embedding-3-small")
retriever = Retriever(store, embedder=embedder_from_model(embedding_model))

# Response caching + rate limiting — composed via get_model()
from kel.caching import InMemoryCache, SQLiteCache
from kel.ratelimit import RateLimiter
model = get_model("openai:gpt-5.2", cache=InMemoryCache(), rate_limit={"requests_per_minute": 60})

# Or share one RateLimiter's token bucket across multiple models directly
# (RateLimitedChatModel is what get_model(rate_limit=...) wraps internally)
from kel.ratelimit import RateLimitedChatModel
shared_limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=100_000)
model_a = RateLimitedChatModel(get_model("openai:gpt-5.2", instrument=False), shared_limiter)
model_b = RateLimitedChatModel(get_model("anthropic:claude-sonnet-5", instrument=False), shared_limiter)

# Structured output (with_structured_output equivalent)
from kel import generate_structured
from pydantic import BaseModel
class Answer(BaseModel):
    value: int
result = generate_structured(model, [Message.user("what is 6*7")], Answer)

# Recursive (structure-aware) splitter + Qdrant + PDF loader
from kel.retrieval import recursive_split_text, load_pdf
from kel.retrieval.qdrant import QdrantVectorStore  # pip install pykel[qdrant]
retriever = Retriever(QdrantVectorStore("my-collection"), embedder=..., splitter=recursive_split_text)
retriever.ingest(load_pdf("handbook.pdf"))  # pip install pykel[pdf]

# Streaming at the agent-loop level (not just one model call)
for event in agent.run_stream("what's 2+2 and search wikipedia for Python"):
    ...  # TextDelta / ToolCallDelta / ToolResultEvent / MessageStop
async for event in agent.arun_stream("..."): ...

# Human-in-the-loop interrupt/resume
from kel.runtime import Interrupt, resume_graph
def approve_node(state):
    if "__resume_value__" not in state:
        raise Interrupt({"action": "approve_payment", "amount": 100})
    return {"approved": state["__resume_value__"]}
paused = run_graph(graph, {})            # paused.interrupted is True
resumed = resume_graph(graph, paused, resume_value=True)

# LLM-graded evaluation (criteria evaluator, not just substring match)
from kel.specs import EvalCase, run_llm_graded_eval_suite
case = EvalCase(name="c1", input="what is 2+2", criteria="the answer is mathematically correct")
results = run_llm_graded_eval_suite(judge_model, [case], respond=lambda text: agent.run(text).text)

# Code execution tool (subprocess-isolated, NOT a security sandbox — see docstring)
from kel.tools import python_exec_tool
tools = [python_exec_tool(timeout=5.0)]

# Prompting techniques: few-shot, chain-of-thought, classic text-based ReAct
from kel.prompting import format_few_shot_prompt, chain_of_thought_prompt, build_react_system_prompt, extract_final_answer

cot_prompt = chain_of_thought_prompt("what is 17% of 240?")   # appends DEFAULT_COT_SUFFIX asking for step-by-step + "Final Answer:"
response_text = model.generate([Message.user(cot_prompt)]).text
answer = extract_final_answer(response_text)   # text after the last "Final Answer:" marker, or the whole text if the model skipped it
```

---

## 18. Lightweight live monitoring dashboard (`kel.monitoring`)

A zero-new-dependency dashboard — stdlib `http.server` only, same pattern
as `kel.sdk.serve`. `MetricsSink` is just another `kel.observability`
sink, so it plugs into the real tracer with `add_sink()`, no separate
wiring:

```python
from kel.observability import add_sink
from kel.monitoring import MetricsSink, start_dashboard

metrics = MetricsSink()
add_sink(metrics)  # now every kel span (model calls, cache hits, runtime nodes...) feeds it

dashboard = start_dashboard(metrics, port=8090)
print(f"open {dashboard.url}")   # http://127.0.0.1:8090/ — auto-refreshes every 1.5s, no restart needed
```

Tracks, per span name and overall: call count, error count/rate, avg/p95/p99
latency, summed input/output tokens, cache hit rate — plus a live scrolling
recent-activity log. `metrics.snapshot()` returns the same data as plain
JSON if you want to feed it elsewhere instead of using the HTML page.

This is explicitly a local/dev view, not a production metrics backend —
in-memory, per-process, resets on restart. For durable/production
dashboards, export to Grafana via `pykel[otel]` instead (§2).

---

## 19. Provider / vector DB / tool breadth

Not chasing "100+ integrations" (see DESIGN.md §7), but covering the real,
commonly-requested top picks in each category:

**5 chat model providers**, all real adapters (not stubs), each with
optional credentials that fall back to environment/ambient resolution:

```python
from kel import get_model
get_model("anthropic:claude-sonnet-5")
get_model("openai:gpt-5.2")
get_model("cohere:command-a-03-2025")
get_model("gemini:gemini-2.5-flash")     # pip install pykel[gemini] — falls back to GEMINI_API_KEY/ADC
get_model("mistral:mistral-large-latest") # pip install pykel[mistral] — falls back to MISTRAL_API_KEY
```

**5 vector store adapters** (plus the in-memory one for local dev):

```python
from kel.retrieval import InMemoryVectorStore
from kel.retrieval.qdrant import QdrantVectorStore              # pip install pykel[qdrant]
from kel.retrieval.pinecone_store import PineconeVectorStore     # pip install pykel[pinecone]
from kel.retrieval.weaviate_store import WeaviateVectorStore     # pip install pykel[weaviate] — no key needed for self-hosted
from kel.retrieval.chroma_store import ChromaVectorStore         # pip install pykel[chroma] — local mode needs nothing at all
from kel.retrieval.pgvector_store import PgVectorStore           # pip install pykel[pgvector] — password optional
```

**5 built-in tools**: `web_search` (7 providers — see §16), `fetch_url`,
`python_exec`, `sql_query`, `shell_exec` — all from `kel.tools`.

**Credentials are optional everywhere it matters**, verified by a
contract test (`tests/test_credentials_optional.py`) that checks every
adapter's constructor signature directly — not just documented, enforced:

- **S3** (`kel.storage.s3.S3BlobStore`): no key/secret param at all — resolves via boto3's default chain, which covers an EC2 instance profile role, an EKS pod's IRSA-injected role, or an ECS task role automatically.
- **Gemini**: falls back to `GEMINI_API_KEY`/`GOOGLE_API_KEY` env vars or, on GCP (GKE Workload Identity, GCE, Cloud Run), Application Default Credentials with no key at all.
- **Mistral**: falls back to `MISTRAL_API_KEY`.
- **Weaviate**: no key needed for self-hosted/in-cluster deployments (the default when `url` isn't given) — only Weaviate Cloud needs one.
- **Chroma**: local/embedded modes (the default) need nothing; only a remote HTTP server with auth enabled needs a token.
- **pgvector**: `password` is optional — falls back to `PGPASSWORD`/`.pgpass`, or accepts a pre-generated IAM auth token (e.g. AWS RDS IAM auth) passed as a plain string.
- **Pinecone**: the one exception — it's SaaS-only with no ambient-credential equivalent to IAM roles, so it always needs a key, but that key can come from `PINECONE_API_KEY` rather than being hardcoded.

---

## Known gaps (see DESIGN.md §7 for the reasoning)

- `kel.testing` replay covers `generate()`, not `stream()`.
- No `kel init <template>` project scaffolding.
- No local trace-viewer UI — `kel trace` (CLI) and Grafana (via `pykel[otel]`) are the two ways to look at a run today.
- `python_exec_tool`/`shell_exec_tool` are process-isolated with a timeout, not a real security sandbox — don't run untrusted/adversarial code through either.
- Cache keys (`kel.caching`) cover `generate()`'s named parameters only, not arbitrary provider-specific `**kwargs`.
- Rate limiting reserves an *estimated* token cost up front and doesn't refund the difference against actual usage — conservative, not exact.
- Pinecone has no built-in keyword/full-text search on the base vector index — `PineconeVectorStore.keyword_query` returns an empty list.
- Nothing here has been exercised against a live API key or a real OTLP/vector-DB/database server except the Cohere path used in the separate `cohere-agent` demo project — everything else (including all 5 vector store adapters, Gemini, Mistral) is tested against fakes/mocks in the test suite.



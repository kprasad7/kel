"""A single agent with a tool + RAG retrieval — the classic "agent with
tools" quickstart, using kel's actual subsystems end to end: model
gateway, budget, observability, retrieval, and the agent tool loop.

Run:
    export ANTHROPIC_API_KEY=sk-...
    python examples/01_tool_and_rag_agent.py
"""

from kel import get_model
from kel.agents import Agent, Tool
from kel.budget import Budget
from kel.retrieval import InMemoryVectorStore, NaiveHashEmbedder, Retriever

# ---------------------------------------------------------------------------
# 1. Retrieval: ingest a couple of "documents" for the agent to search over.
#    Swap NaiveHashEmbedder for a real embedding model in production.
# ---------------------------------------------------------------------------
retriever = Retriever(InMemoryVectorStore(), embedder=NaiveHashEmbedder(dims=128))
retriever.ingest(
    "kel's refund policy: customers can request a full refund within 30 days "
    "of purchase. Refunds are processed within 3-5 business days.",
    id_prefix="refund-policy",
)
retriever.ingest(
    "kel's shipping policy: standard shipping takes 5-7 business days. "
    "Express shipping is available for an additional fee and takes 1-2 days.",
    id_prefix="shipping-policy",
)


def search_docs(tool_input: dict) -> str:
    results = retriever.retrieve_hybrid(tool_input["query"], k=2)
    if not results:
        return "no matching documents found"
    return "\n\n".join(f"(score={r.score:.2f}) {r.chunk.text}" for r in results)


def calculate(tool_input: dict) -> str:
    # a real tool would validate input carefully; kept trivial for the example
    return str(eval(tool_input["expression"], {"__builtins__": {}}))


tools = [
    Tool(
        name="search_docs",
        description="Search kel's policy documents for relevant information.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        fn=search_docs,
    ),
    Tool(
        name="calculate",
        description="Evaluate a simple arithmetic expression.",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        fn=calculate,
    ),
]

# ---------------------------------------------------------------------------
# 2. Model gateway + budget: get_model() wraps the model with tracing
#    (console output by default — see kel.observability to route to Grafana)
#    and enforces the budget on every call.
# ---------------------------------------------------------------------------
model = get_model(
    "anthropic:claude-sonnet-5",
    budget=Budget(max_tokens=20_000, max_cost_usd=0.50, max_tool_calls=5),
)

# ---------------------------------------------------------------------------
# 3. Agent: a bounded tool-calling loop (kel.context.Loop under the hood —
#    stuck-loop detection and a max-iteration cap are automatic).
# ---------------------------------------------------------------------------
agent = Agent(
    "support-agent",
    model,
    system_prompt=(
        "You are a customer support agent for kel. Use search_docs to answer "
        "policy questions, and calculate for any arithmetic. Cite what you found."
    ),
    tools=tools,
)

if __name__ == "__main__":
    response = agent.run("If I paid $120 and want a refund, how much do I get back and by when?")
    print("\n--- final answer ---")
    print(response.text)

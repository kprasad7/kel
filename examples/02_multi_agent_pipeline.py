"""Multi-agent orchestration — three agents, each possibly on a different
model/provider/API key, sharing context through a plain state dict instead
of only seeing each other's last message (DESIGN.md 3.11).

Run:
    export ANTHROPIC_API_KEY=sk-...
    export OPENAI_API_KEY=sk-...
    python examples/02_multi_agent_pipeline.py
"""

from kel import get_model
from kel.agents import Agent, sequential_pipeline
from kel.runtime import run_graph

# Each agent can use its own model/provider/key — get_model() is the only
# thing that changes per agent; the orchestration code doesn't care.
researcher = Agent(
    "researcher",
    get_model("anthropic:claude-sonnet-5"),
    system_prompt="You research a topic and list 3 concise bullet-point facts. No fluff.",
)

writer = Agent(
    "writer",
    get_model("openai:gpt-5.2"),  # a different provider, on purpose
    system_prompt=(
        "You write a short, engaging paragraph based on the facts you're given. "
        "The facts will appear above your task, prefixed with '[researcher_output]'."
    ),
)

editor = Agent(
    "editor",
    get_model("anthropic:claude-haiku-4-5"),  # cheap/fast model for a lightweight pass
    system_prompt="You tighten the given paragraph to 2 sentences, keeping the key facts.",
)

# sequential_pipeline builds a kel.runtime.Graph where each agent's output
# is written into shared state as f"{agent.name}_output" — the writer sees
# what the researcher decided, the editor sees what the writer produced.
graph = sequential_pipeline([researcher, writer, editor])

if __name__ == "__main__":
    result = run_graph(graph, {"input": "the benefits of async I/O in Python"})

    print("--- researcher ---")
    print(result.state["researcher_output"])
    print("\n--- writer ---")
    print(result.state["writer_output"])
    print("\n--- editor (final) ---")
    print(result.state["editor_output"])
    print("\nnode execution order:", result.history)

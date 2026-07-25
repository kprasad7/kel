# 🧠 kel — The Open-Source Agentic AI Framework for Python

**kel** is a universal, production-grade **agentic AI framework for Python** — a normalized multi-provider model gateway plus observability, budget control, memory, retrieval (RAG), multi-agent orchestration, self-healing, and testing, built from the ground up to fix the production pain points that **LLM orchestration frameworks** commonly leave unsolved: opaque execution, no native cost control, no deterministic testing, and context/loop failures.

If you're evaluating **AI agent frameworks**, **production-ready agent orchestration libraries**, or a **Python framework for building autonomous AI agents** with real observability and cost governance — this is built for exactly that.

<p align="center">
  <img alt="build" src="https://img.shields.io/github/actions/workflow/status/kprasad7/kel/ci.yml?branch=develop&label=build&style=for-the-badge&logo=github&color=2ea44f">
  <img alt="security" src="https://img.shields.io/github/actions/workflow/status/kprasad7/kel/security.yml?branch=develop&label=security%20scan&style=for-the-badge&logo=trivy&color=blueviolet">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-yellow.svg?style=for-the-badge">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-ff69b4.svg?style=for-the-badge">
</p>

📖 [**DESIGN.md**](DESIGN.md) — full architecture, scope, and the reasoning behind every tradeoff
📘 [**USAGE.md**](USAGE.md) — a working guide to every subsystem, with real code examples

---

## ✨ Why kel

| 🎯 Common pain point in LLM orchestration frameworks | ✅ How kel solves it |
|---|---|
| 🕳️ No first-class tracing — bolt on a third-party observability product or go without | 📊 **Every call traced by default**, self-hostable via Grafana/OTel |
| 💸 Hidden token/cost consumption, surprise bills | 💰 **Budget objects threaded through every call** — hard caps, not suggestions |
| 🧵 Deeply nested abstractions, painful debugging | 🪶 **Flat, composable classes** — no multi-layer wrapper hierarchies to step through |
| 🔁 Agents loop forever, no stuck-loop detection | 🛑 **Built-in stuck-loop + step-budget guardrails** |
| 🧪 No deterministic testing story | 🎬 **Record/replay testing** — real API calls once, deterministic CI forever |
| 🤖 Multi-agent state gets lost between agents | 🔗 **Shared context bus** — downstream agents see what upstream agents *decided* |
| 🔓 Real disclosed CVEs (deserialization, path traversal, SQLi) | 🛡️ **Hardened by design** — restricted unpickling, parameterized queries, Trivy-scanned every push |
| 🔑 Forces credentials everywhere | ☁️ **Credentials optional by design** — works natively on EC2/EKS/IAM roles, IRSA, self-hosted anonymous access |

---

## 🚀 Feature Highlights

- 🔌 **Model Gateway** — one interface across **Anthropic, OpenAI, Cohere, Gemini, Mistral**, sync + real async
- 📈 **Observability** — every span traced automatically → console, Grafana, or the built-in live dashboard
- 💵 **Budget & Rate Limiting** — token/cost/tool-call caps + RPM/TPM throttling, composable
- ⚡ **Caching** — in-memory or SQLite response caching, never double-charges budget on a hit
- 🧠 **Memory** — working / episodic / semantic / procedural, layered like a real cognitive architecture
- 📚 **Retrieval (RAG)** — Qdrant, Pinecone, Weaviate, Chroma, pgvector, hybrid search, recursive splitting, PDF loading
- 🤝 **Multi-Agent Orchestration** — sequential, supervisor, parallel, and swarm patterns, streaming included
- 🧭 **Brain** — fast rule/embedding routing with LLM fallback, parallel-to-finish scheduling
- 🩹 **Self-Healing** — diagnosis-driven retries with a non-negotiable idempotency guardrail
- 🧰 **Built-in Tools** — web search (7 providers), URL fetch, Python/shell exec, SQL query
- ✅ **Testing** — record/replay + LLM-graded evaluation, no live API key needed in CI
- 🖥️ **Live Monitoring Dashboard** — zero-dependency, real-time metrics + logs in your browser
- 🔐 **DevSecOps Built In** — Trivy vulnerability/secret scanning, `pip-audit`, and Bandit SAST on every push

---

## 📦 Install

```bash
pip install -e ".[dev]"        # core + test tooling
pip install -e ".[anthropic]"  # + Anthropic
pip install -e ".[openai]"     # + OpenAI
pip install -e ".[gemini]"     # + Google Gemini
pip install -e ".[mistral]"    # + Mistral
pip install -e ".[qdrant]"     # + Qdrant vector store
pip install -e ".[all]"        # everything
```

## ⚡ Quickstart

```python
from kel import get_model, Message

model = get_model("anthropic:claude-sonnet-5", api_key="...")
response = model.generate([Message.user("Hello!")])
print(response.text)
```

Swapping providers is a one-line change — same interface, zero rewrites:

```python
model = get_model("openai:gpt-5.2", api_key="...")
model = get_model("gemini:gemini-2.5-flash")      # falls back to env vars / ADC
model = get_model("mistral:mistral-large-latest")  # falls back to MISTRAL_API_KEY
```

---

## 🩺 Status

Every subsystem in [DESIGN.md](DESIGN.md) has a working, tested implementation — model gateway, observability, budget, context/loop, memory, retrieval, specs, runtime graph, multi-agent orchestration, brain, self-healing, testing, storage, SDK/CLI, monitoring dashboard, and realtime orchestration (interfaces only, by design — see DESIGN.md §7). Known gaps are listed honestly at the bottom of [USAGE.md](USAGE.md) — no overclaiming.

## 🛡️ Security

Every push and pull request runs a full DevSecOps pipeline: [Trivy](https://github.com/aquasecurity/trivy) filesystem + secret scanning, `pip-audit` for known CVEs in dependencies, and Bandit static analysis over the codebase. Results surface in the repo's Security tab. See [`.github/workflows/security.yml`](.github/workflows/security.yml).

## 🤝 Contributing

Issues and PRs welcome. Adding a new model provider, vector store, or tool follows the same lazy-import adapter pattern throughout the codebase — see any file under `src/kel/models/providers/` for the template.

## 📄 License

[MIT](LICENSE)

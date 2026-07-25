# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/kprasad7/kel/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/kprasad7/kel/releases/tag/v1.0.1
[1.0.0]: https://github.com/kprasad7/kel/releases/tag/v1.0.0
[0.1.0]: https://github.com/kprasad7/kel/releases/tag/v0.1.0

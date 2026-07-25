# Contributing to kel

Thanks for considering a contribution. This project is young, so small,
focused pull requests are the easiest to review and merge.

## Getting set up

```bash
git clone https://github.com/kprasad7/kel.git
cd kel
python -m venv .venv
. .venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -e ".[dev]"
```

## Before opening a PR

Run the same checks CI runs:

```bash
ruff check src tests
mypy src
pytest -q
```

All three must pass. If you touch a provider or vector-store adapter that
requires a third-party SDK, install it via the relevant extra (e.g.
`pip install -e ".[qdrant]"`) and add tests against an injected fake
client — tests should never require live credentials or a live account.

## Guidelines

- **Credentials stay optional.** Every provider/store adapter must work
  when no explicit API key is passed, falling back to environment
  variables or ambient cloud credentials (IAM role, Workload Identity,
  IRSA). Don't add a required-key check that breaks that path.
- **No new abstractions for their own sake.** kel's whole pitch is fewer
  layers than the alternatives — prefer a direct implementation over a new
  interface unless at least two call sites need it.
- **Security-sensitive changes** (deserialization, subprocess/shell
  execution, URL fetching, SQL construction) need a one-line comment
  explaining why the operation is safe, not just that it works.
- **Docstrings over comments** for adapters: explain what real-world
  deployment scenario a design choice serves (e.g. "works under EKS IRSA
  with no key at all"), not what the code does line by line.

## Reporting bugs / requesting features

Open a [GitHub issue](https://github.com/kprasad7/kel/issues). For
security vulnerabilities, see [SECURITY.md](SECURITY.md) instead of
filing a public issue.

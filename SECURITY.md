# Security Policy

## Supported Versions

kel is pre-1.0 and moving quickly. Security fixes are only guaranteed
against the latest release on the default branch.

| Version | Supported |
| ------- | --------- |
| latest  | ✅        |
| < latest | ❌       |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report privately using one of these channels:

1. [GitHub Security Advisories](https://github.com/kprasad7/kel/security/advisories/new) (preferred)
2. Direct message to the repository owner ([@kprasad7](https://github.com/kprasad7))

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal repro is very helpful)
- Any suggested fix, if you have one

We aim to acknowledge reports within 5 business days and to ship a fix or
mitigation as soon as reasonably possible depending on severity.

## Scope and Design Notes

kel is designed to run with **optional credentials** (env vars, ambient
cloud IAM roles) — this is intentional and not itself a vulnerability.
Areas we treat as security-sensitive and actively harden:

- Deserialization (`kel.storage.checkpoint_store` uses a restricted
  unpickler, not raw `pickle.loads`)
- Outbound URL fetching (`kel.tools.web_fetch` allowlists `http`/`https`
  schemes to prevent `file://`/SSRF-style access)
- SQL construction in vector-store adapters (parameterized where the
  driver supports it; identifiers that must be interpolated are
  developer-controlled, not user input)
- Subprocess/shell execution tools

Every push and PR to `main`/`master`, plus a weekly schedule, runs Trivy,
pip-audit, and Bandit via `.github/workflows/security.yml`.

# Security Policy

## Supported versions

Prot2Vec is a research tool under active development. Fixes land on the latest
release; there are no long-term support branches.

## Reporting a vulnerability

Please report security issues privately rather than in a public issue — use
GitHub's [private vulnerability
reporting](https://github.com/nikhil-kunapareddy/Prot2Vec/security/advisories/new)
on this repository. You can expect an acknowledgement within a few days.

## Scope notes

Two things are worth knowing about how Prot2Vec handles untrusted input:

- **Config files are executable in effect.** An experiment YAML selects which
  code paths run and where files are written. Only run configs you trust, the
  same way you would treat a Makefile. Configs are parsed with
  `yaml.safe_load`, so they cannot construct arbitrary Python objects.
- **API keys are read only from the environment**, via `.env` at the repository
  root or your shell. They are never read from a config file, never logged, and
  never written to a run manifest. `.env` is gitignored; keep it that way.

The `llm` embedder transmits your sequences to a third-party embedding API.
If your sequences are unpublished or otherwise sensitive, that is a data
disclosure — review the provider's data-retention terms before enabling it.

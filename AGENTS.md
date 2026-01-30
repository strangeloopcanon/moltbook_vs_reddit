---
name: moltbook-analysis-agent
description: Project-local instructions for working in this repo.
mode: baseline
---

# moltbook_analysis – agent notes

## Interface contract
- Setup: `make setup`
- Gates: `make all` (stop at first failure)

## Boundaries
- Don’t commit `.env` or the raw corpora / generated reports under `data/` (they’re gitignored).
- Prefer minimal, correct changes over refactors.

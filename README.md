# NexaQuery Core Orchestrator

Multi-agent decision copilot that turns natural-language questions into SQL against a PostgreSQL warehouse, then validates result sets against business rules before they reach the UI.

Built as a FastAPI backend with an orchestrator that chains specialized agents, declarative rule evaluation, and an optional LLM assist — with explicit latency budgets and fail-open behavior so validation never takes down the main request.

## What it demonstrates

- **Multi-agent orchestration** — `Orchestrator` runs Query → Validator; validators register via a dispatcher so new checks plug in without rewriting the pipeline.
- **NL → SQL query path** — `QueryAgent` is the slot for LLM-backed SQL generation and warehouse execution (stubbed in this repo so the validation layer can be developed and tested independently).
- **Declarative business-rule validation** — JSON rules evaluated with pandas (`eval`), including safe coercion of legacy string/date columns and skip-on-missing-columns behavior.
- **Optional LLM validation** — Anthropic pass for domain issues that are hard to express as JSON; bounded by `asyncio.wait_for` (~2s), skipped when rules already fail or the API key is missing; timeouts/errors log and continue (fail-open).
- **Hard-rules AI extension** — separate module for cross-column / “smell test” checks without bloating the core validator.
- **Production-minded defaults** — structured validation payload (`validation`, `validation_ok`), logging, env-driven config, pytest coverage for the validator path.

## Pipeline

```
User query
    → QueryAgent (NL → SQL → DataFrame)
    → Dispatcher.run_validators (business rules ± LLM)
    → Response { data, metadata, validation, validation_ok }
```

Rules live in `docs/rules-draft.json` (e.g. net sales ≥ 0, discount ≤ 100%, no future transaction dates). Path and LLM timeout are configurable via `.env`.

## Stack

Python 3.11 · FastAPI · pandas · Anthropic · SQLAlchemy / Alembic · PostgreSQL · pytest

## Project layout

- `app/agents/` — Orchestrator, QueryAgent, business-rules validator, hard-rules AI helper
- `app/utils/dispatcher.py` — Validator registry and fan-out
- `app/services/llm.py` — Anthropic client wrapper
- `docs/` — Specs, issue brief, draft rules

## Quick setup

```bash
python -m venv venv && source venv/bin/activate
cp .env.example .env
pip install -r requirements.txt
fastapi dev app/main.py
```

```bash
pytest
```

Point `DATABASE_URL` and `ANTHROPIC_API_KEY` in `.env` when exercising the real warehouse / LLM paths. Validator behavior is controlled by `BUSINESS_RULES_PATH`, `VALIDATOR_LLM_TIMEOUT_SEC`, and `VALIDATOR_MAX_SAMPLE_ROWS`.

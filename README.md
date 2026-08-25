# NexaQuery Core Orchestrator

Welcome to the NexaQuery Labs team. This is the core engine of our decision copilot. The service turns natural language into precise SQL queries and returns actionable results.

## Project Structure

- `app/agents/`: Specialized agents (Query, Validator, etc.).
- `app/utils/dispatcher.py`: Tool registry and dispatch for the multi-agent system.
- `app/services/llm.py`: Unified client for language models.
- `docs/`: Technical documentation and architecture specs.

## Quick Setup

1. Clone the repository and create a virtual environment.
2. Copy the environment file:

```bash
cp .env.example .env
```

3. Install dependencies and run the server in development mode:

```bash
pip install -r requirements.txt
fastapi dev app/main.py
```

## Your Team

- **Matías Oyarzún** — Tech Lead & Engineering Manager
- **Diego Méndez** — Senior Backend & AI Engineer
- **Valentina Rojas** — Senior Data Engineer

## Workflow

Review `docs/issues/402-validator-agent.md` for details on your first task. Once you have a solution, open a Pull Request for Diego or Valentina to review.

# AGENTS Instructions

These guidelines ensure consistent contributions to the SAGA autonomous creative writing system.

## Repository Overview
SAGA is a Python project containing multiple modules such as `planner_agent.py`, `drafting_agent.py`, and `world_continuity_agent.py`. These agents collaborate to generate novel content using asynchronous tasks and a Neo4j knowledge graph.

## Commit Messages
- Use Conventional Commits: `<type>(<scope>): <subject>`
- Types include `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Optional scopes: `agents`, `coordinator`, `memory`, `tasks`, `api`, `cli`
- Keep the subject under 50 characters in the imperative mood
- Example: `feat(agents): add emotion tracking`

## Testing Requirements
- Run `ruff check . && ruff format --check .`
- Run `pytest tests/ -v --cov=. --cov-report=term-missing`
- Ensure coverage stays above 85%
- Run `mypy .`
- Skip the above when changing only documentation or comments

## Code Style
- Format with `ruff format` (Black compatible, 88 characters)
- Sort imports via `ruff check --select I --fix`
- Use double quotes for strings
- Provide type hints for all new functions
- Use `async`/`await` for I/O
- Prefer composition over inheritance for new agent features
- Use `structlog` for logging instead of `print`
- Agent names end with `Agent` (e.g., `PlannerAgent`)

## Agent Guidelines
- Agent classes should implement clear entry points such as `process_task()` and `handle_message()`
- Maintain agent state through helper classes or the existing database modules, not bare instance variables
- All agent communication should go through the orchestrator modules

## Dependencies
- Add new packages to `requirements.txt` and pin major versions (e.g., `requests>=2.25,<3`)
- Prefer lightweight libraries when possible (e.g., `httpx` for HTTP)

## PR Expectations
- Provide a concise summary of changes and motivation
- Indicate the change type (bug fix, new feature, breaking change, documentation)
- Include test results in the PR description
- Mention any breaking changes or migrations

## Documentation Standards
- Use Google‑style docstrings for public functions and classes
- Document expected message formats and agent capabilities
- Update `README.md` when adding new agent types or changing setup steps

## Error Handling
- Wrap external calls with `try`/`except` and log exceptions with `logger.exception`
- Fail fast when required environment variables are missing
- Agent failures should not crash the orchestrator

## Configuration Management
- Environment variable names begin with `AGENT_` when possible (e.g., `AGENT_LOG_LEVEL`)
- Document variables in `.env.example`

## Security & Performance
- Never commit API keys
- Validate message payloads between agents
- Limit concurrent tasks per agent (default 10) and monitor memory usage
- Use MAX_CONCURRENT_CHAPTERS setting to control chapter processing concurrency


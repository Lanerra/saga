# SAGA Project Constraints

## What This Project Is
A single-process Python CLI application for single-user autonomous AI novel generation.

## Hard Constraints
- Single user, single machine only
- Neo4j for graph storage; files and local SQLite for LangGraph checkpoints
- No SAGA web server, public API, or distributed service architecture
- Consumer hardware target
- Local-first architecture

## Explicitly NOT Needed
- Authentication/authorization
- Horizontal scaling
- Microservices
- Message queues
- Load balancers
- Container orchestration

## Neo4j Usage
- A local Neo4j server reached through its Bolt driver, optionally in Docker;
  it is not embedded in the Python process. Used for narrative consistency,
  not web-scale data. One database is exclusively owned by one project.
- Think "personal knowledge base" not "social network backend."

## Local LLM Endpoints (Clarification)
- Local-first explicitly permits locally-running HTTP endpoints for LLMs and
  embeddings on the same machine (e.g., OpenAI-compatible gateways, Ollama,
  vLLM). These are treated as local processes, not remote services.
- Remote/cloud endpoints are not used by default. Users may opt-in by
  configuring environment variables, but the system should function with
  fully local endpoints.

## Agent Architecture  
- Sequential processing pipeline, not concurrent microservices.
- Agents are functions/classes, not separate processes.

Operational instructions: [current guide](../README.md).

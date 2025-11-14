# SAGA Project Constraints

## What This Project Is
A single-process Python CLI application for single-user autonomous AI novel generation.

## Hard Constraints
- Single user, single machine only
- No databases beyond Neo4j/file storage
- No web servers, APIs, or network services
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
- Local embedded instance only. Used for narrative consistency, not web-scale data.
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

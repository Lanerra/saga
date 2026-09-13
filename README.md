# SAGA

SAGA (Semantic And Graph-enhanced Authoring) turns a story premise into an outline,
scene-by-scene drafts, and an exportable Markdown manuscript. It uses a Neo4j
knowledge graph to track characters, events, and relationships, and LangGraph to
checkpoint the writing workflow so you can resume an interrupted project.

SAGA is a local-first Python command-line application for one writer on one machine.
There is no SAGA web server. You supply the text-generation, embedding, and Neo4j
services; SAGA does not install or start them for you. Story text is sent to your
configured model endpoints, so use local endpoints if it must stay on your machine.

## Start here

1. [Install the Python environment and spaCy pipeline](#installation).
2. [Configure your model and database connections](#configuration).
3. [Create a project, review its settings, and generate](#write-your-first-story).
4. [Resume or export](#resume-and-export) using the exact project directory.

Start with a short project before attempting a novel. Model calls can be slow and,
with paid endpoints, expensive. Hardware requirements depend on the model server;
the Python application does not bundle model weights or a GPU runtime.

## What to expect

SAGA saves outlines, character/world files, chapter drafts, summaries, and workflow
checkpoints under `projects/{story_title}/`. Export reads accepted manuscripts rather
than unfinished drafts. Keep the project directory and its Neo4j data together.

The repaired workflow has completed a real-model two-chapter, eight-scene project
of about 6,700 words, including a source-repaired continuation, accepted export,
and a fresh-process reopen without model replay or changes to accepted data.
This was a bounded test, not evidence that every model or full-length novel works.
It used an Ornith-1.5-35B-A3B model through llama.cpp and 1,024-dimensional
`qwen3-embedding:0.6b` embeddings, not the placeholder models in `.env.example`.

**Expect to edit the output.** AI review of that manuscript found repetition,
chronology and arithmetic errors, and incomplete adherence to the premise.
Automatic acceptance is not an editorial endorsement. Prose evaluation samples
the first and last 4,000 characters of longer chapters rather than grading every
line. The default author policy can accept with explicit exceptions; the example
project's exceptions were graph-check cadence skips, not executed graph checks.
See [quality acceptance](docs/quality-acceptance.md) for the policy details.

## Screenshots

Progress window (Rich CLI):

![SAGA Progress Window](SAGA.png)

Historical knowledge-graph example (five chapters, not a current quality benchmark):

![SAGA KG Visualization](SAGA-KG-Ch5.png)

## Installation

### Prerequisites

- Linux x86-64; the verified runtime uses CPython **3.12.13** and `uv` **0.11.21**.
- Neo4j with compatible APOC. The tested combination is Community **5.26.8**, its
  bundled APOC core, and JDK **21**. See [database setup](#database-setup).
- A text endpoint supporting OpenAI-compatible chat completions and JSON-schema
  response formats, with enough context and output capacity for your settings.
- An embedding endpoint supporting Ollama's `POST /api/embeddings` protocol.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then
open a shell in this repository. macOS, Windows, and other Python versions have
not been verified.

### Setup

```bash
uv python install 3.12.13 --no-bin
uv venv --python 3.12.13 --managed-python .venv-runtime
uv pip sync --python .venv-runtime/bin/python --require-hashes --only-binary :all: requirements.lock
.venv-runtime/bin/python -m pip check
source .venv-runtime/bin/activate
cp -n .env.example .env
```

Edit your `.env` to match your local services (see `.env.example`).
The copy command preserves an existing `.env`; never replace credentials on retry.

Use a fresh `.venv-runtime` directory. If one already exists, activate that environment
or choose a different destination rather than overwriting it. Run SAGA from the
repository root so imports and relative project paths resolve to this checkout.

`requirements.txt` is the direct-pin resolver input, not the supported installation
command. `requirements.lock` pins and SHA-256-verifies the full Linux Python 3.12
dependency closure, including pip, pytest/asyncio/coverage/timeout, Ruff, Mypy and
aiofiles/PyYAML type stubs. It does not install SAGA as an editable package or install
GPU libraries. Other platforms and Python versions are not verified.

<details>
<summary>For contributors: checking or updating the dependency lock</summary>

To check or deliberately refresh the lock with the supported interpreter, constrain
resolution to the existing lock so unchanged dependencies do not drift:

```bash
uv pip compile requirements.txt --constraint requirements.lock \
  --python .venv-runtime/bin/python --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-emit-index-url --no-header --no-annotate \
  --output-file requirements.candidate.lock
cmp requirements.lock requirements.candidate.lock
```

An intentional version change requires updating its direct pin and reviewing the
corresponding constraint change before resolving. Inspect the candidate, install it
into a separate fresh environment, then run `python -m pip check` and the runtime
regressions before promoting it. A successful `pip check` alone does not prove imports
work: spaCy 3.8.7 also requires the explicitly pinned Click compatibility dependency.

</details>

### Install the spaCy language pipeline

The default statistical pipeline is `en_core_web_lg`. The library lock does not include
model weights, and CLI help does not require them. Install the compatible **3.8.0**
model explicitly when preparing an authoring environment with network access:

```bash
python -m pip install --no-deps \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
python -m pip check
python -c 'import spacy; pipeline = spacy.load("en_core_web_lg"); assert pipeline.meta["version"] == "3.8.0"; print(pipeline.pipe_names)'
```

`--no-deps` preserves the locked library versions. For air-gapped setup, download
that exact wheel on a connected machine, record and verify its SHA-256 during transfer,
then install the local wheel with `--no-index --no-deps`. Model wheels are a separate
resource, not covered by the library lock's hashes; syncing only `requirements.lock`
removes the optional model package. Do not run model download commands automatically
at application startup or in unit fixtures. `SPACY_MODEL` selects a deliberately
installed alternative; the small model does not provide the large model's vectors.

This spaCy package handles language analysis; it is separate from both the text
generation model and the embedding model served by your endpoints.

## Database setup

Use a dedicated Neo4j database for each story. SAGA claims exclusive project
ownership and will refuse an occupied or incompatible database. Do not point a
first test at a database containing work you want to keep.

For real stories, configure persistent storage, keep authentication enabled, and
test your backup/restore procedure. The examples below are disposable smoke-test
setups, not durable deployments. You can skip them if you already have a suitable
empty database with APOC enabled.

<details>
<summary>Standalone Neo4j smoke-test setup (verified engine combination)</summary>

Docker is not required. The exercised standalone combination is Neo4j Community
5.26.8, its bundled APOC core in `labs/`, and JDK 21. Reuse verified distributions
read-only; do not install a host service or launch against existing story storage.
For a new synthetic trial, create a fresh private run directory and a `conf/`
subdirectory. Select an unused loopback port without stopping its current owner.
Use absolute paths for all placeholders below; Neo4j does not expand these markers.
Save this as the fresh run's `conf/neo4j.conf`:

```properties
server.directories.data=<fresh-run>/data
server.directories.logs=<fresh-run>/logs
server.directories.run=<fresh-run>/run
server.directories.plugins=<neo4j-distribution>/labs
server.directories.import=<fresh-run>/import
server.memory.heap.initial_size=512m
server.memory.heap.max_size=512m
server.memory.pagecache.size=256m
server.jvm.additional=-XX:ActiveProcessorCount=2
server.default_listen_address=127.0.0.1
server.bolt.enabled=true
server.bolt.listen_address=127.0.0.1:<unused-port>
server.bolt.advertised_address=127.0.0.1:<unused-port>
server.http.enabled=false
server.https.enabled=false
dbms.security.procedures.allowlist=apoc.*
dbms.security.procedures.unrestricted=apoc.*
dbms.usage_report.enabled=false
```

In a dedicated shell, export `JAVA_HOME` to the verified JDK, `NEO4J_HOME` to the
distribution, and `NEO4J_CONF` to the fresh run's `conf/`. Put the JDK's `bin/` on
`PATH`; set `HOME` and `TMPDIR` to the fresh run directory. Keep authentication
enabled. Set `NEO4J_PASSWORD` to a disposable synthetic password (at least eight
characters, not an existing credential). From the fresh directory:

```bash
"${NEO4J_HOME:?}/bin/neo4j-admin" dbms set-initial-password "${NEO4J_PASSWORD:?}"
timeout --signal=TERM --kill-after=40s 1800s "${NEO4J_HOME:?}/bin/neo4j" console
```

The timeout stops this disposable engine after 30 minutes; it is not suitable for
a long authoring session. Exit 124 means the timeout expired, not that authoring
succeeded. Allow clean shutdown and check that the selected port closes. Never
stop another engine or remove storage to resolve a port collision.

In a separate SAGA shell, select `NEO4J_URI=bolt://127.0.0.1:<unused-port>`,
`NEO4J_USER=neo4j` and the same disposable password, with a fresh project/output/cache.
Require authenticated Bolt/APOC/schema readiness before authoring; process existence
is not readiness. HTTP/Browser is intentionally disabled. Real stories require
operator-reviewed durable storage and a restore-tested backup. Neither this
alternative nor its startup test establishes Docker compatibility.

</details>

<details>
<summary>Optional Docker Compose smoke-test setup (not execution-verified)</summary>

The supplied Compose file is a **disposable example**, not a durable story deployment.
It publishes Bolt on loopback only, disables HTTP/HTTPS, bounds heap/page cache and
processor use, and does not automatically restart. No host story directory is mounted;
container/anonymous-volume storage is not a reviewed durable data or backup policy.
Do not use it for real stories, remove an existing container, or occupy a live service
port. Operator-reviewed durable storage and a restore-tested backup remain prerequisites.
This optional example requires a separately installed `docker-compose` executable.
Its configuration has source-level checks, but the Docker launch has not been
execution-verified. Docker is not required by SAGA.
From a fresh synthetic working directory, select an unused `SAGA_DISPOSABLE_BOLT_PORT`,
set a disposable `NEO4J_PASSWORD` of at least eight characters, and set
`COMPOSE_PROJECT_NAME` to a fresh unique synthetic name. Set `COMPOSE_FILE` to the
absolute path of this checkout's `docker-compose.yml`. Keep credentials in that
dedicated shell; do not let Compose discover an original story's `.env`. The command
below uses an explicit empty env file to prevent automatic dotenv discovery:

```bash
docker-compose --env-file /dev/null up -d
```

The historical `docker-compose up -d` invocation without isolated configuration is
not the recommended command. Compose refuses an unset/empty port or password;
schema checks alone do not prove engine/APOC readiness or shutdown. The user is
`neo4j`; configure SAGA with the same explicitly selected port/password. The
application's `saga_password` placeholder is not a production credential.

</details>

## Write your first story

Start the model and database services, configure `.env`, then ask SAGA to propose
project metadata:

```bash
python main.py bootstrap "A suspenseful thriller set inside a deep-sea facility at the bottom of the ocean"
```

Review `projects/{story_title}/config.candidate.json` before accepting the generated
metadata. Use the exact project path printed by bootstrap. A title whose normalized
directory already exists is rejected without overwriting it; choose a distinct title
for a new project. An interrupted config publication can leave a reserved directory,
but never authorizes overwriting it on retry.

Review `title`, `genre`, `theme`, `setting`, `protagonist_name`, `narrative_style`,
`total_chapters`, and `target_word_count`. For a first run, choose a small chapter
count and word target. Keep the JSON field names and types intact; chapter and word
counts must be positive integers. Then accept that candidate and start generation:

```bash
python main.py generate --project-dir "projects/{story_title}" --from-candidate
```

`--from-candidate` validates and promotes only that project's candidate. It refuses
to replace an existing `config.json`. Replace `projects/{story_title}` in these
commands with the actual path; the braces are placeholders.

To accept the model's metadata without reviewing it first, use the shortcut:

```bash
python main.py quick "A suspenseful thriller set inside a deep-sea facility at the bottom of the ocean"
```

`quick` performs bootstrap and generation. It does not export the manuscript.

## Resume and export

Continue an existing project with the same command, omitting `--from-candidate`:

```bash
python main.py generate --project-dir "projects/{story_title}"
```

Selection is mandatory: SAGA does not choose the newest project or auto-promote an
unrelated candidate. Generation uses the selected project's native checkpoint and
reconciles graph ownership and retained artifacts. It initializes only fresh state;
it does not reset checkpoints, delete files, or clear Neo4j. A successful invocation
is not necessarily a complete manuscript: the terminal summary reports the actual
checksum-verified accepted chapter count. Failures exit 1 and cancellations exit 130;
both can retain partial progress. Resume the same project, not a replacement project.

`CHAPTERS_PER_RUN` defaults to 3, so longer projects may require multiple invocations.
Read the accepted-chapter count in the terminal summary before exporting.

Neo4j ownership is exclusive to one project per configured database. Keep that
project's `graph-project-id`, files and checkpoints together. A copied identity is
a restore of the same story, not a new story fork. A new title or directory is not
permission to adopt an occupied or legacy database. Ownership/resume conflicts
require reconciliation, not a reset or removal of the identity file.

When every configured chapter is accepted:

```bash
python main.py export --project-dir "projects/{story_title}"
python main.py --help
```

Export requires exactly the configured chapters, numbered from 1 without gaps, and
checksum-valid accepted manuscript receipts. Drafts and compatibility mirrors never
supply prose. Bodies are preserved byte-for-byte, joined with two separator newlines
and one terminal newline only when needed. The success summary names the emitted
chapters and `exports/novel_full.md`. Empty, missing, corrupt or duplicate identities
fail before replacing the export. Historical unreceipted prose is not silently adopted.

## Configuration

SAGA uses Pydantic settings loaded from `.env` (see `config/settings.py`).
Process environment values take precedence over the current working directory's
`.env`, including on `config.reload()`. Reload changes only validated defaults for
future runs; active runs retain immutable snapshots. Settings import also creates
configured output/log paths, so maintenance probes must use a synthetic cwd/output
and must not run inside an original story or with inherited private configuration.

Copy [.env.example](.env.example) only for a new setup, then edit the connection
values and model names. Never commit your real `.env` or include it in bug reports.
Set `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` for your dedicated database;
the example password is a placeholder, not a secure credential.

The public example uses `qwen3-a3b` as a placeholder for every text role, not a claim
that your endpoint offers it. Set **all four** of `LARGE_MODEL`, `MEDIUM_MODEL`,
`SMALL_MODEL`, and `NARRATIVE_MODEL` to exact server model IDs. They may use the
same installed model.
`TEMPERATURE_OVERRIDE=1.0` overrides role-specific temperatures. The shipped provider
wire flag is `STRUCTURED_OUTPUT_STRICT=False`; this does not disable Pydantic,
JSON-schema, exact-ID or catalog admission. Context is 131072 tokens and the
completion/planning/summary/extraction allowances are 65536 tokens. Allowances
include reasoning where the provider counts it; they do not guarantee answer content.

Key environment variables (examples in `.env.example`):

- `OPENAI_API_BASE`: OpenAI-compatible base URL (example: `http://127.0.0.1:8080/v1`)
- `OPENAI_API_KEY`: token (can be dummy for purely local gateways)
- `EMBEDDING_API_BASE`: embeddings endpoint base URL (example: `http://127.0.0.1:11434`)
- `EMBEDDING_MODEL`: embedding model name
- `EXPECTED_EMBEDDING_DIM` and `NEO4J_VECTOR_DIMENSIONS`: must match your embedding model's output dimension

Shipped dimension defaults in `config/settings.py` (also 768 in `.env.example`):

| Setting | Shipped default |
| --- | --- |
| `EXPECTED_EMBEDDING_DIM` | 768 |
| `NEO4J_VECTOR_DIMENSIONS` | 768 |

These defaults accompany `EMBEDDING_MODEL=nomic-embed-text:latest`; they do not
describe every supported model. The explicitly supplied `qwen3-embedding:0.6b`
service produces **1024** components. To choose it, set `EMBEDDING_MODEL` and
`EMBEDDING_API_BASE` explicitly, with `EXPECTED_EMBEDDING_DIM=1024` and
`NEO4J_VECTOR_DIMENSIONS=1024` together for a **new isolated database and cache**.
Do not resize an old index, reuse a previous model's cache,
or truncate/pad embeddings to make a different model fit. Existing story indexes
and retained content must remain untouched by a new-model trial.

Keep embedding dimensions consistent across:
`EXPECTED_EMBEDDING_DIM`, `NEO4J_VECTOR_DIMENSIONS`, and your embedding model.

## Local LLM endpoint expectation

SAGA assumes generation is done via an **OpenAI-compatible HTTP API** (configured via `OPENAI_API_BASE`).

Embeddings use a separate protocol: `EMBEDDING_API_BASE` must serve `POST /api/embeddings`
with a JSON `model` and `prompt` request and an `embedding` list response, as implemented
in [the embedding client](core/http_client_service.py). An OpenAI-compatible text
endpoint alone is insufficient. Model identity and vector dimension must also match.

## Outputs

High-level output layout:

```
projects/{story_title}/
├── .saga/
│   └── content/                # Externalized content (blobs)
│       ├── drafts/
│       ├── chapter_outlines/
│       ├── summaries/
│       ├── extracted_entities/
│       └── scene_embeddings/
├── graph-project-id            # Exclusive graph identity; preserve on restore
├── checkpoints/
│   └── saga.db                 # LangGraph SQLite checkpoints
├── chapters/
│   ├── .manuscripts/           # Retained canonical Markdown and prepared receipts
│   ├── chapter_001.accepted.json # Checksum-bound accepted selection
│   └── chapter_001.md          # Compatibility mirror, not export authority
├── summaries/           # Per-chapter summaries
│   └── chapter_001.md
├── outline/             # Story/act/chapter outline YAML
│   ├── structure.yaml          # Global act/plot structure
│   └── beats.yaml              # Detailed outline per act/chapter
├── characters/          # Character profiles YAML
│   └── protagonist.yaml
├── world/               # World items + rules YAML
│   └── items.yaml
└── exports/             # Compiled manuscript exports
    └── novel_full.md
```

This layout shows representative generated artifacts, not a template to create by
hand. Bootstrap first produces `config.candidate.json`; acceptance publishes
`config.json`. Runtime logs follow `LOG_FILE` under `BASE_OUTPUT_DIR` (or console
only in simple logging mode), not a guaranteed per-project `.saga/logs/` directory.

## How it works

Initialization builds character sheets, an outline, and a catalog of story facts.
Generation plans and drafts individual scenes with retrieved graph context, extracts
entities and relationships, validates embeddings, and stages a chapter attempt.
Consistency and prose checks feed revision or acceptance. Accepted manuscripts have
checksum-bound receipts; summaries and graph maintenance support later chapters.

Graph transactions, manuscript files, and SQLite checkpoints are separate storage
boundaries. Recovery reconciles them rather than assuming one atomic write.
Do not edit checkpoints or acceptance receipts to bypass a failure.

The default `QUALITY_ACCEPTANCE_POLICY=author` records permitted quality exceptions.
Choose `strict` before starting a project to require passing narrative gates, and
`QA_ACCEPTANCE_POLICY=mandatory` to require graph-quality checks instead of advisory
cadence. Settings and acceptance policies are retained with the run; changing `.env`
does not rewrite prior decisions. Neither policy guarantees factual or literary quality.

## For contributors

### Current source map

- [CLI](main.py), [project configuration](core/project_config.py) and
  [project manager](core/project_manager.py): explicit project selection and create-only config
- [Orchestrator](orchestration/langgraph_orchestrator.py), [workflow](core/langgraph/workflow.py)
  and [state](core/langgraph/state.py): fresh initialization, validated handoff and native resume
- [Initialization](core/langgraph/initialization/), [nodes](core/langgraph/nodes/) and
  [subgraphs](core/langgraph/subgraphs/): scene-based workflow stages
- [Lifecycle](core/langgraph/chapter_lifecycle.py), [manuscripts](core/langgraph/manuscript.py)
  and [content store](core/langgraph/content_manager.py): retained attempt and file boundaries
- [Database manager](core/db_manager.py), [graph ownership](core/graph_ownership.py)
  and [queries](data_access/): project-bound graph operations
- [Run services](core/service_context.py), [LLM interface](core/llm_interface_refactored.py),
  [embedding service](core/entity_embedding_service.py) and [prompts](prompts/): model access

### Offline workflow visualization

Run from the repository root with the selected runtime. These commands compile and
inspect the graph without invoking authoring nodes or a model. Use a fresh output
path: visualization exports replace a file at the explicitly selected destination.

```bash
python visualize_workflow.py --help
python visualize_workflow.py --workflow full --summary
python visualize_workflow.py --workflow full --output output/workflow_full.mmd
python visualize_workflow.py --workflow full --format ascii --output output/workflow_full.txt
python visualize_workflow.py --all
```

PNG export is disabled: LangChain's default Mermaid PNG path calls an external
renderer, not local Graphviz. No automatic renderer installation or network fallback
is permitted. The Mermaid source or text summary is the offline deliverable.
`--all` defaults to ignored runtime storage `output/workflows/`, not source docs.

### Retained initialization acceptance and maintenance utilities

`python main.py parse --project-dir /exact/project` accepts or recovers the selected
complete frozen initialization import. It does not generate initialization, discover
legacy files or run independent parser writes. The legacy `--parser` selector is
retained only to return an explicit refusal. Follow the retained import's failure
and recovery receipts rather than replaying arbitrary character/outline files.

`verify_split.py` checks local extraction imports; `verify_subgraph.py` constructs
the extraction subgraph. Neither invokes authoring nodes nor proves model or engine
quality. Run them only with the same synthetic configuration boundary as other probes.
The optional `config.docs_generator` utility emits declared defaults, redacts secret
fields and labels settings-group factories without resolving private runtime values.
Its import still has the settings side effects described above. Retired reset/cleanup
entrypoints remain explicit refusals, not working maintenance procedures.

### Tests and static checks

See the [test boundary guide](tests/README.md), [AGENTS.md](AGENTS.md)
and [CLAUDE.md](CLAUDE.md) for Python/pytest contributor conventions.
The zero-error Mypy requirement covers maintained application code, tests and
operational tooling. Frozen evidence under `docs/audits/2026-09-06/` is excluded
from recursive maintained-code checking and remains unchanged; its historical
diagnostics are reported separately, not treated as maintained-code failures.
With the locked runtime selected, run from the intended repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests
PYTHONDONTWRITEBYTECODE=1 python -m mypy .
python -m ruff check --no-cache .
```

Pytest installs synthetic configuration and denies external I/O before application
imports. Tests use temporary storage, not real stories or a live graph. CI should
also deny network at the operating-system boundary. See the test guide for the
separate startup matrix and explicitly provisioned integration cases.
Ruff checks all configured `F`, `B`, `I`, and `UP` rules; a targeted `F` pass is not
a full lint pass. Formatting is a separate check (`python -m ruff format --check .`);
legacy formatting differences are not a reason to weaken lint or behavioral tests.

### Historical references

These are preserved history, not current commands, repair status or release approval:

- [Architecture reference (2026-02-13)](docs/langgraph-architecture.md)
- [Codebase audit (2026-02-15)](docs/CODEBASE_AUDIT_REPORT.md)
- [Field audit (2026-05-05)](docs/field-audit-20260505.md)
- [Pre-repair current-state audit](docs/CURRENT_STATE_AUDIT.md)
- [Frozen audit (2026-09-06)](docs/audits/2026-09-06/README.md)

The frozen audit's original findings and counts are intentionally unchanged.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Connection refused or authentication failure | Start the intended service and check its URL and credentials in your `.env`. The example values are not your service configuration. |
| Missing spaCy model | Install the separate language pipeline after syncing the library lock. |
| Unknown text model or rejected structured output | Use the exact server model ID for every text role and a server supporting chat completions with JSON-schema responses. |
| Embedding dimension or identity mismatch | Match the model, both dimension settings, and the database/cache created for that identity. Do not pad vectors or resize an existing story's index. |
| Project ownership or checkpoint conflict | Confirm the selected project and database belong together. Preserve both and investigate the error rather than resetting either. |
| Fewer accepted chapters than requested | Check the terminal summary, `CHAPTERS_PER_RUN`, and the logs; continue the same project. |
| Export refuses missing or corrupt chapters | Complete or reconcile the project first. Export deliberately does not substitute draft files. |

Logs go to the console and the configured `LOG_FILE` under `BASE_OUTPUT_DIR`.
For a bug report, include the command, error, Python version, server/model versions,
and relevant settings with credentials and private story content removed.

`reset_neo4j.py` and `cleanup.sh` are retained refusal entrypoints: they do not reset
databases or delete files, even with legacy flags. They are not recovery tools.
Back up the complete project and matching graph before maintenance. A new folder
name does not turn a copied graph identity into an independent story.

## License

Apache-2.0 - See [LICENSE](LICENSE).

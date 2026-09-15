# Project Bootstrapper & Configuration System

The [README](../README.md) is the maintained operational guide. This page explains
bootstrap metadata only; use the README for setup, project selection, recovery,
export and verification limits.

SAGA includes a bootstrapping system that allows users to initialize new narrative projects from high-level prompts. This system separates infrastructure configuration (API keys, models) from narrative configuration (genre, theme, setting).

## Core Concepts

### 1. Narrative Configuration
Each project is defined by a `NarrativeProjectConfig` schema (`core/project_config.py`), which includes:
- **Title**: Model-proposed title, normalized by ProjectManager for directory creation.
- **Genre**: Specific subgenre (e.g., "Hard Sci-Fi", "Cozy Mystery").
- **Theme**: Central thematic question or premise.
- **Setting**: Detailed world description.
- **Protagonist**: Name of the main character.
- **Narrative Style**: Writing style (currently defaults to "Third-person limited, past tense").
- **Total Chapters**: Target length of the novel.
- **Metadata**: Provenance info (`created_from`, `original_prompt`).

### 2. Project Bootstrapper
The `ProjectBootstrapper` (`core/project_bootstrapper.py`) converts a user's raw prompt into a structured configuration:
1. Accepts a user prompt (e.g., "A cyberpunk detective story in Neo-Tokyo").
2. Renders a Jinja2 prompt template (`prompts/initialization/bootstrap_project.j2`).
3. Calls the LLM to generate a JSON configuration.
4. Validates the JSON against the `NarrativeProjectConfig` Pydantic model.
5. Saves the result as a **candidate** configuration (`config.candidate.json`).

### 3. Project Manager
The `ProjectManager` (`core/project_manager.py`) handles the lifecycle of projects on disk:
- **Discovery**: Finds projects in the `projects/` directory.
- **Sanitization**: Converts titles to safe directory names (e.g., "My Story" -> `projects/my_story`).
- **Persistence**: Saves and loads `config.json` and `config.candidate.json`.
- **Promotion**: Validates and publishes a selected `config.candidate.json` as `config.json`, without replacing an existing active config.
- **Resume**: Requires the explicit project path; directory modification times do not select a story.

## CLI Workflow

The writer workflow has these explicit steps:

### 1. Bootstrap Mode (`bootstrap`)
Generates a candidate configuration for review.

```bash
python main.py bootstrap "A western about a robot sheriff"
```

**Outcome**:
- Creates `projects/{normalized_generated_title}/config.candidate.json`.
- Use the actual path printed by bootstrap; the model's title need not equal the prompt.
- Existing normalized title directories are rejected without overwrite.
- Users can inspect and edit this JSON file before proceeding.

### 2. Generate Mode (`generate`)
Promotes a candidate to an active project and starts generation.

```bash
python main.py generate --project-dir "projects/{normalized_generated_title}" --from-candidate
```

**Outcome**:
- Validates only the selected project's candidate and refuses an existing `config.json`.
- Publishes `config.json` without overwriting another config.
- Initializes the `LangGraphOrchestrator` with this configuration.
- Starts the narrative generation loop.

### 3. Quick Mode (`quick`)
Combines bootstrap and generate in one step (no manual review).

```bash
python main.py quick "A pirate adventure in space"
```

**Outcome**:
- Generates configuration and immediately starts generation.
- Requires configured model and database services; it is not an offline smoke command.

### 4. Explicit continuation
Resume the same selected project without candidate promotion.

```bash
python main.py generate --project-dir "projects/{normalized_generated_title}"
```

**Outcome**:
- Selects only the requested project.
- Loads its `config.json`.
- Resumes the LangGraph workflow from the last checkpoint.

## Configuration Files

### `config/settings.py` (Infrastructure)
Handles environment-specific settings via `.env`:
- LLM provider and model names.
- API keys.
- Neo4j connection details.
- Logging levels.

### `projects/<name>/config.json` (Narrative)
Handles story-specific settings:
- Plot details (theme, setting).
- Structural goals (total chapters).
- Create-only publication; do not rewrite saved checkpoints to accommodate configuration edits.

## Directory Structure

```text
projects/
└── my_story_title/
    ├── config.json          # Active configuration
    ├── config.candidate.json # Pending configuration (waiting for review)
    ├── checkpoints/         # SQLite checkpoints for LangGraph
    │   └── saga.db
    └── chapters/            # Generated artifacts
        ├── chapter_001.accepted.json
        └── chapter_001.md
```

The Markdown chapter is a compatibility mirror, not export authority. Accepted
receipts select retained checksum-valid manuscripts under `chapters/.manuscripts/`.
See the [current output contract](../README.md#outputs) for the full layout.
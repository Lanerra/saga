# Project Bootstrapper & Configuration System

SAGA includes a bootstrapping system that allows users to initialize new narrative projects from high-level prompts. This system separates infrastructure configuration (API keys, models) from narrative configuration (genre, theme, setting).

## Core Concepts

### 1. Narrative Configuration
Each project is defined by a `NarrativeProjectConfig` schema (`core/project_config.py`), which includes:
- **Title**: Sanitized project name used for directory creation.
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
- **Promotion**: Renames `config.candidate.json` to `config.json` when the user approves a candidate.
- **Resume**: Finds the most recent active project to resume generation.

## CLI Workflow

The `main.py` entry point supports three primary modes:

### 1. Bootstrap Mode (`bootstrap`)
Generates a candidate configuration for review.

```bash
python main.py bootstrap "A western about a robot sheriff"
```

**Outcome**:
- Creates `projects/a_western_about_a_robot_sheriff/config.candidate.json`.
- Users can inspect and edit this JSON file before proceeding.

### 2. Generate Mode (`generate`)
Promotes a candidate to an active project and starts generation.

```bash
python main.py generate
```

**Outcome**:
- Finds the most recent `config.candidate.json`.
- Renames it to `config.json`.
- Initializes the `LangGraphOrchestrator` with this configuration.
- Starts the narrative generation loop.

### 3. Quick Mode (`quick`)
Combines bootstrap and generate in one step (no manual review).

```bash
python main.py "A pirate adventure in space"
```

**Outcome**:
- Generates configuration and immediately starts generation.
- Useful for automated testing or rapid prototyping.

### 4. Resume Mode (Default)
Resumes the most recently modified active project.

```bash
python main.py
```

**Outcome**:
- Finds the most recent project with incomplete chapters.
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
- Immutable once generation starts (mostly).

## Directory Structure

```text
projects/
└── my_story_title/
    ├── config.json          # Active configuration
    ├── config.candidate.json # Pending configuration (waiting for review)
    ├── checkpoints/         # SQLite checkpoints for LangGraph
    │   └── saga.db
    └── chapters/            # Generated artifacts
        ├── chapter_01.md
        └── ...
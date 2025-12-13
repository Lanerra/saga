# core/langgraph/content_manager.py
"""
Content Externalization Manager for SAGA LangGraph Workflow.

This module addresses state bloat by externalizing large content fields to files,
storing only file references in the LangGraph state. This reduces SQLite checkpoint
sizes and enables efficient content versioning and diffing.

Design Goals:
1. Reduce state bloat (megabytes -> kilobytes per checkpoint)
2. Enable easy diffing of revisions
3. Maintain backward compatibility
4. Provide atomic write operations

Reference: docs/complexity-hotspots.md - "Externalize content from state"
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

import structlog

logger = structlog.get_logger(__name__)


# Type definitions for file references
class ContentRef(TypedDict, total=False):
    """Reference to externalized content."""

    path: str  # Relative path from project root
    content_type: str  # Type of content (draft, outline, summary, etc.)
    version: int  # Version number for revision tracking
    size_bytes: int  # Size for monitoring
    checksum: str  # SHA256 checksum for integrity


class ContentManager:
    """
    Manages externalized content storage and retrieval.

    All content is stored relative to the project root in a .saga/content directory.
    This enables:
    - Tiny state checkpoints (only file paths)
    - Easy content diffing (compare file versions)
    - Atomic writes with versioning
    - Fast state serialization/deserialization
    """

    def __init__(self, project_dir: str):
        """
        Initialize content manager.

        Args:
            project_dir: Project root directory
        """
        self.project_dir = Path(project_dir)
        self.content_dir = self.project_dir / ".saga" / "content"
        self.content_dir.mkdir(parents=True, exist_ok=True)

    def _get_content_path(
        self,
        content_type: str,
        identifier: str | int,
        version: int = 1,
        extension: str = "txt",
    ) -> Path:
        """
        Generate a consistent file path for content.

        Args:
            content_type: Type of content (e.g., 'draft', 'outline', 'summary')
            identifier: Unique identifier (e.g., chapter number, character name)
            version: Version number for revision tracking
            extension: File extension

        Returns:
            Absolute path to content file
        """
        # Sanitize identifier for filesystem
        safe_id = str(identifier).replace("/", "_").replace("\\", "_")

        # Create subdirectory for content type
        type_dir = self.content_dir / content_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with version
        filename = f"{safe_id}_v{version}.{extension}"
        return type_dir / filename

    def _get_relative_path(self, absolute_path: Path) -> str:
        """Convert absolute path to relative path from project root."""
        return str(absolute_path.relative_to(self.project_dir))

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum for data integrity."""
        import hashlib

        return hashlib.sha256(data).hexdigest()

    def save_text(
        self,
        content: str,
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """
        Save text content to file and return reference.

        Args:
            content: Text content to save
            content_type: Type of content (e.g., 'draft', 'summary')
            identifier: Unique identifier
            version: Version number

        Returns:
            ContentRef with file path and metadata
        """
        path = self._get_content_path(content_type, identifier, version, "txt")

        # Write content atomically (write to temp, then rename)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(content, encoding="utf-8")
        temp_path.replace(path)

        # Create reference
        data_bytes = content.encode("utf-8")
        return ContentRef(
            path=self._get_relative_path(path),
            content_type=content_type,
            version=version,
            size_bytes=len(data_bytes),
            checksum=self._compute_checksum(data_bytes),
        )

    def save_json(
        self,
        data: dict[str, Any] | list[Any],
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """
        Save JSON data to file and return reference.

        Args:
            data: JSON-serializable data
            content_type: Type of content
            identifier: Unique identifier
            version: Version number

        Returns:
            ContentRef with file path and metadata
        """
        path = self._get_content_path(content_type, identifier, version, "json")

        # Serialize once with the exact formatting we want on disk, then compute
        # checksum over the exact bytes written. This avoids mismatch between
        # `ContentRef.checksum` and on-disk content.
        json_text = json.dumps(data, indent=2, ensure_ascii=False)
        data_bytes = json_text.encode("utf-8")

        # Write content atomically (bytes) to preserve checksum correctness
        self._write_bytes_atomically(path, data_bytes)

        # Create reference (checksum computed over exact bytes written)
        return ContentRef(
            path=self._get_relative_path(path),
            content_type=content_type,
            version=version,
            size_bytes=len(data_bytes),
            checksum=self._compute_checksum(data_bytes),
        )

    def _write_bytes_atomically(self, path: Path, data_bytes: bytes) -> None:
        """
        Write bytes atomically (write to temp file, then rename).

        This is the lowest-level writer used by safe "binary" and JSON serialization
        to ensure checksums are computed from the exact bytes written on disk.
        """
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("wb") as f:
            f.write(data_bytes)
        temp_path.replace(path)

    def save_binary(
        self,
        data: Any,
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """
        Save "binary" data in a safe format.

        Security note:
        - This method intentionally does NOT use pickle.
        - Only bytes-like payloads or JSON-serializable payloads are supported.

        Supported payloads:
        - `bytes` / `bytearray` / `memoryview` -> stored as `.bin`
        - JSON-serializable objects (e.g., `list[float]` embeddings) -> stored as compact `.json`

        Args:
            data: Bytes-like payload or JSON-serializable object
            content_type: Type of content (e.g., "embedding")
            identifier: Unique identifier
            version: Version number

        Returns:
            ContentRef with file path and metadata (checksum matches on-disk bytes)
        """
        if isinstance(data, (bytes, bytearray, memoryview)):
            data_bytes = bytes(data)
            extension = "bin"
        else:
            try:
                # Compact JSON to reduce file size and ensure deterministic bytes.
                # NOTE: embeddings are typically list[float] which is JSON-serializable.
                json_text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
            except TypeError as e:
                raise TypeError(
                    "ContentManager.save_binary only supports bytes-like payloads or JSON-serializable objects. " "Refusing to serialize arbitrary Python objects (pickle is unsafe)."
                ) from e

            data_bytes = json_text.encode("utf-8")
            extension = "json"

        path = self._get_content_path(content_type, identifier, version, extension)

        # Write content atomically
        self._write_bytes_atomically(path, data_bytes)

        # Create reference (checksum computed over exact bytes written)
        return ContentRef(
            path=self._get_relative_path(path),
            content_type=content_type,
            version=version,
            size_bytes=len(data_bytes),
            checksum=self._compute_checksum(data_bytes),
        )

    def _resolve_ref_path(self, ref: ContentRef | str | Path, *, caller: str) -> str:
        """
        Resolve a `ContentRef | str | Path` into a relative-ish path string.

        Contract:
        - If `ref` is a dict-like `ContentRef`, it must contain a non-empty string at key `"path"`.
        - If `ref` is a `str` or `Path`, it is treated as a (typically relative) path under `project_dir`.

        Raises:
            ValueError: if a dict ref is missing `"path"` (or `"path"` is not a non-empty str)
            TypeError: if an unsupported ref type is provided
        """
        if isinstance(ref, Path):
            return str(ref)

        if isinstance(ref, str):
            return ref

        if isinstance(ref, dict):
            path = ref.get("path")
            if isinstance(path, str) and path:
                return path

            # Make the failure explicit and actionable (avoid KeyError hazards).
            keys = sorted(list(ref.keys()))
            raise ValueError(f"{caller} expected a ContentRef dict with required key 'path' (non-empty str); " f"got keys={keys}, ref={ref!r}")

        raise TypeError(f"{caller} expected ContentRef | str | Path; got {type(ref)}")

    def load_text(self, ref: ContentRef | str | Path) -> str:
        """
        Load text content from file reference.

        Args:
            ref: ContentRef, direct path string, or Path

        Returns:
            Text content
        """
        path_str = self._resolve_ref_path(ref, caller="ContentManager.load_text")
        full_path = self.project_dir / path_str

        if not full_path.exists():
            raise FileNotFoundError(f"Content file not found: {full_path}")

        return full_path.read_text(encoding="utf-8")

    def load_json(self, ref: ContentRef | str | Path) -> dict[str, Any] | list[Any]:
        """
        Load JSON data from file reference.

        Args:
            ref: ContentRef, direct path string, or Path

        Returns:
            Deserialized JSON data
        """
        path_str = self._resolve_ref_path(ref, caller="ContentManager.load_json")
        full_path = self.project_dir / path_str

        if not full_path.exists():
            raise FileNotFoundError(f"Content file not found: {full_path}")

        with full_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_binary(self, ref: ContentRef | str | Path) -> Any:
        """
        Load "binary" data from file reference using safe parsers only.

        Security note:
        - This method intentionally refuses to load `.pkl` artifacts.
        - If you have legacy pickle artifacts, re-generate them in the new safe format.

        Args:
            ref: ContentRef, direct path string, or Path

        Returns:
            - For `.json`: deserialized JSON object
            - For `.npy`: Python list (from a 1D numpy array) with `allow_pickle=False`
            - For other extensions (e.g., `.bin`): raw bytes
        """
        path_str = self._resolve_ref_path(ref, caller="ContentManager.load_binary")
        full_path = self.project_dir / path_str

        if not full_path.exists():
            raise FileNotFoundError(f"Content file not found: {full_path}")

        suffix = full_path.suffix.lower()

        if suffix == ".pkl":
            raise ValueError(
                "Refusing to load legacy pickle artifact (unsafe deserialization / RCE risk). "
                "Delete and re-generate embeddings/artifacts in the new safe format (.json/.npy), "
                f"or remove the file: {full_path}"
            )

        if suffix == ".json":
            with full_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        if suffix == ".npy":
            # `.npy` is safe iff `allow_pickle=False`.
            import numpy as np

            arr = np.load(full_path, allow_pickle=False)
            return arr.tolist()

        # Default: treat as raw bytes
        return full_path.read_bytes()

    def save_list_of_texts(
        self,
        texts: list[str],
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """
        Save a list of texts (e.g., scene_drafts, summaries) as JSON array.

        Args:
            texts: List of text strings
            content_type: Type of content
            identifier: Unique identifier
            version: Version number

        Returns:
            ContentRef with file path and metadata
        """
        return self.save_json(texts, content_type, identifier, version)

    def load_list_of_texts(self, ref: ContentRef | str) -> list[str]:
        """
        Load a list of texts from file reference.

        Args:
            ref: ContentRef or direct path string

        Returns:
            List of text strings
        """
        data = self.load_json(ref)
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")
        return data

    def exists(self, ref: ContentRef | str) -> bool:
        """
        Check if content file exists.

        Args:
            ref: ContentRef or direct path string

        Returns:
            True if file exists
        """
        path_str = ref["path"] if isinstance(ref, dict) else ref
        full_path = self.project_dir / path_str
        return full_path.exists()

    def delete(self, ref: ContentRef | str) -> None:
        """
        Delete content file.

        Args:
            ref: ContentRef or direct path string
        """
        path_str = ref["path"] if isinstance(ref, dict) else ref
        full_path = self.project_dir / path_str

        if full_path.exists():
            full_path.unlink()

    def get_latest_version(
        self,
        content_type: str,
        identifier: str | int,
    ) -> int:
        """
        Get the latest version number for a piece of content.

        Args:
            content_type: Type of content
            identifier: Unique identifier

        Returns:
            Latest version number (0 if no versions exist)
        """
        type_dir = self.content_dir / content_type
        if not type_dir.exists():
            return 0

        safe_id = str(identifier).replace("/", "_").replace("\\", "_")
        pattern = f"{safe_id}_v*.txt"

        versions = []
        for ext in ["txt", "json", "bin", "npy", "pkl"]:
            pattern = f"{safe_id}_v*.{ext}"
            for path in type_dir.glob(pattern):
                # Extract version from filename
                stem = path.stem  # e.g., "chapter_1_v3"
                if "_v" in stem:
                    version_str = stem.split("_v")[-1]
                    try:
                        versions.append(int(version_str))
                    except ValueError:
                        continue

        return max(versions) if versions else 0


# Convenience functions for common operations


def save_draft(
    manager: ContentManager,
    draft_text: str,
    chapter: int,
    version: int = 1,
) -> ContentRef:
    """Save chapter draft text."""
    return manager.save_text(draft_text, "draft", f"chapter_{chapter}", version)


def load_draft(manager: ContentManager, ref: ContentRef | str) -> str:
    """Load chapter draft text."""
    return manager.load_text(ref)


def save_scenes(
    manager: ContentManager,
    scene_drafts: list[str],
    chapter: int,
    version: int = 1,
) -> ContentRef:
    """Save scene drafts for a chapter."""
    return manager.save_list_of_texts(scene_drafts, "scenes", f"chapter_{chapter}", version)


def load_scenes(manager: ContentManager, ref: ContentRef | str) -> list[str]:
    """Load scene drafts for a chapter."""
    return manager.load_list_of_texts(ref)


def save_outline(
    manager: ContentManager,
    outline: dict[str, Any],
    outline_type: str,  # 'global', 'act', or 'chapter'
    identifier: str | int,
    version: int = 1,
) -> ContentRef:
    """Save an outline (global, act, or chapter)."""
    return manager.save_json(outline, f"{outline_type}_outline", identifier, version)


def load_outline(manager: ContentManager, ref: ContentRef | str) -> dict[str, Any]:
    """Load an outline."""
    data = manager.load_json(ref)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict outline, got {type(data)}")
    return data


def save_character_sheets(
    manager: ContentManager,
    character_sheets: dict[str, dict[str, Any]],
    version: int = 1,
) -> ContentRef:
    """Save character sheets."""
    return manager.save_json(character_sheets, "character_sheets", "all", version)


def load_character_sheets(manager: ContentManager, ref: ContentRef | str) -> dict[str, dict[str, Any]]:
    """Load character sheets."""
    data = manager.load_json(ref)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict character sheets, got {type(data)}")
    return cast(dict[str, dict[str, Any]], data)


def save_summaries(
    manager: ContentManager,
    summaries: list[str],
    version: int = 1,
) -> ContentRef:
    """Save chapter summaries."""
    return manager.save_list_of_texts(summaries, "summaries", "all", version)


def load_summaries(manager: ContentManager, ref: ContentRef | str) -> list[str]:
    """Load chapter summaries."""
    return manager.load_list_of_texts(ref)


def save_embedding(
    manager: ContentManager,
    embedding: list[float],
    chapter: int,
    version: int = 1,
) -> ContentRef:
    """Save chapter embedding (safe format; never pickle)."""
    return manager.save_binary(embedding, "embedding", f"chapter_{chapter}", version)


def load_embedding(manager: ContentManager, ref: ContentRef | str) -> list[float]:
    """Load chapter embedding (safe formats only)."""
    data = manager.load_binary(ref)

    # `load_binary` returns list for `.npy` and JSON-decoded value for `.json`.
    if isinstance(data, list):
        # JSON may contain ints; normalize to floats.
        normalized: list[float] = []
        for v in data:
            if isinstance(v, (int, float)):
                normalized.append(float(v))
            else:
                raise ValueError(f"Embedding vector must be numeric; got element {type(v)}")
        return normalized

    raise ValueError(f"Unexpected embedding payload type: {type(data)}")


def save_extracted_entities(
    manager: ContentManager,
    entities: dict[str, list[dict[str, Any]]],
    chapter: int,
    version: int = 1,
) -> ContentRef:
    """Save extracted entities for a chapter."""
    return manager.save_json(entities, "extracted_entities", f"chapter_{chapter}", version)


def load_extracted_entities(manager: ContentManager, ref: ContentRef | str) -> dict[str, list[dict[str, Any]]]:
    """Load extracted entities for a chapter."""
    data = manager.load_json(ref)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict extracted entities, got {type(data)}")
    return cast(dict[str, list[dict[str, Any]]], data)


def save_extracted_relationships(
    manager: ContentManager,
    relationships: list[dict[str, Any]],
    chapter: int,
    version: int = 1,
) -> ContentRef:
    """Save extracted relationships for a chapter."""
    return manager.save_json(relationships, "extracted_relationships", f"chapter_{chapter}", version)


def load_extracted_relationships(manager: ContentManager, ref: ContentRef | str) -> list[dict[str, Any]]:
    """Load extracted relationships for a chapter."""
    data = manager.load_json(ref)
    if not isinstance(data, list):
        raise ValueError(f"Expected list extracted relationships, got {type(data)}")
    return data


def save_active_characters(
    manager: ContentManager,
    characters: list[dict[str, Any]],
    chapter: int,
    version: int = 1,
) -> ContentRef:
    """Save active characters for a chapter."""
    return manager.save_json(characters, "active_characters", f"chapter_{chapter}", version)


def load_active_characters(manager: ContentManager, ref: ContentRef | str) -> list[dict[str, Any]]:
    """Load active characters for a chapter."""
    data = manager.load_json(ref)
    if not isinstance(data, list):
        raise ValueError(f"Expected list active characters, got {type(data)}")
    return data


def save_chapter_plan(
    manager: ContentManager,
    plan: list[dict[str, Any]],
    chapter: int,
    version: int = 1,
) -> ContentRef:
    """Save chapter plan (scene details) for a chapter."""
    return manager.save_json(plan, "chapter_plan", f"chapter_{chapter}", version)


def load_chapter_plan(manager: ContentManager, ref: ContentRef | str) -> list[dict[str, Any]]:
    """Load chapter plan for a chapter."""
    data = manager.load_json(ref)
    if not isinstance(data, list):
        raise ValueError(f"Expected list chapter plan, got {type(data)}")
    return data


# Helper functions for content loading from external files (Phase 3: No fallback)


def get_draft_text(state: Mapping[str, Any], manager: ContentManager) -> str | None:
    """
    Get draft text from externalized content.

    Phase 3: No fallback to in-state content. External files are required.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        Draft text or None if not available

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    import structlog

    structlog.get_logger(__name__)

    draft_ref = state.get("draft_ref")
    if not draft_ref:
        return None

    return manager.load_text(draft_ref)


def get_scene_drafts(state: Mapping[str, Any], manager: ContentManager) -> list[str]:
    """
    Get scene drafts from externalized content.

    Phase 3: No fallback to in-state content. External files are required.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        List of scene draft texts (empty list if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    scene_drafts_ref = state.get("scene_drafts_ref")
    if not scene_drafts_ref:
        return []

    return manager.load_list_of_texts(scene_drafts_ref)


def get_previous_summaries(state: Mapping[str, Any], manager: ContentManager) -> list[str]:
    """
    Get previous chapter summaries from externalized content.

    Phase 3: No fallback to in-state content. External files are required.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        List of summary texts (empty list if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    summaries_ref = state.get("summaries_ref")
    if not summaries_ref:
        return []

    return manager.load_list_of_texts(summaries_ref)


def get_hybrid_context(state: Mapping[str, Any], manager: ContentManager) -> str | None:
    """
    Get hybrid context from externalized content.

    Phase 3: No fallback to in-state content. External files are required.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        Hybrid context text or None if not available

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    hybrid_context_ref = state.get("hybrid_context_ref")
    if not hybrid_context_ref:
        return None

    return manager.load_text(hybrid_context_ref)


def get_character_sheets(state: Mapping[str, Any], manager: ContentManager) -> dict[str, dict]:
    """
    Get character sheets from externalized content.

    Phase 3: No fallback to in-state content. External files are required.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        Character sheets dict (empty dict if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    character_sheets_ref = state.get("character_sheets_ref")
    if not character_sheets_ref:
        return {}

    data = manager.load_json(character_sheets_ref)
    if not isinstance(data, dict):
        return {}
    return cast(dict[str, dict[str, Any]], data)


def get_chapter_outlines(state: Mapping[str, Any], manager: ContentManager) -> dict[int, dict]:
    """
    Get chapter outlines from externalized content.

    Phase 3: No fallback to in-state content. External files are required.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        Chapter outlines dict (empty dict if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    chapter_outlines_ref = state.get("chapter_outlines_ref")
    if not chapter_outlines_ref:
        return {}

    data = manager.load_json(chapter_outlines_ref)
    # Convert string keys to int keys if needed, skipping non-int keys
    result = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                result[int(k)] = v
            except (ValueError, TypeError):
                # Skip non-integer keys (e.g. metadata)
                pass
    return result


def get_global_outline(state: Mapping[str, Any], manager: ContentManager) -> dict | None:
    """
    Get global outline from externalized content.

    Phase 3: No fallback to in-state content. External files are required.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        Global outline dict or None if not available

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    global_outline_ref = state.get("global_outline_ref")
    if not global_outline_ref:
        return None

    data = manager.load_json(global_outline_ref)
    if not isinstance(data, dict):
        return None
    return data


def get_act_outlines(state: Mapping[str, Any], manager: ContentManager) -> dict[int, dict]:
    """
    Get act outlines from externalized content.

    Phase 3: No fallback to in-state content. External files are required.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        Act outlines dict (empty dict if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    act_outlines_ref = state.get("act_outlines_ref")
    if not act_outlines_ref:
        return {}

    data = manager.load_json(act_outlines_ref)
    # Convert string keys to int keys if needed
    result = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                result[int(k)] = v
            except (ValueError, TypeError):
                pass
    return result


def get_extracted_entities(state: Mapping[str, Any], manager: ContentManager) -> dict[str, list[dict[str, Any]]]:
    """
    Get extracted entities from externalized content.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        Extracted entities dict (empty dict if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    entities_ref = state.get("extracted_entities_ref")
    if not entities_ref:
        # Fallback to in-state content if ref not available
        return state.get("extracted_entities", {})

    data = manager.load_json(entities_ref)
    if not isinstance(data, dict):
        return {}
    return cast(dict[str, list[dict[str, Any]]], data)


def get_extracted_relationships(state: Mapping[str, Any], manager: ContentManager) -> list[dict[str, Any]]:
    """
    Get extracted relationships from externalized content.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        List of extracted relationships (empty list if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    relationships_ref = state.get("extracted_relationships_ref")
    if not relationships_ref:
        # Fallback to in-state content if ref not available
        return state.get("extracted_relationships", [])

    data = manager.load_json(relationships_ref)
    if not isinstance(data, list):
        return []
    return data


def set_extracted_relationships(
    manager: ContentManager,
    relationships: list[Any],
    state: Mapping[str, Any],
    *,
    version: int | None = None,
) -> ContentRef:
    """
    Save extracted relationships to externalized content and return the new reference.

    This is used by the relationship normalization node to persist normalized
    relationships back to externalized storage.

    Versioning contract:
    - If `version` is provided, that exact version is used.
    - If `version` is None, we write the next available version for this chapter
      using [`ContentManager.get_latest_version()`](core/langgraph/content_manager.py:334) + 1.

    Args:
        manager: ContentManager instance
        relationships: List of relationships to save (ExtractedRelationship objects or dicts)
        state: Current state (for accessing current_chapter)
        version: Optional explicit version to write

    Returns:
        ContentRef pointing at the saved relationships JSON file.
    """
    chapter = state.get("current_chapter", 1)

    # Choose version (never hardcode to 1)
    chosen_version = int(version) if version is not None else manager.get_latest_version("extracted_relationships", f"chapter_{chapter}") + 1

    # Convert to dicts for JSON serialization
    rel_dicts: list[dict[str, Any]] = []
    for r in relationships:
        # Handle both dict and object types
        if isinstance(r, dict):
            rel_dicts.append(cast(dict[str, Any], r))
        else:
            # Convert ExtractedRelationship object to dict
            rel_dicts.append(
                {
                    "source_name": r.source_name,
                    "target_name": r.target_name,
                    "relationship_type": r.relationship_type,
                    "description": r.description,
                    "chapter": r.chapter,
                    "confidence": r.confidence,
                    "source_type": getattr(r, "source_type", None),
                    "target_type": getattr(r, "target_type", None),
                }
            )

    # Save to content manager
    ref = save_extracted_relationships(manager, rel_dicts, chapter, chosen_version)

    logger.debug(
        "Saved extracted relationships",
        chapter=chapter,
        count=len(relationships),
        version=chosen_version,
        ref_path=ref.get("path") if isinstance(ref, dict) else None,
    )

    return ref


def get_active_characters(state: Mapping[str, Any], manager: ContentManager) -> list[dict[str, Any]]:
    """
    Get active characters from externalized content.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        List of active character profiles (empty list if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    characters_ref = state.get("active_characters_ref")
    if not characters_ref:
        # Fallback to in-state content if ref not available
        return state.get("active_characters", [])

    data = manager.load_json(characters_ref)
    if not isinstance(data, list):
        return []
    return data


def get_chapter_plan(state: Mapping[str, Any], manager: ContentManager) -> list[dict[str, Any]]:
    """
    Get chapter plan from externalized content.

    Args:
        state: NarrativeState dict
        manager: ContentManager instance

    Returns:
        List of scene details (empty list if not available)

    Raises:
        FileNotFoundError: If external file reference exists but file is missing
    """
    plan_ref = state.get("chapter_plan_ref")
    if not plan_ref:
        # Fallback to in-state content if ref not available
        return state.get("chapter_plan") or []

    data = manager.load_json(plan_ref)
    if not isinstance(data, list):
        return []
    return data


__all__ = [
    "ContentRef",
    "ContentManager",
    "save_draft",
    "load_draft",
    "save_scenes",
    "load_scenes",
    "save_outline",
    "load_outline",
    "save_character_sheets",
    "load_character_sheets",
    "save_summaries",
    "load_summaries",
    "save_embedding",
    "load_embedding",
    "save_extracted_entities",
    "load_extracted_entities",
    "save_extracted_relationships",
    "load_extracted_relationships",
    "save_active_characters",
    "load_active_characters",
    "save_chapter_plan",
    "load_chapter_plan",
    # Phase 2: Safe content getters with fallback
    "get_draft_text",
    "get_scene_drafts",
    "get_previous_summaries",
    "get_hybrid_context",
    "get_character_sheets",
    "get_chapter_outlines",
    "get_global_outline",
    "get_act_outlines",
    "get_extracted_entities",
    "get_extracted_relationships",
    "set_extracted_relationships",
    "get_active_characters",
    "get_chapter_plan",
]

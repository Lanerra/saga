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
import os
import pickle
from pathlib import Path
from typing import Any, TypedDict

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

        # Write content atomically
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(path)

        # Create reference
        data_bytes = json.dumps(data).encode("utf-8")
        return ContentRef(
            path=self._get_relative_path(path),
            content_type=content_type,
            version=version,
            size_bytes=len(data_bytes),
            checksum=self._compute_checksum(data_bytes),
        )

    def save_binary(
        self,
        data: Any,
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """
        Save binary data (e.g., embeddings) using pickle.

        Args:
            data: Any Python object
            content_type: Type of content
            identifier: Unique identifier
            version: Version number

        Returns:
            ContentRef with file path and metadata
        """
        path = self._get_content_path(content_type, identifier, version, "pkl")

        # Write content atomically
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(path)

        # Create reference
        data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return ContentRef(
            path=self._get_relative_path(path),
            content_type=content_type,
            version=version,
            size_bytes=len(data_bytes),
            checksum=self._compute_checksum(data_bytes),
        )

    def load_text(self, ref: ContentRef | str) -> str:
        """
        Load text content from file reference.

        Args:
            ref: ContentRef or direct path string

        Returns:
            Text content
        """
        path_str = ref["path"] if isinstance(ref, dict) else ref
        full_path = self.project_dir / path_str

        if not full_path.exists():
            raise FileNotFoundError(f"Content file not found: {full_path}")

        return full_path.read_text(encoding="utf-8")

    def load_json(self, ref: ContentRef | str) -> dict[str, Any] | list[Any]:
        """
        Load JSON data from file reference.

        Args:
            ref: ContentRef or direct path string

        Returns:
            Deserialized JSON data
        """
        path_str = ref["path"] if isinstance(ref, dict) else ref
        full_path = self.project_dir / path_str

        if not full_path.exists():
            raise FileNotFoundError(f"Content file not found: {full_path}")

        with full_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_binary(self, ref: ContentRef | str) -> Any:
        """
        Load binary data from file reference.

        Args:
            ref: ContentRef or direct path string

        Returns:
            Deserialized Python object
        """
        path_str = ref["path"] if isinstance(ref, dict) else ref
        full_path = self.project_dir / path_str

        if not full_path.exists():
            raise FileNotFoundError(f"Content file not found: {full_path}")

        with full_path.open("rb") as f:
            return pickle.load(f)

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
        for ext in ["txt", "json", "pkl"]:
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
    return manager.save_list_of_texts(
        scene_drafts, "scenes", f"chapter_{chapter}", version
    )


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
    return manager.load_json(ref)


def save_character_sheets(
    manager: ContentManager,
    character_sheets: dict[str, dict[str, Any]],
    version: int = 1,
) -> ContentRef:
    """Save character sheets."""
    return manager.save_json(character_sheets, "character_sheets", "all", version)


def load_character_sheets(
    manager: ContentManager, ref: ContentRef | str
) -> dict[str, dict[str, Any]]:
    """Load character sheets."""
    return manager.load_json(ref)


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
    """Save chapter embedding."""
    return manager.save_binary(embedding, "embedding", f"chapter_{chapter}", version)


def load_embedding(manager: ContentManager, ref: ContentRef | str) -> list[float]:
    """Load chapter embedding."""
    return manager.load_binary(ref)


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
]

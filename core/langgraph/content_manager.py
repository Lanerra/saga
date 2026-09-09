# core/langgraph/content_manager.py
"""
Manage externalized workflow content stored on disk.

This module reduces LangGraph checkpoint bloat by storing large artifacts (drafts,
outlines, embeddings, summaries) under `<project_dir>/.saga/content` and keeping
only lightweight references in workflow state.

Notes:
- Writes are durable and create-only; identical replay preserves the original inode.
- "Binary" persistence refuses unsafe formats (e.g., pickle) to avoid arbitrary
  code execution on load.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NoReturn, TypedDict, cast

import structlog

import config
from core.embedding_contract import embedding_identity, validate_embedding
from core.exceptions import ContentIntegrityError
from utils.file_io import ContainedFiles

logger = structlog.get_logger(__name__)


class FrozenContentRef(dict[str, Any]):
    """An immutable, JSON-serializable mapping used for workflow `ContentRef`s.

    Rationale:
        LangGraph checkpoints persist state. If a `ContentRef` dict is mutated after
        being placed in state, it can invalidate checkpoint assumptions and create
        confusing "time travel" behavior (old checkpoints referencing new paths).

        This type is a `dict` subclass so:
        - the standard library JSON encoder treats it as an object (serializable)
        - existing call sites using `isinstance(ref, dict)` keep working

        Mutation methods are blocked to ensure immutability.
    """

    def __init__(self, reference: dict[str, Any]) -> None:
        if self:
            self._raise_immutable()
        if not reference or any(type(value) not in (str, int) for value in reference.values()):
            raise ValueError("ContentRef requires nonempty scalar metadata")
        super().__init__(reference)

    # Mypy requires the generic dict.__or__ overload shape even for a forbidden operation.
    def __ior__(self, other: Any) -> NoReturn:  # type: ignore[misc]
        self._raise_immutable()

    def __copy__(self) -> FrozenContentRef:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenContentRef:
        return self

    def _raise_immutable(self) -> NoReturn:
        raise TypeError("ContentRef is immutable")

    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        self._raise_immutable()

    def __delitem__(self, key: str) -> None:  # type: ignore[override]
        self._raise_immutable()

    def clear(self) -> None:  # type: ignore[override]
        self._raise_immutable()

    def pop(self, key: str, default: Any = None) -> NoReturn:  # type: ignore[override]
        self._raise_immutable()

    def popitem(self) -> NoReturn:  # type: ignore[override]
        self._raise_immutable()

    def setdefault(self, key: str, default: Any = None) -> NoReturn:  # type: ignore[override]
        self._raise_immutable()

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self._raise_immutable()


def _freeze_content_ref(ref: dict[str, Any]) -> FrozenContentRef:
    return FrozenContentRef(ref)


# Type definitions for file references
class ContentRef(TypedDict, total=False):
    """Describe a reference to externalized workflow content.

    The reference resolves relative to the project root directory managed by
    [`ContentManager`](core/langgraph/content_manager.py:41).

    Notes:
        Instances returned from `ContentManager.save_*` are immutable `dict` objects
        (see `FrozenContentRef`) to prevent accidental state mutation after
        checkpointing.
    """

    path: str  # Relative path from project root
    content_type: str  # Type of content (draft, outline, summary, etc.)
    version: int  # Version number for revision tracking
    size_bytes: int  # Size for monitoring
    checksum: str  # SHA256 checksum for integrity


class ContentManager:
    """Store and load externalized workflow artifacts for a project.

    Content is stored under `<project_dir>/.saga/content/<content_type>/` and
    versioned by filename to support revision tracking without inflating state.
    """

    def __init__(self, project_dir: str):
        """Initialize a content manager rooted at `project_dir`.

        Args:
            project_dir: Project root directory. The manager creates
                `<project_dir>/.saga/content` if it does not exist.
        """
        if not isinstance(project_dir, str) or not project_dir.strip():
            raise ValueError("project_dir is required")

        self.project_dir = Path(project_dir).absolute()
        self.content_dir = self.project_dir / ".saga" / "content"
        self._files = ContainedFiles(self.content_dir, durable=True)

    def clear_cache(self) -> None:
        """Invalidate any in-process caches.

        Notes:
            The current implementation does not cache reads, so this is a no-op.
            The method exists to keep a stable invalidation surface if caching is
            introduced later.
        """
        return None

    def _get_content_path(
        self,
        content_type: str,
        identifier: str | int,
        version: int = 1,
        extension: str = "txt",
    ) -> Path:
        """Build an absolute on-disk path for a content artifact.

        Args:
            content_type: Logical bucket name (used as a subdirectory).
            identifier: Identifier used in the filename (sanitized for filesystem use).
            version: Revision counter embedded in the filename.
            extension: File extension to use.

        Returns:
            Absolute path to the content file on disk.
        """
        self._validate_component(content_type)
        self._validate_component(extension)
        if type(version) is not int or version < 0:
            raise ValueError("Content version must be a non-negative integer")
        safe_id = self._safe_identifier(identifier)
        filename = f"{safe_id}_v{version}.{extension}"
        relative = f"{content_type}/{filename}"
        self._files.exists(relative)
        return self.content_dir / relative

    @staticmethod
    def _validate_component(value: str) -> None:
        if len(ContainedFiles.relative_path(value).parts) != 1:
            raise ValueError("Content bucket and extension must be single path components")

    @staticmethod
    def _safe_identifier(identifier: str | int) -> str:
        if type(identifier) not in (str, int):
            raise ValueError("Content identifier must be a string or integer")
        safe_id = str(identifier).replace("/", "_").replace("\\", "_").replace("..", "_")
        ContentManager._validate_component(safe_id)
        return safe_id

    def _content_relative_path(self, project_relative: str) -> str:
        path = ContainedFiles.relative_path(project_relative)
        if path.parts[:2] != (".saga", "content") or len(path.parts) < 3:
            raise ValueError("Content reference must identify a file under .saga/content")
        return str(Path(*path.parts[2:]))

    def _get_relative_path(self, absolute_path: Path) -> str:
        """Convert an absolute path to a project-relative path string."""
        return str(absolute_path.relative_to(self.project_dir))

    def _compute_checksum(self, data: bytes) -> str:
        """Compute a SHA-256 checksum for the provided bytes."""
        import hashlib

        return hashlib.sha256(data).hexdigest()

    def _require_reference(self, ref: ContentRef | str | Path) -> ContentRef:
        if not isinstance(ref, dict):
            raise ValueError("Content reads require a ContentRef; path-only legacy references require explicit admit_legacy_reference")
        self._validate_component(ref.get("content_type", ""))
        version = ref.get("version")
        if type(version) is not int or version < 0:
            raise ContentIntegrityError("ContentRef.version must be a non-negative integer")
        size = ref.get("size_bytes")
        if type(size) is not int or size < 0:
            raise ContentIntegrityError("strict read requires ContentRef.size_bytes metadata as a non-negative integer")
        checksum = ref.get("checksum")
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise ContentIntegrityError("Strict read requires ContentRef.checksum as a SHA-256 hex digest")
        return ref

    def admit_legacy_reference(self, *, path: str, content_type: str, version: int, size_bytes: int, checksum: str) -> ContentRef:
        """Explicitly admit a contained legacy path using independently expected metadata.

        The caller supplies the checksum and size from a trusted checkpoint or
        preservation manifest. Never infer those values from unchecked current
        bytes. This does not migrate lifecycle state or authorize project import.
        Source bytes and names remain unchanged; no loader calls this adapter.
        """
        reference: ContentRef = {"path": path, "content_type": content_type, "version": version, "size_bytes": size_bytes, "checksum": checksum}
        self._resolve_ref_path(reference, caller="ContentManager.admit_legacy_reference")
        self._require_reference(reference)
        content = self._files.read_bytes(self._content_relative_path(path))
        self._validate_content_ref_integrity(content_ref=reference, full_path=self.project_dir / path, data_bytes=content, caller="ContentManager.admit_legacy_reference")
        return cast(ContentRef, _freeze_content_ref(dict(reference)))

    def _validate_content_ref_integrity(
        self,
        *,
        content_ref: ContentRef,
        full_path: Path,
        data_bytes: bytes,
        caller: str,
    ) -> None:
        expected_size_bytes = content_ref.get("size_bytes")
        if expected_size_bytes is None:
            raise ContentIntegrityError(
                f"{caller} strict read requires ContentRef.size_bytes metadata; path={full_path}",
            )
        if not isinstance(expected_size_bytes, int) or isinstance(expected_size_bytes, bool) or expected_size_bytes < 0:
            raise ContentIntegrityError(
                f"{caller} strict read requires ContentRef.size_bytes as non-negative int; got {expected_size_bytes!r} for path={full_path}",
            )

        actual_size_bytes = len(data_bytes)
        if actual_size_bytes != expected_size_bytes:
            raise ContentIntegrityError(
                f"{caller} strict read detected size mismatch; path={full_path}, expected={expected_size_bytes}, actual={actual_size_bytes}",
            )

        expected_checksum = content_ref.get("checksum")
        if expected_checksum is None:
            raise ContentIntegrityError(
                f"{caller} strict read requires ContentRef.checksum metadata; path={full_path}",
            )
        if not isinstance(expected_checksum, str) or not expected_checksum:
            raise ContentIntegrityError(
                f"{caller} strict read requires ContentRef.checksum as non-empty str; got {expected_checksum!r} for path={full_path}",
            )

        actual_checksum = self._compute_checksum(data_bytes)
        if actual_checksum != expected_checksum:
            raise ContentIntegrityError(
                f"{caller} strict read detected checksum mismatch; path={full_path}, expected={expected_checksum}, actual={actual_checksum}",
            )

        return None

    def save_text(
        self,
        content: str,
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """Persist UTF-8 text content and return a state reference.

        The checksum is computed over the exact bytes written to disk.

        Args:
            content: Text to write.
            content_type: Logical bucket name (subdirectory).
            identifier: Artifact identifier used in the filename.
            version: Revision counter embedded in the filename.

        Returns:
            A content reference containing the relative path and integrity metadata.
        """
        path = self._get_content_path(content_type, identifier, version, "txt")

        data_bytes = content.encode("utf-8")

        # Write content atomically (bytes) to preserve checksum correctness.
        self._write_bytes_atomically(path, data_bytes)

        return cast(
            ContentRef,
            _freeze_content_ref(
                {
                    "path": self._get_relative_path(path),
                    "content_type": content_type,
                    "version": version,
                    "size_bytes": len(data_bytes),
                    "checksum": self._compute_checksum(data_bytes),
                }
            ),
        )

    def save_json(
        self,
        data: object,
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """Persist JSON content and return a state reference.

        The checksum is computed over the exact bytes written to disk.

        Args:
            data: JSON-serializable object.
            content_type: Logical bucket name (subdirectory).
            identifier: Artifact identifier used in the filename.
            version: Revision counter embedded in the filename.

        Returns:
            A content reference containing the relative path and integrity metadata.
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
        return cast(
            ContentRef,
            _freeze_content_ref(
                {
                    "path": self._get_relative_path(path),
                    "content_type": content_type,
                    "version": version,
                    "size_bytes": len(data_bytes),
                    "checksum": self._compute_checksum(data_bytes),
                }
            ),
        )

    def _write_bytes_atomically(self, path: Path, data_bytes: bytes) -> None:
        """Create immutable bytes durably, or verify an identical replay.

        Args:
            path: Destination path.
            data_bytes: Bytes to write.
        """
        relative = self._content_relative_path(self._get_relative_path(path))
        self._files.write_bytes(relative, data_bytes, create_only=True)

    def save_binary(
        self,
        data: Any,
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """Persist bytes or JSON-serializable "binary" content safely.

        Security-sensitive behavior:
            This method refuses to serialize arbitrary Python objects (no pickle).

        Args:
            data: Bytes-like payload or JSON-serializable object.
            content_type: Logical bucket name (subdirectory).
            identifier: Artifact identifier used in the filename.
            version: Revision counter embedded in the filename.

        Returns:
            A content reference containing the relative path and integrity metadata.

        Raises:
            TypeError: If `data` is not bytes-like and is not JSON-serializable.
        """
        if isinstance(data, bytes | bytearray | memoryview):
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

        return cast(
            ContentRef,
            _freeze_content_ref(
                {
                    "path": self._get_relative_path(path),
                    "content_type": content_type,
                    "version": version,
                    "size_bytes": len(data_bytes),
                    "checksum": self._compute_checksum(data_bytes),
                }
            ),
        )

    def _resolve_ref_path(self, ref: ContentRef | str | Path, *, caller: str) -> str:
        """Resolve a content reference into a path string.

        Args:
            ref: A content reference dict (requires a non-empty `"path"` key) or a
                path-like value.
            caller: Caller label used to produce actionable error messages.

        Returns:
            A path string (typically project-relative) suitable for joining under
            `project_dir`.

        Raises:
            ValueError: If a `ContentRef` dict is missing `"path"` or `"path"` is
                not a non-empty string.
            TypeError: If `ref` is not a supported reference type.
        """
        if isinstance(ref, Path):
            path = str(ref)
        elif isinstance(ref, str):
            path = ref
        elif isinstance(ref, dict):
            reference_path = ref.get("path")
            if not isinstance(reference_path, str) or not reference_path:
                raise ValueError(f"{caller} expected a ContentRef dict with required key 'path' (non-empty str)")
            path = reference_path
        else:
            raise TypeError(f"{caller} expected ContentRef | str | Path; got {type(ref)}")
        self._content_relative_path(path)
        return path

    def load_text(self, ref: ContentRef | str | Path) -> str:
        """Load UTF-8 text from a content reference.

        Args:
            ref: A complete content reference; legacy paths require explicit admission.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the referenced file does not exist.
            TypeError: If `ref` is not a supported reference type.
            ValueError: If `ref` is a dict without a valid `"path"`, or if checksum
                validation fails.
        """
        caller = "ContentManager.load_text"
        path_str = self._resolve_ref_path(ref, caller=caller)
        content_ref = self._require_reference(ref)
        full_path = self.project_dir / path_str

        data_bytes = self._files.read_bytes(self._content_relative_path(path_str))
        self._validate_content_ref_integrity(content_ref=content_ref, full_path=full_path, data_bytes=data_bytes, caller=caller)

        return data_bytes.decode("utf-8")

    def load_text_strict(self, content_ref: ContentRef) -> str:
        """Load UTF-8 text and fail fast on `ContentRef` integrity mismatch."""
        caller = "ContentManager.load_text_strict"
        path_str = self._resolve_ref_path(content_ref, caller=caller)
        self._require_reference(content_ref)
        full_path = self.project_dir / path_str

        data_bytes = self._files.read_bytes(self._content_relative_path(path_str))
        self._validate_content_ref_integrity(content_ref=content_ref, full_path=full_path, data_bytes=data_bytes, caller=caller)
        return data_bytes.decode("utf-8")

    def load_json(self, ref: ContentRef | str | Path) -> dict[str, Any] | list[Any]:
        """Load JSON from a content reference.

        Args:
            ref: A complete content reference; legacy paths require explicit admission.

        Returns:
            The parsed JSON value.

        Raises:
            FileNotFoundError: If the referenced file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            TypeError: If `ref` is not a supported reference type.
            ValueError: If `ref` is a dict without a valid `"path"`, or if checksum
                validation fails.
        """
        caller = "ContentManager.load_json"
        path_str = self._resolve_ref_path(ref, caller=caller)
        content_ref = self._require_reference(ref)
        full_path = self.project_dir / path_str

        data_bytes = self._files.read_bytes(self._content_relative_path(path_str))
        self._validate_content_ref_integrity(content_ref=content_ref, full_path=full_path, data_bytes=data_bytes, caller=caller)

        return json.loads(data_bytes.decode("utf-8"))

    def load_json_strict(self, content_ref: ContentRef) -> Any:
        """Load JSON and fail fast on `ContentRef` integrity mismatch."""
        caller = "ContentManager.load_json_strict"
        path_str = self._resolve_ref_path(content_ref, caller=caller)
        self._require_reference(content_ref)
        full_path = self.project_dir / path_str

        data_bytes = self._files.read_bytes(self._content_relative_path(path_str))
        self._validate_content_ref_integrity(content_ref=content_ref, full_path=full_path, data_bytes=data_bytes, caller=caller)
        return json.loads(data_bytes.decode("utf-8"))

    def load_binary(self, ref: ContentRef | str | Path) -> Any:
        """Load "binary" content using safe parsers only.

        Security-sensitive behavior:
            This method refuses to load `.pkl` artifacts to avoid unsafe
            deserialization.

        Args:
            ref: A complete content reference; legacy paths require explicit admission.

        Returns:
            The parsed payload:
            - `.json` returns the decoded JSON value.
            - `.npy` returns `list` (loaded with `allow_pickle=False`).
            - Other extensions return raw bytes.

        Raises:
            FileNotFoundError: If the referenced file does not exist.
            ValueError: If the referenced file is a `.pkl` artifact, or if checksum
                validation fails.
            TypeError: If `ref` is not a supported reference type.
        """
        caller = "ContentManager.load_binary"
        path_str = self._resolve_ref_path(ref, caller=caller)
        full_path = self.project_dir / path_str

        suffix = full_path.suffix.lower()

        if suffix == ".pkl":
            raise ValueError(
                "Refusing to load legacy pickle artifact (unsafe deserialization / RCE risk). "
                "Delete and re-generate embeddings/artifacts in the new safe format (.json/.npy), "
                f"or remove the file: {full_path}"
            )

        content_ref = self._require_reference(ref)
        data_bytes = self._files.read_bytes(self._content_relative_path(path_str))
        self._validate_content_ref_integrity(content_ref=content_ref, full_path=full_path, data_bytes=data_bytes, caller=caller)

        if suffix == ".json":
            return json.loads(data_bytes.decode("utf-8"))

        if suffix == ".npy":
            import io

            import numpy as np

            arr = np.load(io.BytesIO(data_bytes), allow_pickle=False)
            return arr.tolist()

        return data_bytes

    def save_list_of_texts(
        self,
        texts: list[str],
        content_type: str,
        identifier: str | int,
        version: int = 1,
    ) -> ContentRef:
        """Persist a list of strings as JSON and return a state reference."""
        return self.save_json(texts, content_type, identifier, version)

    def load_list_of_texts(self, ref: ContentRef | str) -> list[str]:
        """Load a list of strings from a JSON content reference.

        Raises:
            ValueError: If the referenced JSON payload is not a list.
        """
        data = self.load_json(ref)
        if not isinstance(data, list):
            raise ValueError(f"Expected list, got {type(data)}")
        return data

    def exists(self, ref: ContentRef | str) -> bool:
        """Return whether the referenced content file exists on disk."""
        path = self._resolve_ref_path(ref, caller="ContentManager.exists")
        return self._files.exists(self._content_relative_path(path))

    def delete(self, ref: ContentRef | str) -> None:
        """Delete the referenced content file if it exists."""
        path = self._resolve_ref_path(ref, caller="ContentManager.delete")
        self._files.delete(self._content_relative_path(path))

    def get_latest_version(
        self,
        content_type: str,
        identifier: str | int,
    ) -> int:
        """Return the highest discovered version number for an artifact.

        Args:
            content_type: Logical bucket name (subdirectory).
            identifier: Artifact identifier used in filenames.

        Returns:
            The latest version number, or 0 when no versions exist.
        """
        self._validate_component(content_type)
        safe_id = self._safe_identifier(identifier)
        pattern = re.compile(rf"{re.escape(safe_id)}_v([0-9]+)\.(txt|json|bin|npy|pkl)")
        versions = []
        for name in self._files.list_names(content_type):
            match = pattern.fullmatch(name)
            if match is not None and self._files.exists(f"{content_type}/{name}"):
                versions.append(int(match.group(1)))
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
    if isinstance(ref, dict):
        return manager.load_text_strict(ref)
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
    data = manager.load_json_strict(ref) if isinstance(ref, dict) else manager.load_json(ref)
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
    *, embedding_model: str = "",
) -> ContentRef:
    """Save a chapter embedding in a safe, non-pickle format."""
    vector = validate_embedding(embedding, model=embedding_model).tolist()
    return manager.save_json({"model": embedding_model, "identity": embedding_identity(), "vector": vector}, "embedding", f"chapter_{chapter}", version)


def load_embedding(manager: ContentManager, ref: ContentRef | str) -> list[float]:
    """Load a chapter embedding (safe formats only)."""
    data = manager.load_json_strict(ref) if isinstance(ref, dict) else manager.load_json(ref)
    if not isinstance(data, dict) or set(data) != {"model", "identity", "vector"} or data["identity"] != embedding_identity():
        raise ValueError("Embedding artifact identity mismatch; regenerate from source text")
    return validate_embedding(data["vector"], model=data["model"]).tolist()


def save_scene_embeddings(
    manager: ContentManager,
    embeddings: list[list[float]],
    chapter: int,
    version: int = 1,
    *, embedding_model: str = "",
) -> ContentRef:
    """Save scene-level embeddings for a chapter as a single JSON artifact."""
    if not isinstance(embeddings, list):
        raise TypeError(f"scene embeddings must be a list; got {type(embeddings)}")

    for embedding_index, embedding in enumerate(embeddings):
        if not isinstance(embedding, list):
            raise TypeError("each scene embedding must be a list of floats; " f"index={embedding_index}, got {type(embedding)}")
        for value_index, value in enumerate(embedding):
            if isinstance(value, bool):
                raise TypeError("scene embedding values must be float (bool is not allowed); " f"scene_index={embedding_index}, value_index={value_index}")
            if not isinstance(value, float):
                raise TypeError("scene embedding values must be float; " f"scene_index={embedding_index}, value_index={value_index}, got {type(value)}")

    vectors = [validate_embedding(vector, model=embedding_model).tolist() for vector in embeddings]
    if embedding_model != config.EMBEDDING_MODEL:
        raise ValueError("Scene embedding model identity mismatch")
    return manager.save_json({"model": embedding_model, "identity": embedding_identity(), "vectors": vectors}, "scene_embeddings", f"chapter_{chapter}", version)


def load_scene_embeddings(manager: ContentManager, ref: ContentRef | str) -> list[list[float]]:
    """Load scene-level embeddings for a chapter (safe formats only)."""
    payload = manager.load_json_strict(ref) if isinstance(ref, dict) else manager.load_json(ref)
    if not isinstance(payload, dict) or set(payload) != {"model", "identity", "vectors"} or payload["identity"] != embedding_identity() or payload["model"] != config.EMBEDDING_MODEL:
        raise ValueError("Scene embedding artifact identity mismatch; regenerate from source text")
    data = payload["vectors"]

    if not isinstance(data, list):
        raise ValueError(f"scene embeddings payload must be a list; got {type(data)}")

    embeddings: list[list[float]] = []
    for embedding_index, embedding in enumerate(data):
        if not isinstance(embedding, list):
            raise ValueError("each scene embedding must be a list of floats; " f"index={embedding_index}, got {type(embedding)}")

        vector: list[float] = []
        for value_index, value in enumerate(embedding):
            if isinstance(value, bool):
                raise ValueError("scene embedding values must be numeric (bool is not allowed); " f"scene_index={embedding_index}, value_index={value_index}")
            if not isinstance(value, int | float):
                raise ValueError("scene embedding values must be numeric; " f"scene_index={embedding_index}, value_index={value_index}, got {type(value)}")
            vector.append(float(value))

        embeddings.append(validate_embedding(vector, model=payload["model"]).tolist())

    return embeddings


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
    data = manager.load_json_strict(ref) if isinstance(ref, dict) else manager.load_json(ref)
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
    data = manager.load_json_strict(ref) if isinstance(ref, dict) else manager.load_json(ref)
    if not isinstance(data, list):
        raise ValueError(f"Expected list extracted relationships, got {type(data)}")
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
    data = manager.load_json_strict(ref) if isinstance(ref, dict) else manager.load_json(ref)
    if not isinstance(data, list):
        raise ValueError(f"Expected list chapter plan, got {type(data)}")
    return data


# Helper functions for loading content from externalized files.


def get_draft_text(state: Mapping[str, Any], manager: ContentManager) -> str:
    """Load the current chapter draft text from `draft_ref`.

    Contract:
        Drafts are refs-only. Inline `draft_text` is not part of the state contract.

    Args:
        state: Workflow state mapping. This function reads `draft_ref`.
        manager: Content manager rooted at the project directory.

    Returns:
        The draft text.

    Raises:
        MissingDraftReferenceError: If `draft_ref` is missing or falsy.
        FileNotFoundError: If `draft_ref` is present but the referenced file is missing.
        ContentIntegrityError: If `draft_ref` is a `ContentRef` and strict integrity validation fails.
    """
    from core.exceptions import MissingDraftReferenceError

    draft_ref = state.get("draft_ref")
    if not draft_ref:
        raise MissingDraftReferenceError("Missing required state key: draft_ref")

    if isinstance(draft_ref, dict):
        return manager.load_text_strict(cast(ContentRef, draft_ref))
    return manager.load_text(draft_ref)


def get_scene_drafts(state: Mapping[str, Any], manager: ContentManager) -> list[str]:
    """Load scene drafts from `scene_drafts_ref`.

    Args:
        state: Workflow state mapping. This function reads `scene_drafts_ref`.
        manager: Content manager rooted at the project directory.

    Returns:
        Scene draft texts, or an empty list when `scene_drafts_ref` is missing.

    Raises:
        FileNotFoundError: If `scene_drafts_ref` is present but the referenced file is missing.
        ValueError: If the referenced JSON payload is not a list.
    """
    scene_drafts_ref = state.get("scene_drafts_ref")
    if not scene_drafts_ref:
        return []

    return manager.load_list_of_texts(scene_drafts_ref)


def get_previous_summaries(state: Mapping[str, Any], manager: ContentManager) -> list[str]:
    """Load prior chapter summaries from `summaries_ref`.

    Args:
        state: Workflow state mapping. This function reads `summaries_ref`.
        manager: Content manager rooted at the project directory.

    Returns:
        Summary texts, or an empty list when `summaries_ref` is missing.

    Raises:
        FileNotFoundError: If `summaries_ref` is present but the referenced file is missing.
        ValueError: If the referenced JSON payload is not a list.
    """
    summaries_ref = state.get("summaries_ref")
    if not summaries_ref:
        return []

    return manager.load_list_of_texts(summaries_ref)


def get_hybrid_context(state: Mapping[str, Any], manager: ContentManager) -> str | None:
    """Load the hybrid context block from `hybrid_context_ref`.

    Args:
        state: Workflow state mapping. This function reads `hybrid_context_ref`.
        manager: Content manager rooted at the project directory.

    Returns:
        Hybrid context text, or `None` when `hybrid_context_ref` is missing.

    Raises:
        FileNotFoundError: If `hybrid_context_ref` is present but the referenced file is missing.
    """
    hybrid_context_ref = state.get("hybrid_context_ref")
    if not hybrid_context_ref:
        return None

    return manager.load_text(hybrid_context_ref)


def get_character_sheets(state: Mapping[str, Any], manager: ContentManager) -> dict[str, dict]:
    """Load character sheets from `character_sheets_ref`.

    Args:
        state: Workflow state mapping. This function reads `character_sheets_ref`.
        manager: Content manager rooted at the project directory.

    Returns:
        Character sheets keyed by character name, or an empty dict when
        `character_sheets_ref` is missing or the referenced payload is not a dict.

    Raises:
        FileNotFoundError: If `character_sheets_ref` is present but the referenced file is missing.
    """
    character_sheets_ref = state.get("character_sheets_ref")
    if not character_sheets_ref:
        return {}

    data = manager.load_json(character_sheets_ref)
    if not isinstance(data, dict):
        return {}
    return cast(dict[str, dict[str, Any]], data)


def get_chapter_outlines(state: Mapping[str, Any], manager: ContentManager) -> dict[int, dict]:
    """Load chapter outlines from `chapter_outlines_ref`.

    The externalized JSON may use string keys; this function converts keys that
    parse as integers into `int` keys and ignores non-integer keys.

    Args:
        state: Workflow state mapping. This function reads `chapter_outlines_ref`.
        manager: Content manager rooted at the project directory.

    Returns:
        Outlines keyed by chapter number, or an empty dict when `chapter_outlines_ref`
        is missing or the referenced payload is not a dict.

    Raises:
        FileNotFoundError: If `chapter_outlines_ref` is present but the referenced file is missing.
    """
    chapter_outlines_ref = state.get("chapter_outlines_ref")
    if not chapter_outlines_ref:
        return {}

    data = manager.load_json_strict(cast(ContentRef, chapter_outlines_ref)) if isinstance(chapter_outlines_ref, dict) else manager.load_json(chapter_outlines_ref)
    # Convert string keys to int keys if needed, skipping non-int keys
    result = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(k, str) and k.isdigit():
                result[int(k)] = v
            elif isinstance(k, int):
                result[k] = v
    return result


def get_global_outline(state: Mapping[str, Any], manager: ContentManager) -> dict | None:
    """Load the global outline from `global_outline_ref`.

    Args:
        state: Workflow state mapping. This function reads `global_outline_ref`.
        manager: Content manager rooted at the project directory.

    Returns:
        Global outline dict, or `None` when `global_outline_ref` is missing or the
        referenced payload is not a dict.

    Raises:
        FileNotFoundError: If `global_outline_ref` is present but the referenced file is missing.
    """
    global_outline_ref = state.get("global_outline_ref")
    if not global_outline_ref:
        return None

    data = manager.load_json_strict(cast(ContentRef, global_outline_ref)) if isinstance(global_outline_ref, dict) else manager.load_json(global_outline_ref)
    if not isinstance(data, dict):
        return None
    return data


def get_act_outlines(state: Mapping[str, Any], manager: ContentManager) -> dict[int, dict]:
    """Load act outlines from `act_outlines_ref` (supports multiple formats).

    Accepted externalized shapes:
        - v1 mapping: `{<act_number>: <outline_dict>}` where act_number is an `int`
          or a string that parses to an int.
        - v2 list: `[{"act_number": 1, ...}, {"act_number": 2, ...}]`
        - v2 container: `{"format_version": 2, "acts": [ ... ]}`

    Args:
        state: Workflow state mapping. This function reads `act_outlines_ref`.
        manager: Content manager rooted at the project directory.

    Returns:
        A dict keyed by act number, or an empty dict when `act_outlines_ref` is missing.

    Raises:
        FileNotFoundError: If `act_outlines_ref` is present but the referenced file is missing.
        ValueError: If the externalized data is malformed or uses an unsupported format version.
    """

    def _normalize_v1_mapping(raw: dict[object, object]) -> dict[int, dict]:
        result_by_act_number: dict[int, dict] = {}
        for key, value in raw.items():
            if isinstance(key, bool):
                raise ValueError("Act outlines v1 key must be an integer act number; got bool")

            if isinstance(key, int):
                act_number = key
            elif isinstance(key, str):
                try:
                    act_number = int(key)
                except ValueError as error:
                    raise ValueError(f"Act outlines v1 key must be an integer act number; got {key!r}") from error
            else:
                raise ValueError(f"Act outlines v1 key must be an integer act number; got {type(key)}")

            if isinstance(act_number, bool) or act_number <= 0:
                raise ValueError(f"Act outlines v1 key must be a positive integer; got {act_number!r}")

            if not isinstance(value, dict):
                raise ValueError(f"Act outlines v1 value must be a dict for act_number={act_number}; got {type(value)}")

            if act_number in result_by_act_number:
                raise ValueError(f"Duplicate act_number in act outlines v1 mapping: {act_number}")

            result_by_act_number[act_number] = value

        return {act_number: result_by_act_number[act_number] for act_number in sorted(result_by_act_number.keys())}

    def _normalize_v2_list(raw: list[object]) -> dict[int, dict]:
        result_by_act_number: dict[int, dict] = {}
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"Act outlines v2 list items must be dict; index={index}, got {type(item)}")

            act_number = item.get("act_number")
            if not isinstance(act_number, int) or isinstance(act_number, bool) or act_number <= 0:
                raise ValueError(f"Act outlines v2 items must include act_number as positive int; index={index}, got {act_number!r}")

            if act_number in result_by_act_number:
                raise ValueError(f"Duplicate act_number in act outlines v2 list: {act_number}")

            result_by_act_number[act_number] = item

        return {act_number: result_by_act_number[act_number] for act_number in sorted(result_by_act_number.keys())}

    act_outlines_ref = state.get("act_outlines_ref")
    if not act_outlines_ref:
        return {}

    if isinstance(act_outlines_ref, dict) and "checksum" in act_outlines_ref and "size_bytes" in act_outlines_ref:
        data = manager.load_json_strict(cast(ContentRef, act_outlines_ref))
    else:
        data = manager.load_json(act_outlines_ref)

    if isinstance(data, list):
        return _normalize_v2_list(cast(list[object], data))

    if not isinstance(data, dict):
        raise ValueError(f"Act outlines externalized content must be dict or list; got {type(data)}")

    format_version = data.get("format_version")
    if format_version is not None:
        if not isinstance(format_version, int) or isinstance(format_version, bool):
            raise ValueError(f"Act outlines container format_version must be int; got {type(format_version)}")

        if format_version != 2:
            raise ValueError(f"Unsupported act outlines container format_version: {format_version}")

        acts = data.get("acts")
        if not isinstance(acts, list):
            raise ValueError(f"Act outlines v2 container must contain 'acts' list; got {type(acts)}")

        return _normalize_v2_list(cast(list[object], acts))

    return _normalize_v1_mapping(cast(dict[object, object], data))


def get_extracted_entities(state: Mapping[str, Any], manager: ContentManager) -> dict[str, list[dict[str, Any]]]:
    """Load extracted entities for the current chapter.

    This function prefers externalized content via `extracted_entities_ref` and
    falls back to in-state `extracted_entities` for back-compat.

    Args:
        state: Workflow state mapping.
        manager: Content manager rooted at the project directory.

    Returns:
        A mapping with extraction buckets (e.g., `"characters"`, `"world_items"`).
        Returns an empty dict when neither externalized nor in-state content is present.

    Raises:
        FileNotFoundError: If `extracted_entities_ref` is present but the referenced file is missing.
    """
    entities_ref = state.get("extracted_entities_ref")
    if not entities_ref:
        # Fallback to in-state content if ref not available
        return state.get("extracted_entities", {})

    data = manager.load_json_strict(cast(ContentRef, entities_ref)) if isinstance(entities_ref, dict) else manager.load_json(entities_ref)
    if not isinstance(data, dict):
        return {}
    return cast(dict[str, list[dict[str, Any]]], data)


def get_extracted_relationships(state: Mapping[str, Any], manager: ContentManager) -> list[dict[str, Any]]:
    """Load extracted relationships for the current chapter.

    This function prefers externalized content via `extracted_relationships_ref`
    and falls back to in-state `extracted_relationships` for back-compat.

    Args:
        state: Workflow state mapping.
        manager: Content manager rooted at the project directory.

    Returns:
        Extracted relationships as a list of dicts, or an empty list when neither
        externalized nor in-state content is present.

    Raises:
        FileNotFoundError: If `extracted_relationships_ref` is present but the referenced file is missing.
    """
    relationships_ref = state.get("extracted_relationships_ref")
    if not relationships_ref:
        # Fallback to in-state content if ref not available
        return state.get("extracted_relationships", [])

    data = manager.load_json_strict(cast(ContentRef, relationships_ref)) if isinstance(relationships_ref, dict) else manager.load_json(relationships_ref)
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
    """Persist extracted relationships and return the new reference.

    This is used by normalization nodes to write an updated relationship list.

    Versioning contract:
        - If `version` is provided, that exact version is used.
        - If `version` is `None`, this writes the next available version for the
          current chapter.

    Args:
        manager: Content manager rooted at the project directory.
        relationships: Relationship objects or dicts. Objects must expose
            `source_name`, `target_name`, `relationship_type`, `description`,
            `chapter`, `confidence`.
        state: Workflow state mapping. This function reads `current_chapter`.
        version: Optional explicit version to write.

    Returns:
        A `ContentRef` pointing at the saved relationships JSON file.
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


def get_chapter_plan(state: Mapping[str, Any], manager: ContentManager) -> list[dict[str, Any]]:
    """Load the chapter plan (scene details) for the current chapter.

    This function prefers externalized content via `chapter_plan_ref` and falls
    back to in-state `chapter_plan` for back-compat.

    Args:
        state: Workflow state mapping.
        manager: Content manager rooted at the project directory.

    Returns:
        A list of scene detail dicts, or an empty list when unavailable.

    Raises:
        FileNotFoundError: If `chapter_plan_ref` is present but the referenced file is missing.
    """
    plan_ref = state.get("chapter_plan_ref")
    if not plan_ref:
        # Fallback to in-state content if ref not available
        return state.get("chapter_plan") or []

    data = manager.load_json_strict(cast(ContentRef, plan_ref)) if isinstance(plan_ref, dict) else manager.load_json(plan_ref)
    if not isinstance(data, list):
        return []
    return data


def require_project_dir(state: Mapping[str, Any]) -> str:
    project_dir = state.get("project_dir")
    if not isinstance(project_dir, str) or not project_dir.strip():
        raise ValueError("project_dir is required")
    return project_dir


__all__ = [
    "ContentRef",
    "ContentManager",
    "require_project_dir",
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
    "save_scene_embeddings",
    "load_scene_embeddings",
    "save_extracted_entities",
    "load_extracted_entities",
    "save_extracted_relationships",
    "load_extracted_relationships",
    "save_chapter_plan",
    "load_chapter_plan",
    # Content getters (some include fallback for back-compat)
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
    "get_chapter_plan",
]

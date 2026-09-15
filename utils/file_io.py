# utils/file_io.py
from __future__ import annotations

import fcntl
import os
import secrets
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


class ContainedFiles:
    """Operate beneath a trusted root without following symbolic links (POSIX).

    Every operation walks directories using O_NOFOLLOW and directory descriptors,
    including the root's ancestors. A checked Path alone is not a capability.
    Directory owners must not move opened directories out of the root, introduce
    mounts or hard links to outside data, or modify private temporary files.
    This is a reference boundary, not isolation from a hostile same-user process.

    durable=True also syncs ancestor directory entries during creation/walk and
    the containing directory after replacement/deletion. Errors propagate;
    post-rename fsync failure means visibility changed but durability is unknown.
    Guarantees assume a POSIX filesystem and device honoring fsync.
    """

    def __init__(self, root: Path, *, durable: bool = False) -> None:
        self.root = root.absolute()
        self.durable = durable
        if ".." in self.root.parts:
            raise ValueError("Content root must not contain parent traversal")
        with self._directory(Path(), create=True):
            pass

    @staticmethod
    def relative_path(value: str) -> Path:
        """Validate a canonical relative file or directory name before any I/O."""
        if not isinstance(value, str) or not value.strip() or "\x00" in value or "\\" in value:
            raise ValueError("Content path must be a non-empty relative path")
        if Path(value).is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
            raise ValueError("Content path must not be absolute or contain traversal")
        return Path(value)

    @contextmanager
    def _directory(self, relative: Path, *, create: bool = False) -> Iterator[int]:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open("/", flags)
        try:
            for component in (*self.root.parts[1:], *relative.parts):
                if create:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                child = os.open(component, flags, dir_fd=descriptor)
                if create and self.durable:
                    try:
                        os.fsync(descriptor)
                    except BaseException:
                        os.close(child)
                        raise
                os.close(descriptor)
                descriptor = child
            yield descriptor
        finally:
            os.close(descriptor)

    @staticmethod
    def _require_regular_file(descriptor: int, name: str) -> None:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Content path must identify a regular file, not a link or directory")

    def read_bytes(self, value: str) -> bytes:
        path = self.relative_path(value)
        with self._directory(path.parent) as parent:
            return self._read_at(parent, path.name)

    def _read_at(self, parent: int, name: str, *, sync: bool = False) -> bytes:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("Content path must identify a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as content:
                data = content.read()
            if sync:
                os.fsync(descriptor)
            return data
        finally:
            os.close(descriptor)

    def _verify_existing(self, parent: int, name: str, data: bytes) -> None:
        if self._read_at(parent, name, sync=self.durable) != data:
            raise FileExistsError(f"Immutable content conflict: {name}")
        if self.durable:
            os.fsync(parent)

    def exists(self, value: str) -> bool:
        path = self.relative_path(value)
        try:
            with self._directory(path.parent) as parent:
                self._require_regular_file(parent, path.name)
        except FileNotFoundError:
            return False
        return True

    def delete(self, value: str) -> None:
        path = self.relative_path(value)
        try:
            with self._directory(path.parent) as parent:
                self._require_regular_file(parent, path.name)
                os.unlink(path.name, dir_fd=parent)
                if self.durable:
                    os.fsync(parent)
        except FileNotFoundError:
            return

    def write_bytes(self, value: str, data: bytes, *, create_only: bool = False) -> None:
        """Publish complete bytes; create-only permits identical replay, not replacement.

        Exclusive linking publishes a fully fsynced temporary inode. Concurrent
        creators verify the winner's bytes. A crash can leave an unreferenced
        private temporary file, never a partially written final artifact.
        """
        path = self.relative_path(value)
        with self._directory(path.parent, create=True) as parent:
            try:
                self._require_regular_file(parent, path.name)
            except FileNotFoundError:
                pass
            else:
                if create_only:
                    self._verify_existing(parent, path.name, data)
                    return
            temporary = f".saga-{secrets.token_hex(16)}.tmp"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
            published = False
            try:
                with os.fdopen(descriptor, "wb") as content:
                    content.write(data)
                    content.flush()
                    os.fsync(content.fileno())
                if create_only:
                    try:
                        os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                    except FileExistsError:
                        self._verify_existing(parent, path.name, data)
                else:
                    os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
                    published = True
                if self.durable:
                    os.fsync(parent)
            finally:
                if not published:
                    os.unlink(temporary, dir_fd=parent)
                    if self.durable:
                        os.fsync(parent)

    @contextmanager
    def exclusive_lock(self, value: str) -> Iterator[None]:
        """Admit one local writer; keep the lock inode stable across reopen."""
        path = self.relative_path(value)
        with self._directory(path.parent, create=True) as parent:
            descriptor = os.open(path.name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600, dir_fd=parent)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ValueError("Writer lock must be a regular file")
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield
            finally:
                os.close(descriptor)

    def list_names(self, value: str) -> list[str]:
        path = self.relative_path(value)
        try:
            with self._directory(path) as directory:
                return os.listdir(directory)
        except FileNotFoundError:
            return []


def _atomic_write(target: Path, data: str) -> None:
    """Write data to a file atomically via temp file + fsync + rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        os.unlink(tmp_path)
        raise


def write_text_file(path: str | Path, text: str) -> None:
    """
    Write text content to a file atomically with consistent UTF-8 encoding and LF newlines.

    Guarantees:
    - Converts ``path`` to ``Path``.
    - Ensures parent directories exist.
    - Writes with encoding="utf-8" and newline="\\n".
    - Atomic: writes to a temp file, fsyncs, then renames.
    - No logging side effects.
    """
    target = Path(path)
    data = str(text).replace("\r\n", "\n").replace("\r", "\n")
    _atomic_write(target, data)


def write_yaml_file(path: str | Path, data: Any) -> None:
    """
    Write YAML content to a file atomically with consistent UTF-8 encoding and LF newlines.

    Guarantees:
    - Converts ``path`` to ``Path``.
    - Ensures parent directories exist.
    - Uses yaml.dump with:
        - default_flow_style=False
        - sort_keys=False
        - allow_unicode=True
      (callers can rely on globally-registered representers such as _LiteralString)
    - Writes with encoding="utf-8" and newline="\\n".
    - Atomic: writes to a temp file, fsyncs, then renames.
    - No logging or global YAML configuration changes.
    """
    target = Path(path)
    yaml_text = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    yaml_text = yaml_text.replace("\r\n", "\n").replace("\r", "\n")
    _atomic_write(target, yaml_text)

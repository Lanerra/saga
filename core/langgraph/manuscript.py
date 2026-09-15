"""Single-writer POSIX manuscript publication, independent of workflow reconciliation.

A prepared receipt retains canonical Markdown before either compatibility mirror
is replaced. Only the accepted receipt selects prose for export. The caller must
obtain graph acknowledgement before accept(); this is not a cross-store transaction.
Mirrors may be mixed after interruption: recover() rebuilds both from acceptance,
never promotes a prepared candidate. Retained objects are not garbage-collected.
Existing mirrors must independently match verified accepted/prepared bytes before
publication. Unreceipted historical prose and unretained author edits block writes;
they require explicit operator reconciliation, not automatic migration/adoption.
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from utils.file_io import ContainedFiles


class ManuscriptReceipt(BaseModel):
    """Exact UTF-8 Markdown identity and byte range of its unmodified prose."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    chapter_number: int = Field(gt=0)
    artifact_path: str
    markdown_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    markdown_size: int = Field(gt=0)
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    body_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = f"chapters/.manuscripts/chapter_{self.chapter_number:03d}/{self.markdown_sha256}.md"
        if self.artifact_path != expected or self.body_offset >= self.markdown_size:
            raise ValueError("Invalid manuscript artifact identity or body range")
        return self

    @property
    def accepted_path(self) -> str:
        return f"chapters/chapter_{self.chapter_number:03d}.accepted.json"

    @property
    def prepared_path(self) -> str:
        return self.artifact_path.removesuffix(".md") + ".prepared.json"


class ManuscriptStore:
    """Retain candidates and atomically select accepted Markdown beneath a project."""

    def __init__(self, project_dir: Path) -> None:
        self.files = ContainedFiles(project_dir, durable=True)

    def read(self, receipt: ManuscriptReceipt) -> tuple[bytes, bytes]:
        markdown = self.files.read_bytes(receipt.artifact_path)
        if len(markdown) != receipt.markdown_size or hashlib.sha256(markdown).hexdigest() != receipt.markdown_sha256:
            raise ValueError("Canonical manuscript checksum or size mismatch")
        body = markdown[receipt.body_offset:]
        if hashlib.sha256(body).hexdigest() != receipt.body_sha256:
            raise ValueError("Canonical manuscript body checksum mismatch")
        markdown.decode("utf-8")
        return markdown, body

    def _write_verified(self, path: str, content: bytes) -> None:
        self.files.write_bytes(path, content)
        if self.files.read_bytes(path) != content:
            raise ValueError("Manuscript publication readback mismatch")

    def _require_owned_mirrors(self, chapter_number: int) -> None:
        prefix = f"chapters/chapter_{chapter_number:03d}"
        unmatched = {suffix: self.files.read_bytes(prefix + suffix) for suffix in (".md", ".txt") if self.files.exists(prefix + suffix)}
        receipts = []
        if self.files.exists(prefix + ".accepted.json"):
            receipts.append(self.accepted(chapter_number))
        directory = f"chapters/.manuscripts/chapter_{chapter_number:03d}"
        for name in sorted(self.files.list_names(directory)):
            if name.endswith(".prepared.json"):
                path = f"{directory}/{name}"
                receipt = ManuscriptReceipt.model_validate_json(self.files.read_bytes(path))
                if receipt.chapter_number != chapter_number or receipt.prepared_path != path:
                    raise ValueError("Prepared manuscript receipt identity mismatch")
                receipts.append(receipt)
        for receipt in receipts:
            markdown, body = self.read(receipt)
            for suffix, content in ((".md", markdown), (".txt", body)):
                if unmatched.get(suffix) == content:
                    unmatched.pop(suffix)
        if unmatched:
            raise ValueError(f"Unowned manuscript mirror requires explicit reconciliation: {', '.join(prefix + suffix for suffix in unmatched)}")

    def prepare(self, chapter_number: int, text: str) -> ManuscriptReceipt:
        if type(chapter_number) is not int or chapter_number <= 0 or not text:
            raise ValueError("Manuscript requires a positive chapter number and nonempty prose")
        self._require_owned_mirrors(chapter_number)
        front_matter = (
            f"---\nchapter: {chapter_number}\ntitle: Chapter {chapter_number}\n"
            f"word_count: {len(text.split())}\ngenerated_at: {datetime.now(UTC).isoformat()}\nversion: 1\n---\n"
        ).encode()
        body = text.encode("utf-8")
        markdown = front_matter + body
        checksum = hashlib.sha256(markdown).hexdigest()
        receipt = ManuscriptReceipt(
            chapter_number=chapter_number,
            artifact_path=f"chapters/.manuscripts/chapter_{chapter_number:03d}/{checksum}.md",
            markdown_sha256=checksum,
            markdown_size=len(markdown),
            body_sha256=hashlib.sha256(body).hexdigest(),
            body_offset=len(front_matter),
        )
        if self.files.exists(receipt.artifact_path):
            self.read(receipt)
        else:
            self._write_verified(receipt.artifact_path, markdown)
        self._write_verified(receipt.prepared_path, receipt.model_dump_json().encode("utf-8"))
        self._write_mirrors(receipt)
        return receipt

    def _write_mirrors(self, receipt: ManuscriptReceipt) -> None:
        markdown, body = self.read(receipt)
        prefix = f"chapters/chapter_{receipt.chapter_number:03d}"
        for suffix, content in ((".md", markdown), (".txt", body)):
            path = prefix + suffix
            if not self.files.exists(path) or self.files.read_bytes(path) != content:
                self._write_verified(path, content)
        self._sync_publication_directory()

    def _sync_publication_directory(self) -> None:
        # Visible identical bytes can follow interruption before directory fsync.
        with self.files._directory(Path("chapters")) as directory:
            os.fsync(directory)

    def resume_prepared(self, receipt: ManuscriptReceipt) -> None:
        """Restore the exact prepared candidate without conferring acceptance."""
        prepared = ManuscriptReceipt.model_validate_json(self.files.read_bytes(receipt.prepared_path))
        if prepared != receipt:
            raise ValueError("Prepared manuscript receipt mismatch")
        self.read(receipt)
        self._require_owned_mirrors(receipt.chapter_number)
        self._write_mirrors(receipt)

    def accept(self, receipt: ManuscriptReceipt) -> None:
        """Select a prepared artifact only after the caller's graph acknowledgement.

        All manuscript and mirror writes precede this step. Errors are fatal;
        graph/file reconciliation following an uncertain acknowledgement is owned
        by the workflow, not inferred from the presence of a prepared receipt.
        Identical acceptance is verified without replacing its durable selection.
        """
        self.read(receipt)
        prepared = ManuscriptReceipt.model_validate_json(self.files.read_bytes(receipt.prepared_path))
        if prepared != receipt:
            raise ValueError("Prepared manuscript receipt mismatch")
        previous = self.files.read_bytes(receipt.accepted_path) if self.files.exists(receipt.accepted_path) else None
        if previous is not None and ManuscriptReceipt.model_validate_json(previous) == receipt:
            self._sync_publication_directory()
            return
        try:
            self._write_verified(receipt.accepted_path, receipt.model_dump_json().encode("utf-8"))
        except Exception:
            # A failed post-rename fsync has uncertain durability. Retain the old
            # selection when storage permits; rollback failure also propagates.
            if previous is None:
                self.files.delete(receipt.accepted_path)
            elif self.files.read_bytes(receipt.accepted_path) != previous:
                self._write_verified(receipt.accepted_path, previous)
            raise

    def accepted(self, chapter_number: int) -> ManuscriptReceipt:
        if type(chapter_number) is not int or chapter_number <= 0:
            raise ValueError("Chapter number must be a positive integer")
        receipt = ManuscriptReceipt.model_validate_json(self.files.read_bytes(f"chapters/chapter_{chapter_number:03d}.accepted.json"))
        if receipt.chapter_number != chapter_number:
            raise ValueError("Accepted manuscript chapter mismatch")
        self.read(receipt)
        return receipt

    def recover(self, chapter_number: int) -> ManuscriptReceipt:
        """Rebuild interrupted compatibility mirrors from verified accepted prose."""
        receipt = self.accepted(chapter_number)
        self._require_owned_mirrors(chapter_number)
        self._write_mirrors(receipt)
        return receipt

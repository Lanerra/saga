"""Partition actual stderr writes without suppressing or rerouting CLI logging."""
import logging
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr
from io import StringIO
from unittest.mock import patch


class CLIStderr(StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.log_spans: list[tuple[int, int, logging.LogRecord]] = []

    @property
    def diagnostics(self) -> str:
        raw = self.getvalue()
        cursor = 0
        pieces = []
        for start, end, _ in self.log_spans:
            assert cursor <= start <= end <= len(raw)
            pieces.append(raw[cursor:start])
            cursor = end
        pieces.append(raw[cursor:])
        return "".join(pieces)


@contextmanager
def capture_cli_stderr(stream: CLIStderr) -> Iterator[None]:
    original = logging.StreamHandler.emit

    def record_emission(handler: logging.StreamHandler, record: logging.LogRecord) -> None:
        if handler.stream is not stream:
            original(handler, record)
            return
        expected = handler.format(record) + handler.terminator
        start = stream.tell()
        original(handler, record)
        end = stream.tell()
        assert stream.getvalue()[start:end] == expected
        stream.log_spans.append((start, end, record))

    with redirect_stderr(stream), patch.object(logging.StreamHandler, "emit", record_emission):
        yield

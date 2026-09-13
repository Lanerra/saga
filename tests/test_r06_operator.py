"""Operator regressions using synthetic configuration and local artifacts only."""

import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

import config
from config.docs_generator import generate_docs
from config.settings import EffectiveSettings, SagaSettings
from core.logging_config import setup_saga_logging

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.run_settings(NEO4J_PASSWORD="synthetic-do-not-publish-r06")
def test_schema_reference_does_not_publish_runtime_password(tmp_path: Path) -> None:
    output = tmp_path / "schema.md"
    generate_docs(output)
    assert "synthetic-do-not-publish-r06" not in output.read_text()


@pytest.mark.run_settings(LARGE_MODEL="synthetic-runtime-only-r06")
def test_schema_reference_documents_declared_not_active_defaults(tmp_path: Path) -> None:
    output = tmp_path / "schema.md"
    generate_docs(output)
    text = output.read_text()
    assert "synthetic-runtime-only-r06" not in text
    assert f'| LARGE_MODEL | str | "{SagaSettings.model_fields["LARGE_MODEL"].default}" |' in text


def test_schema_reference_has_four_columns_for_union_types(tmp_path: Path) -> None:
    output = tmp_path / "schema.md"
    generate_docs(output)
    rows = [line for line in output.read_text().splitlines() if line.startswith("|")]
    assert all(len(re.findall(r"(?<!\\)\|", row)) == 5 for row in rows)


@pytest.fixture
def isolated_logging() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.handlers = [logging.NullHandler()]
    try:
        yield root
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers = original_handlers
        root.setLevel(original_level)


@pytest.mark.run_settings(ENABLE_RICH_PROGRESS=False, SIMPLE_LOGGING_MODE=False, LOG_LEVEL_STR="DEBUG", LOG_FILE=None)
def test_logging_applies_requested_level_with_existing_handlers(isolated_logging: logging.Logger) -> None:
    isolated_logging.setLevel(logging.WARNING)
    setup_saga_logging()
    assert isolated_logging.level == logging.DEBUG


@pytest.mark.run_settings(ENABLE_RICH_PROGRESS=False, SIMPLE_LOGGING_MODE=False, LOG_FILE="synthetic.log")
def test_file_logging_keeps_plain_console_without_rich(isolated_logging: logging.Logger, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with config.bind_settings(config.snapshot_settings().model_copy(update={"BASE_OUTPUT_DIR": str(tmp_path)})):
        setup_saga_logging()
    isolated_logging.warning("synthetic-visible-r06")
    assert "synthetic-visible-r06" in (tmp_path / "synthetic.log").read_text()
    console = capsys.readouterr().err
    assert "synthetic-visible-r06" in console
    assert "[bold]" not in console


@pytest.mark.run_settings(ENABLE_RICH_PROGRESS=False, SIMPLE_LOGGING_MODE=True)
def test_simple_logging_is_plain_text(isolated_logging: logging.Logger, capsys: pytest.CaptureFixture[str]) -> None:
    setup_saga_logging()
    isolated_logging.warning("synthetic-plain-r06")
    console = capsys.readouterr().err
    assert "synthetic-plain-r06" in console
    assert "[bold]" not in console


def test_parse_help_explains_legacy_selector_refusal(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import main

    monkeypatch.setattr(sys, "argv", ["main.py", "parse", "--help"])
    with pytest.raises(SystemExit) as stopped:
        main.main()
    assert stopped.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "individual parser writes are blocked" in help_text.lower()
    assert "all parsers are run" not in help_text


def test_disposable_compose_is_loopback_bounded_and_explicitly_authenticated() -> None:
    service = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["neo4j-apoc"]
    assert all(port.startswith("127.0.0.1:") for port in service["ports"])
    environment = dict(item.split("=", 1) for item in service["environment"])
    assert environment["NEO4J_AUTH"].startswith("neo4j/${NEO4J_PASSWORD:?")
    assert environment["NEO4J_server_memory_heap_max__size"] == "512m"
    assert environment["NEO4J_server_memory_pagecache_size"] == "256m"
    assert environment["NEO4J_server_http_enabled"] == "false"
    assert environment["NEO4J_server_https_enabled"] == "false"
    assert service["restart"] == "no"
    assert "volumes" not in service


def test_all_visualizations_default_to_runtime_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import visualize_workflow

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["visualize_workflow.py", "--all"])
    visualize_workflow.main()
    assert "init_character_sheets" in (tmp_path / "output/workflows/workflow_full.md").read_text()
    assert not (tmp_path / "docs").exists()


def test_all_visualizations_reject_png_before_output_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import visualize_workflow

    output = tmp_path / "absent"
    monkeypatch.setattr(sys, "argv", ["visualize_workflow.py", "--all", "--format", "png", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as stopped:
        visualize_workflow.main()
    assert stopped.value.code == 1
    assert not output.exists()


def test_snapshot_preserves_nested_run_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextvars import Context

    from core.embedding_contract import embedding_identity, validate_embedding

    monkeypatch.setitem(vars(config), "EMBEDDING_MODEL", "legacy-default")
    enclosing = config.get_settings()
    first = EffectiveSettings(_env_file=None, EMBEDDING_MODEL="first", EXPECTED_EMBEDDING_DIM=2, NEO4J_VECTOR_DIMENSIONS=2)
    second = EffectiveSettings(_env_file=None, EMBEDDING_MODEL="second", EXPECTED_EMBEDDING_DIM=3, NEO4J_VECTOR_DIMENSIONS=3)
    with config.bind_settings(first):
        assert config.snapshot_settings() is first
        assert embedding_identity() == embedding_identity(first)
        assert validate_embedding([0.2, 0.4], model="first").shape == (2,)
        with config.bind_settings(second):
            assert config.snapshot_settings() is second
            assert embedding_identity() == embedding_identity(second)
            assert embedding_identity(first) != embedding_identity()
        assert config.snapshot_settings() is first
    assert config.get_settings() is enclosing
    assert Context().run(config.snapshot_settings).EMBEDDING_MODEL == "legacy-default"


def test_snapshot_rejects_expired_copied_scope() -> None:
    from contextvars import copy_context

    with config.bind_settings(EffectiveSettings(_env_file=None)):
        inherited = copy_context()
    with pytest.raises(RuntimeError, match="scope has expired"):
        inherited.run(config.snapshot_settings)

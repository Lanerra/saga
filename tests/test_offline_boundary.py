"""Behavioral checks for the repository's default unit boundary."""

import asyncio
import os
import socket
from pathlib import Path

import pytest
import pytest_asyncio

from tests.offline import OfflineBoundary


def test_unit_session_uses_synthetic_storage(pytestconfig: pytest.Config) -> None:
    assert Path(os.environ["BASE_OUTPUT_DIR"]).is_relative_to(Path.cwd())
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert Path.home().is_relative_to(Path.cwd())


@pytest_asyncio.fixture
async def running_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def test_async_fixture_and_case_share_function_loop(running_loop: asyncio.AbstractEventLoop, pytestconfig: pytest.Config) -> None:
    assert asyncio.get_running_loop() is running_loop
    assert pytestconfig.getini("asyncio_default_fixture_loop_scope") == "function"
    assert pytestconfig.getini("asyncio_default_test_loop_scope") == "function"


def test_real_timeout_plugin_is_required(pytestconfig: pytest.Config) -> None:
    assert pytestconfig.pluginmanager.hasplugin("timeout")
    assert "pytest-timeout==2.4.0" in pytestconfig.getini("required_plugins")


@pytest.mark.parametrize("event", ["socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"])
def test_audit_denies_and_records_even_caught_violations(event: str) -> None:
    boundary = OfflineBoundary()
    try:
        with pytest.raises(pytest.fail.Exception, match="SAGA_UNIT_OFFLINE"):
            boundary.audit(event, ())
        assert boundary.attempts == [{"operation": event, "test": "collection"}]
    finally:
        boundary.close()


def test_nested_boundary_blocks_dns_before_transport() -> None:
    boundary = OfflineBoundary()
    try:
        boundary.install()
        with pytest.raises(pytest.fail.Exception, match="SAGA_UNIT_OFFLINE: getaddrinfo"):
            socket.getaddrinfo("offline.invalid", 443)
        assert boundary.attempts == [{"operation": "getaddrinfo", "test": "collection"}]
    finally:
        boundary.close()


def test_tests_and_application_import_from_this_tree() -> None:
    import core
    import tests

    root = Path(__file__).resolve().parents[1]
    assert Path(tests.__file__).resolve() == root / "tests/__init__.py"
    assert Path(core.__file__).resolve() == root / "core/__init__.py"

"""Fail-closed Python test boundary; not a sandbox for hostile native code."""

import os
import socket
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NoReturn

import pytest


class OfflineBoundary:
    def __init__(self) -> None:
        self.active = True
        self.attempts: list[dict[str, str]] = []
        self.current_test = "collection"
        self.patches = pytest.MonkeyPatch()
        self.directory = TemporaryDirectory(prefix="saga-pytest-")
        self.scratch = Path(self.directory.name)

    def deny(self, operation: str) -> NoReturn:
        self.attempts.append({"operation": operation, "test": self.current_test})
        pytest.fail(f"SAGA_UNIT_OFFLINE: {operation}; provide a narrow deterministic fixture or mark a genuine integration dependency")

    def audit(self, event: str, arguments: tuple[Any, ...]) -> None:
        if not self.active:
            return
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.sendmsg", "subprocess.Popen", "os.system", "os.posix_spawn"}:
            self.deny(event)
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            name = Path(os.fsdecode(arguments[0])).name
            if name == ".env" or name.startswith(".env."):
                self.deny("credential-file-open")

    def install(self, *, change_directory: bool = True) -> None:
        import dotenv
        from pydantic_settings import DotEnvSettingsSource

        self.patches.setattr(dotenv, "load_dotenv", lambda *arguments, **keywords: False)
        self.patches.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})
        preserved = {name: value for name, value in os.environ.items() if name in {"PATH", "PYTHONPATH", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "TMPDIR"}}
        for name in list(os.environ):
            self.patches.delenv(name)
        for name, value in preserved.items():
            self.patches.setenv(name, value)
        for name, value in {
            "HOME": str(self.scratch / "home"),
            "XDG_CACHE_HOME": str(self.scratch / "cache"),
            "HF_HOME": str(self.scratch / "cache/huggingface"),
            "TIKTOKEN_CACHE_DIR": str(self.scratch / "cache/tiktoken"),
            "BASE_OUTPUT_DIR": str(self.scratch / "output"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "OPENAI_API_BASE": "http://127.0.0.1:9/v1",
            "OPENAI_API_KEY": "synthetic-unused",
            "EMBEDDING_API_BASE": "http://127.0.0.1:9",
            "EMBEDDING_API_KEY": "synthetic-unused",
            "NEO4J_URI": "bolt://127.0.0.1:9",
            "NEO4J_USER": "synthetic",
            "NEO4J_PASSWORD": "synthetic-unused",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
        }.items():
            self.patches.setenv(name, value)
        (self.scratch / "home").mkdir()
        if change_directory:
            self.patches.chdir(self.scratch)
        for target, names in (
            (socket, ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "create_connection")),
            (socket.socket, ("connect", "connect_ex", "sendto", "sendmsg")),
            (subprocess, ("Popen",)),
            (os, ("system", "posix_spawn", "posix_spawnp")),
        ):
            for name in names:
                original = getattr(target, name)

                def guarded(*arguments: Any, _name: str = name, _original: Any = original, **keywords: Any) -> Any:
                    if self.active:
                        self.deny(_name)
                    return _original(*arguments, **keywords)

                self.patches.setattr(target, name, guarded)
        sys.addaudithook(self.audit)

    def close(self) -> None:
        self.active = False
        self.patches.undo()
        self.directory.cleanup()


boundary_key = pytest.StashKey[OfflineBoundary]()


def pytest_plugin_registered(plugin: Any) -> None:
    """Historic registration runs before pytest imports child conftests."""
    if not isinstance(plugin, pytest.Config) or boundary_key in plugin.stash:
        return
    if "config" in sys.modules:
        raise pytest.UsageError("SAGA offline pytest requires a fresh interpreter: application configuration is already imported")
    boundary = OfflineBoundary()
    plugin.stash[boundary_key] = boundary
    plugin.add_cleanup(boundary.close)
    # Pytest resolves relative testpaths until parsing finishes. Both dotenv
    # sources and inherited configuration are isolated without changing cwd.
    boundary.install(change_directory=False)


def pytest_configure(config: pytest.Config) -> None:
    config.stash[boundary_key].patches.chdir(config.stash[boundary_key].scratch)

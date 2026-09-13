"""Run ordinary pytest startup regressions outside the unit subprocess guard.

Usage: python tests/offline_startup_probe.py --evidence /absolute/fresh/directory
The child receives only synthetic configuration. A late audit tripwire prevents
external I/O without installing or preloading the repository boundary.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

OBSERVER = """import atexit
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

context = json.loads(Path(__file__).with_name("context.json").read_text())
root = Path(context["tree"])
nested_path = str(root / "tests/test_langgraph/conftest.py")
receipt = {"nested_entries": [], "operations": {}, "tripwire": [], "imports": {}, "dotenv_sources": {}}
configure_started = False
initial_environment = dict(os.environ)
initial_directory = str(Path.cwd())
initial_functions = (socket.getaddrinfo, socket.socket.connect, socket.socket.sendto, subprocess.Popen)

class EscapedBoundary(BaseException):
    pass

def tripwire(event, arguments):
    operation = None
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto", "socket.sendmsg", "subprocess.Popen", "os.system", "os.posix_spawn"}:
        operation = event
    if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
        if Path(os.fsdecode(arguments[0])).name.startswith(".env"):
            operation = "credential-file-open"
    if operation:
        receipt["tripwire"].append(operation)
        raise EscapedBoundary(operation)

def profile(frame, event, argument):
    global configure_started
    if event == "call" and frame.f_globals.get("__name__") == "_pytest.config" and frame.f_code.co_name == "_do_configure":
        configure_started = True
    if frame.f_code.co_filename != nested_path or frame.f_code.co_name != "<module>":
        return
    if event == "call":
        receipt["nested_entries"].append({"cwd": str(Path.cwd()), "output": os.environ.get("BASE_OUTPUT_DIR"), "before_pytest_configure": not configure_started})
        # The real boundary must already own an earlier audit hook here.
        sys.addaudithook(tripwire)
        if context["mode"] == "import-error":
            raise RuntimeError("synthetic initial conftest failure")
        if context["mode"] == "attempts":
            import dotenv
            from pydantic_settings import BaseSettings, SettingsConfigDict

            class SyntheticSettings(BaseSettings):
                model_config = SettingsConfigDict(env_file=context["dotenv"])

            for name, source in {"python_dotenv": lambda: dotenv.load_dotenv(context["dotenv"]), "pydantic": lambda: SyntheticSettings().model_dump()}.items():
                try:
                    receipt["dotenv_sources"][name] = source()
                except BaseException as error:
                    receipt["dotenv_sources"][name] = str(error)
            # AF_UNIX exercises the same patched socket methods even when an
            # outer seccomp boundary denies AF_INET socket construction first.
            def socket_operation(datagram):
                with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM if datagram else socket.SOCK_STREAM) as connection:
                    if datagram:
                        connection.sendto(b"synthetic", str(Path(context["receipt"]).with_name("unused.socket")))
                    else:
                        connection.connect(str(Path(context["receipt"]).with_name("unused.socket")))

            actions = {
                "dotenv": lambda: Path(context["dotenv"]).read_text(),
                "dns": lambda: socket.getaddrinfo("offline.invalid", 443),
                "socket_connect": lambda: socket_operation(False),
                "socket_sendto": lambda: socket_operation(True),
                "subprocess": lambda: subprocess.run([sys.executable, "-c", "pass"], check=True),
            }
            for name, action in actions.items():
                try:
                    action()
                except BaseException as error:
                    receipt["operations"][name] = str(error).split(";")[0]
                else:
                    receipt["operations"][name] = "ESCAPED"
    elif event == "return":
        for name in ("config", "config.settings", "core.langgraph.state", "core.langgraph.graph_context", "core.service_context"):
            module = sys.modules.get(name)
            if module is not None:
                receipt["imports"][name] = module.__file__
        configuration = sys.modules.get("config")
        if configuration is not None:
            receipt["cached_output"] = configuration.BASE_OUTPUT_DIR
            receipt["cached_endpoint"] = configuration.OPENAI_API_BASE
            receipt["cached_database"] = configuration.NEO4J_URI
        sys.setprofile(None)

sys.setprofile(profile)

def finish():
    receipt["environment_restored"] = dict(os.environ) == initial_environment
    receipt["cwd_restored"] = str(Path.cwd()) == initial_directory
    receipt["functions_restored"] = initial_functions == (socket.getaddrinfo, socket.socket.connect, socket.socket.sendto, subprocess.Popen)
    receipt["sentinel_created"] = Path(context["sentinel"]).exists()
    Path(context["receipt"]).write_text(json.dumps(receipt, indent=2) + "\\n")

atexit.register(finish)
"""


def run_case(root: Path, directory: Path, target: list[str], mode: str, executable: str) -> dict[str, Any]:
    directory.mkdir(parents=True)
    temporary = directory / "temporary"
    temporary.mkdir()
    sentinel = directory / "inherited-output"
    dotenv = directory / ".env.synthetic"
    dotenv.write_text("OPENAI_API_BASE=http://synthetic-dotenv.invalid:9\n")
    (directory / "sitecustomize.py").write_text(OBSERVER)
    (directory / "context.json").write_text(
        json.dumps(
            {
                "tree": str(root),
                "mode": mode,
                "dotenv": str(dotenv),
                "sentinel": str(sentinel),
                "receipt": str(directory / "child.json"),
            }
        )
    )
    environment = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "HOME": str(directory),
        "TMPDIR": str(temporary),
        "PYTHONPATH": str(directory),
        "PYTHONDONTWRITEBYTECODE": "1",
        "BASE_OUTPUT_DIR": str(sentinel),
        "OPENAI_API_BASE": "http://synthetic-inherited.invalid:9",
        "NEO4J_URI": "bolt://synthetic-inherited.invalid:9",
    }
    command = [sys.executable, "-m", "pytest"] if executable == "module" else [str(Path(sys.executable).with_name("pytest"))]
    command += [*target, "-q", "-p", "no:cacheprovider"]
    if mode != "full":
        command.append("--collect-only")
    if mode == "parse-error":
        command.append("--synthetic-unknown-option")
    if mode == "repeat":
        arguments = [*target, "--collect-only", "-q", "-p", "no:cacheprovider"]
        command = [
            sys.executable,
            "-c",
            (
                "import pytest, sitecustomize; "
                f"codes = [int(pytest.main({arguments!r})), int(pytest.main({arguments!r}))]; "
                "sitecustomize.receipt['repeat_exit_codes'] = codes; "
                "raise SystemExit(0 if codes == [0, 4] else 1)"
            ),
        ]
    started = time.monotonic()
    result = subprocess.run(command, cwd=root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    elapsed = time.monotonic() - started
    (directory / "child.log").write_text(result.stdout)
    receipt = json.loads((directory / "child.json").read_text())
    entries = receipt["nested_entries"]
    expected_operations = (
        {
            "dotenv": "SAGA_UNIT_OFFLINE: credential-file-open",
            "dns": "SAGA_UNIT_OFFLINE: getaddrinfo",
            "socket_connect": "SAGA_UNIT_OFFLINE: connect",
            "socket_sendto": "SAGA_UNIT_OFFLINE: sendto",
            "subprocess": "SAGA_UNIT_OFFLINE: Popen",
        }
        if mode == "attempts"
        else {}
    )
    checks = {
        "exit": result.returncode == ({"attempts": 1, "parse-error": 4, "import-error": 4}.get(mode, 0)),
        "real_nested_conftest": len(entries) == 1,
        "initial_conftest_before_configure": len(entries) == 1 and entries[0]["before_pytest_configure"] is True,
        "protected_before_nested_import": len(entries) == 1 and entries[0]["output"] != str(sentinel),
        "no_inherited_output": receipt["sentinel_created"] is False,
        "safe_cached_configuration": (
            "cached_output" not in receipt
            if mode == "import-error"
            else receipt.get("cached_output") != str(sentinel) and receipt.get("cached_endpoint") == "http://127.0.0.1:9/v1" and receipt.get("cached_database") == "bolt://127.0.0.1:9"
        ),
        "real_import_provenance": receipt["imports"]
        == {
            name: str(root / (name.replace(".", "/") + ("/__init__.py" if name == "config" else ".py")))
            for name in (
                "config",
                "config.settings",
                "core.langgraph.state",
                "core.langgraph.graph_context",
                "core.service_context",
            )
        },
        "operations_denied_by_repository": receipt["operations"] == expected_operations,
        "dotenv_sources_disabled": receipt["dotenv_sources"] == ({"python_dotenv": False, "pydantic": {}} if mode == "attempts" else {}),
        "no_tripwire_escape": receipt["tripwire"] == [],
        "environment_restored": receipt["environment_restored"],
        "cwd_restored": receipt["cwd_restored"],
        "functions_restored": receipt["functions_restored"],
    }
    if mode == "import-error":
        checks["real_import_provenance"] = receipt["imports"] == {}
        checks["expected_import_error"] = "synthetic initial conftest failure" in result.stdout
    if mode == "repeat":
        checks["cached_configuration_rejected"] = receipt["repeat_exit_codes"] == [0, 4] and "application configuration is already imported" in result.stdout
    return {
        "command": command,
        "cwd": str(root),
        "environment": environment,
        "exit_code": result.returncode,
        "elapsed_seconds": elapsed,
        "log": str(directory / "child.log"),
        "receipt": str(directory / "child.json"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--tree", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--startup-only", action="store_true", help="Run the startup/collection boundary matrix without the two full-suite executions")
    options = parser.parse_args()
    root = options.tree.resolve()
    evidence = options.evidence.resolve()
    if evidence.exists():
        parser.error("Evidence directory must be fresh")
    results = []
    for executable in ("module", "console"):
        for name, target in (("tests", ["tests"]), ("default", []), ("nested", ["tests/test_langgraph/test_state.py"])):
            for mode in ("observe", "attempts"):
                result = run_case(root, evidence / f"{executable}-{name}-{mode}", target, mode, executable)
                results.append(result)
                (evidence / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps({"case": f"{executable}-{name}-{mode}", "passed": result["passed"], "checks": result["checks"]}), flush=True)
    for mode in ("parse-error", "import-error", "repeat"):
        result = run_case(root, evidence / mode, ["tests/test_langgraph/test_state.py"], mode, "module")
        results.append(result)
        print(json.dumps({"case": mode, "passed": result["passed"], "checks": result["checks"]}), flush=True)
    if not options.startup_only:
        for executable, target in (("module", ["tests"]), ("console", [])):
            result = run_case(root, evidence / f"{executable}-full", target, "full", executable)
            results.append(result)
            print(json.dumps({"case": f"{executable}-full", "passed": result["passed"], "checks": result["checks"]}), flush=True)
    (evidence / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

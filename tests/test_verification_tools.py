"""Retained verification helpers use their own checkout from any working directory."""

import inspect
import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["verify_split.py", "verify_subgraph.py"])
@pytest.mark.parametrize("unrelated_directory", [False, True])
def test_verification_helpers_use_local_source(
    script: str, unrelated_directory: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path if unrelated_directory else repository)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    namespace = runpy.run_path(str(repository / script), run_name="__main__")
    assert Path(sys.path[0]).resolve() == repository
    for value in namespace.values():
        if inspect.isfunction(value) and value.__module__.startswith("core."):
            assert Path(inspect.getfile(value)).resolve().is_relative_to(repository)
    if script == "verify_subgraph.py":
        assert set(namespace["graph"].nodes) == {"__start__", "extract_from_scenes", "consolidate"}

"""Check the supported runtime declarations without importing application services."""

import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RuntimeDependencies(unittest.TestCase):
    def test_spacy_import_and_blank_pipeline(self) -> None:
        import spacy

        pipeline = spacy.blank("en")
        self.assertEqual([token.text for token in pipeline("Synthetic runtime check.")], ["Synthetic", "runtime", "check", "."])

    def test_development_tools_are_exactly_pinned(self) -> None:
        declarations = {
            line.split("==")[0].lower().replace("_", "-"): line
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line and not line.startswith("#")
        }
        for name in ("pip", "pytest", "pytest-asyncio", "pytest-cov", "pytest-timeout", "ruff", "mypy", "types-aiofiles", "types-pyyaml"):
            with self.subTest(package=name):
                self.assertIn(name, declarations)
                self.assertRegex(declarations[name].lower(), rf"^{re.escape(name)}==[0-9][0-9.a-z]*$")

    def test_python_matches_tool_targets(self) -> None:
        version = (ROOT / ".python-version").read_text().strip()
        self.assertRegex(version, r"^3\.12\.\d+$")
        configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(configuration["tool"]["ruff"]["target-version"], "py312")
        self.assertEqual(configuration["tool"]["mypy"]["python_version"], "3.12")

    def test_lock_contains_hashed_direct_and_transitive_pins(self) -> None:
        lock = (ROOT / "requirements.lock").read_text()
        entries = {}
        for record in lock.replace("\\\n", "").splitlines():
            if not record or record.startswith("#") or record.startswith(" "):
                continue
            match = re.fullmatch(r"([a-z0-9-]+)==([^\s]+)(\s+--hash=sha256:[a-f0-9]{64})+", record)
            self.assertIsNotNone(match, record)
            assert match is not None
            entries[match[1]] = match[2]
        for line in (ROOT / "requirements.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                name, version = line.split("==")
                self.assertEqual(entries[name.lower().replace("_", "-")], version)
        for name in ("pydantic-core", "httpcore", "langgraph-checkpoint", "thinc"):
            self.assertIn(name, entries)

    def test_pytest_requires_configured_plugins(self) -> None:
        configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            configuration["tool"]["pytest"]["ini_options"]["required_plugins"],
            ["pytest-asyncio==1.0.0", "pytest-cov==5.0.0", "pytest-timeout==2.4.0"],
        )

# Test boundaries

With the locked Python 3.12 environment installed, run `python -m pytest tests`.
The default selection is offline. Both `tests/core/langgraph/` and
`tests/test_langgraph/` remain part of that selection.

Root `conftest.py` loads `tests.offline`. Its historic `pytest_plugin_registered`
hook installs the boundary on replay of Config registration, before any nested
conftest/application import, including `tests/test_langgraph/conftest.py`.
It replaces inherited configuration with synthetic home, output, and empty model
caches and disables both python-dotenv and Pydantic's dotenv source. The launch
directory remains unchanged only while pytest resolves relative test targets;
`pytest_configure` moves execution to private temporary storage. Config-owned
cleanup restores environment, cwd, and patched functions even on parsing or
initial-conftest failures, before configuration completes.
Python DNS/TCP/UDP, subprocess launch, and `.env` access fail closed. Even an
attempt swallowed by a test or `asyncio.gather(return_exceptions=True)` makes
the session exit nonzero. The terminal summary identifies rejected operations
and their test IDs. This is a trusted-Python test boundary, not an OS sandbox
for hostile native extensions or third-party plugins loaded before conftest.
CI should additionally deny network at the OS/container boundary.

Ordinary repository-cwd `pytest`, `python -m pytest tests`, and direct nested
selection require no wrapper or preloaded conftest. Use a fresh interpreter:
an application configuration already cached before pytest (including a second
`pytest.main` after application collection) is rejected with a usage error,
rather than reusing inherited configuration or a deleted prior scratch path.
Pytest invocations that intentionally suppress the root conftest are unsupported.

Run the process-level startup regression matrix separately from the unit guard:

    python tests/offline_startup_probe.py --evidence /absolute/fresh/directory

It exercises module and console entrypoints, explicit/default/nested targets,
preconfigure configuration and external-I/O sentinels, and failed/repeated-session
cleanup. Every child uses synthetic configuration and the real nested conftest.
An independent late audit tripwire denies any attempt escaping the repository
guard; such an escape fails the matrix. It does not pre-seed the child's offline
environment or preload `tests.conftest`. Logs, exact child environments, exit
codes, import provenance, and strict checks are written to the fresh directory.

Unit language assets are deterministic:

- tiktoken's real encoding engine operates on a synthetic byte vocabulary;
  these tests do not establish real model token-budget accuracy.
- spaCy model loading defaults to a deterministic missing-model error. Tests
  of successful loading provide a narrow `spacy.blank("en")` pipeline, while
  service extraction tests supply only their expected document/token interface.
- Database/embedding fixtures are opt-in at their relevant consumers. They do
  not globally replace `Neo4jManager`, `ContentManager`, or real SQLite behavior.

Tests must mark actual external-resource obligations with `pytest.mark.integration`.
A filename containing `integration` is not sufficient: parser and graph-composition
cases with deterministic providers are still meaningful offline units.

Inventory of model/service integration exclusions:

- `tests/test_spacy_service.py::test_installed_statistical_model_loads` needs the
  separately provisioned `en_core_web_lg` statistical model. It is not downloaded
  by pytest. It supplements, rather than replaces, the offline loading/idempotence
  tests.

Inspect that inventory with:

    python -m pytest tests --run-integration -m integration --collect-only

Run explicitly provisioned integration dependencies with:

    python -m pytest tests --run-integration -m integration

The flag includes marked cases and permits their Python external I/O during
setup/call/teardown; it does not restore user credentials or configured services.
Collection is always offline. Integration fixtures must explicitly supply only
synthetic/disposable endpoints and locally provisioned assets. Unmarked cases
remain offline even when the flag is present. This opt-in is not authorization
to connect to an existing user graph or use real provider credentials.

The checked-in pytest configuration requires actual pytest-timeout and
pytest-asyncio plugins; fixture and test event loops both use function scope.
Unknown markers remain errors. No broad skip, xfail, or failure deselection is
used. Statistical-model/Neo4j/APOC acceptance still requires separately authorized
real dependencies; a green unit suite is not engine or release acceptance.

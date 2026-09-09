"""Load the offline policy before any nested test conftest imports."""

pytest_plugins = ["tests.offline"]

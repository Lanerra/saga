from __future__ import annotations

from pathlib import Path

from config.docs_generator import _format_default, generate_docs
from config.settings import SagaSettings


class TestFormatDefault:
    def test_string_wrapped_in_quotes(self) -> None:
        assert _format_default("hello") == '"hello"'

    def test_integer(self) -> None:
        assert _format_default(42) == "42"

    def test_float(self) -> None:
        assert _format_default(3.14) == "3.14"

    def test_boolean(self) -> None:
        assert _format_default(True) == "True"

    def test_list(self) -> None:
        assert _format_default([1, 2]) == "[1, 2]"

    def test_tuple_converted_to_list(self) -> None:
        assert _format_default((1, 2)) == "[1, 2]"

    def test_none_uses_repr(self) -> None:
        assert _format_default(None) == "None"


class TestGenerateDocs:
    def test_output_file_is_created(self, tmp_path: Path) -> None:
        output = tmp_path / "schema.md"
        generate_docs(output_path=output)
        assert output.exists()

    def test_output_starts_with_heading(self, tmp_path: Path) -> None:
        output = tmp_path / "schema.md"
        generate_docs(output_path=output)
        content = output.read_text(encoding="utf-8")
        assert content.startswith("# Generated Configuration Schema")

    def test_output_contains_table_header(self, tmp_path: Path) -> None:
        output = tmp_path / "schema.md"
        generate_docs(output_path=output)
        content = output.read_text(encoding="utf-8")
        assert "| Setting | Type | Default | Description |" in content

    def test_output_contains_at_least_one_field_row(self, tmp_path: Path) -> None:
        output = tmp_path / "schema.md"
        generate_docs(output_path=output)
        lines = output.read_text(encoding="utf-8").splitlines()
        separator_index = lines.index("|---------|------|---------|-------------|")
        field_rows = [line for line in lines[separator_index + 1 :] if line.startswith("|")]
        assert len(field_rows) == len(SagaSettings.model_fields)

    def test_parent_directories_created_automatically(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "deeply" / "schema.md"
        generate_docs(output_path=output)
        assert output.exists()

    def test_pipe_characters_in_descriptions_are_escaped(self, tmp_path: Path) -> None:
        output = tmp_path / "schema.md"
        generate_docs(output_path=output)
        lines = output.read_text(encoding="utf-8").splitlines()
        separator_index = lines.index("|---------|------|---------|-------------|")
        field_rows = lines[separator_index + 1 :]
        for row in field_rows:
            if not row.startswith("|"):
                continue
            cells = row.split(" | ")
            description_cell = cells[-1]
            unescaped_pipes = description_cell.replace("\\|", "").count("|")
            assert unescaped_pipes <= 1

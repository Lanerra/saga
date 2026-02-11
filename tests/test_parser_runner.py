from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.parser_runner import ParserRunner


def _make_runner() -> ParserRunner:
    with patch("core.langgraph.content_manager.ContentManager") as FakeContentManager:
        instance = FakeContentManager.return_value
        instance._get_content_path.return_value = Path("/fake/path")
        runner = ParserRunner(Path("/fake/project"))
    return runner


def _make_fake_parser(success: bool = True, message: str = "Parsed successfully") -> MagicMock:
    fake_parser = MagicMock()
    fake_parser.parse_and_persist = AsyncMock(return_value=(success, message))
    return fake_parser


class TestParserRunnerInputValidation:
    def test_unknown_parser_name_raises_value_error(self) -> None:
        runner = _make_runner()
        with pytest.raises(ValueError, match="Unknown parser: nonexistent"):
            import asyncio

            asyncio.run(runner.run_parser("nonexistent"))

    def test_unsupported_parser_class_raises_value_error(self) -> None:
        runner = _make_runner()

        class BogusParser:
            pass

        with pytest.raises(ValueError, match="Unsupported parser class"):
            runner._create_parser_instance(BogusParser)


class TestRunParser:
    @patch("core.parser_runner.CharacterSheetParser")
    async def test_run_parser_character_sheets_calls_correct_class(self, FakeCharacterSheetParser: MagicMock) -> None:
        fake_parser = _make_fake_parser()
        FakeCharacterSheetParser.return_value = fake_parser
        runner = _make_runner()

        await runner.run_parser("character_sheets")

        fake_parser.parse_and_persist.assert_awaited_once()

    @patch("core.parser_runner.CharacterSheetParser")
    async def test_parser_success_returns_true_and_message(self, FakeCharacterSheetParser: MagicMock) -> None:
        fake_parser = _make_fake_parser(success=True, message="All characters parsed")
        FakeCharacterSheetParser.return_value = fake_parser
        runner = _make_runner()

        result = await runner.run_parser("character_sheets")

        assert result == (True, "All characters parsed")

    @patch("core.parser_runner.CharacterSheetParser")
    async def test_parser_exception_returns_false_and_error_message(self, FakeCharacterSheetParser: MagicMock) -> None:
        failing_parser = MagicMock()
        failing_parser.parse_and_persist = AsyncMock(side_effect=RuntimeError("Neo4j connection lost"))
        FakeCharacterSheetParser.return_value = failing_parser
        runner = _make_runner()

        result = await runner.run_parser("character_sheets")

        assert result == (False, "Neo4j connection lost")


class TestRunAllParsers:
    @patch("core.parser_runner.ChapterOutlineParser")
    @patch("core.parser_runner.ActOutlineParser")
    @patch("core.parser_runner.GlobalOutlineParser")
    @patch("core.parser_runner.CharacterSheetParser")
    async def test_runs_all_four_parsers_in_correct_order(
        self,
        FakeCharacterSheetParser: MagicMock,
        FakeGlobalOutlineParser: MagicMock,
        FakeActOutlineParser: MagicMock,
        FakeChapterOutlineParser: MagicMock,
    ) -> None:
        call_order: list[str] = []

        for name, fake_class in [
            ("character_sheets", FakeCharacterSheetParser),
            ("global_outline", FakeGlobalOutlineParser),
            ("act_outlines", FakeActOutlineParser),
            ("chapter_outlines", FakeChapterOutlineParser),
        ]:
            fake_instance = MagicMock()
            captured_name = name

            def make_side_effect(parser_name: str) -> tuple[bool, str]:
                call_order.append(parser_name)
                return (True, f"{parser_name} done")

            fake_instance.parse_and_persist = AsyncMock(side_effect=lambda n=captured_name: make_side_effect(n))
            fake_class.return_value = fake_instance

        runner = _make_runner()
        await runner.run_all_parsers()

        assert call_order == [
            "character_sheets",
            "global_outline",
            "act_outlines",
            "chapter_outlines",
        ]

    @patch("core.parser_runner.ChapterOutlineParser")
    @patch("core.parser_runner.ActOutlineParser")
    @patch("core.parser_runner.GlobalOutlineParser")
    @patch("core.parser_runner.CharacterSheetParser")
    async def test_returns_results_for_all_parsers(
        self,
        FakeCharacterSheetParser: MagicMock,
        FakeGlobalOutlineParser: MagicMock,
        FakeActOutlineParser: MagicMock,
        FakeChapterOutlineParser: MagicMock,
    ) -> None:
        for fake_class, message in [
            (FakeCharacterSheetParser, "characters ok"),
            (FakeGlobalOutlineParser, "global ok"),
            (FakeActOutlineParser, "acts ok"),
            (FakeChapterOutlineParser, "chapters ok"),
        ]:
            fake_instance = MagicMock()
            fake_instance.parse_and_persist = AsyncMock(return_value=(True, message))
            fake_class.return_value = fake_instance

        runner = _make_runner()
        results = await runner.run_all_parsers()

        assert results == {
            "character_sheets": (True, "characters ok"),
            "global_outline": (True, "global ok"),
            "act_outlines": (True, "acts ok"),
            "chapter_outlines": (True, "chapters ok"),
        }

    @patch("core.parser_runner.ChapterOutlineParser")
    @patch("core.parser_runner.ActOutlineParser")
    @patch("core.parser_runner.GlobalOutlineParser")
    @patch("core.parser_runner.CharacterSheetParser")
    async def test_continues_running_remaining_parsers_after_one_fails(
        self,
        FakeCharacterSheetParser: MagicMock,
        FakeGlobalOutlineParser: MagicMock,
        FakeActOutlineParser: MagicMock,
        FakeChapterOutlineParser: MagicMock,
    ) -> None:
        for fake_class, return_value in [
            (FakeCharacterSheetParser, (True, "characters ok")),
            (FakeGlobalOutlineParser, (False, "global outline missing")),
            (FakeActOutlineParser, (True, "acts ok")),
            (FakeChapterOutlineParser, (True, "chapters ok")),
        ]:
            fake_instance = MagicMock()
            fake_instance.parse_and_persist = AsyncMock(return_value=return_value)
            fake_class.return_value = fake_instance

        runner = _make_runner()
        results = await runner.run_all_parsers()

        assert len(results) == 4
        assert results["character_sheets"] == (True, "characters ok")
        assert results["global_outline"] == (False, "global outline missing")
        assert results["act_outlines"] == (True, "acts ok")
        assert results["chapter_outlines"] == (True, "chapters ok")

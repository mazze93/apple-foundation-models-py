"""Tests for the `afm` command-line interface (applefoundationmodels.cli)."""

import io
import json

import pytest

from applefoundationmodels import cli

OVERSIZED = "The quick brown fox jumps over the lazy dog. " * 2000


def run(argv, stdin=""):
    """Invoke cli.main with in-memory streams; return (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


class TestComposePrompt:
    def test_words_only(self):
        assert cli.compose_prompt(["say", "hi"], None) == "say hi"

    def test_piped_only(self):
        assert cli.compose_prompt([], "  body\n") == "body"

    def test_words_then_piped(self):
        assert cli.compose_prompt(["summarize"], "body") == "summarize\n\nbody"


class TestFitToBudget:
    # One "token" per character keeps the arithmetic obvious
    count = staticmethod(len)

    def test_returns_input_unchanged_when_it_fits(self):
        assert cli.fit_to_budget(["hi"], "abc", 100, self.count) == "abc"

    def test_keeps_longest_head_that_fits(self):
        kept = cli.fit_to_budget(["hi"], "x" * 50, 20, self.count)
        assert self.count(cli.compose_prompt(["hi"], kept)) <= 20
        assert self.count(cli.compose_prompt(["hi"], "x" * (len(kept) + 1))) > 20

    def test_raises_when_instruction_alone_is_too_big(self):
        with pytest.raises(cli.InputTooLargeError):
            cli.fit_to_budget(["word"] * 20, "body", 5, self.count)


class TestArgumentHandling:
    def test_temperature_out_of_range_is_usage_error(self):
        code, _, err = run(["-t", "3", "hi"])
        assert code == cli.EXIT_USAGE
        assert "temperature" in err

    def test_no_prompt_is_usage_error(self, check_availability):
        code, _, err = run([])
        assert code == cli.EXIT_USAGE
        assert "no prompt" in err

    def test_unreadable_schema_is_usage_error(self, check_availability, tmp_path):
        code, _, err = run(["--schema", str(tmp_path / "missing.json"), "hi"])
        assert code == cli.EXIT_USAGE
        assert "schema" in err

    def test_double_dash_sends_command_word_as_prompt(self, check_availability):
        code, out, _ = run(["--no-stream", "-m", "40", "--", "status"])
        assert code == cli.EXIT_OK
        assert "availability:" not in out


class TestCommands:
    def test_status(self, check_availability):
        code, out, _ = run(["status"])
        assert code == cli.EXIT_OK
        assert "AVAILABLE" in out
        assert "context window:" in out

    def test_one_shot_streams_text(self, check_availability):
        code, out, _ = run(["-t", "0", "-m", "40", "What is 2 + 2? Answer briefly."])
        assert code == cli.EXIT_OK
        assert "4" in out
        assert out.endswith("\n")

    def test_piped_input_is_used(self, check_availability):
        code, out, _ = run(
            ["--no-stream", "-t", "0", "-m", "40", "Repeat the code word only."],
            stdin="The code word is PELICAN.",
        )
        assert code == cli.EXIT_OK
        assert "PELICAN" in out.upper()

    def test_schema_prints_json(self, check_availability, tmp_path):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        path = tmp_path / "schema.json"
        path.write_text(json.dumps(schema))
        code, out, _ = run(["--schema", str(path), "Alice is 28 years old."])
        assert code == cli.EXIT_OK
        data = json.loads(out)
        assert data["age"] == 28

    def test_oversized_input_fails_with_clear_message(self, check_availability):
        code, out, err = run(["Summarize."], stdin=OVERSIZED)
        assert code == cli.EXIT_USAGE
        assert out == ""
        assert "context window" in err

    def test_truncate_fits_oversized_input(self, check_availability):
        code, out, err = run(
            ["--truncate", "--no-stream", "-m", "60", "Summarize in one sentence."],
            stdin=OVERSIZED,
        )
        assert code == cli.EXIT_OK, err
        assert "truncated" in err
        assert out.strip()

    def test_chat_keeps_history_and_handles_commands(self, check_availability):
        script = "\n".join(
            [
                "My code word is WALRUS. Reply OK.",
                "What is my code word?",
                "/bogus",
                "/reset",
                "/exit",
            ]
        )
        code, out, err = run(["chat", "-t", "0", "-m", "40"], stdin=script)
        assert code == cli.EXIT_OK
        assert "WALRUS" in out.upper()
        assert "unknown command" in err
        assert "history cleared" in err

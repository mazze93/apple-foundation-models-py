"""
`afm` — command-line front end for the on-device Apple Foundation Model.

    afm "prompt"                       one-shot, streams to stdout
    cat notes.txt | afm "summarize"    piped input is appended to the prompt
    afm -s "You are terse." "prompt"   system instructions
    afm --schema s.json "extract..."   structured output, printed as JSON
    afm chat                           interactive session that keeps history
    afm status                         availability, version, context size

`status` and `chat` are only recognised as the first argument; send them as a
literal prompt with `afm -- status`.

Exit codes: 0 ok, 1 unavailable or generation error, 2 usage error or input
too large for the context window, 130 interrupted.
"""

import argparse
import json
import sys
from typing import Callable, List, Optional, TextIO

from . import __version__
from .constants import DEFAULT_MAX_TOKENS
from .exceptions import (
    ContextWindowExceededError,
    FoundationModelsError,
    NotAvailableError,
)
from .session import Session
from .types import Availability

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

COMMANDS = ("status", "chat")


class InputTooLargeError(Exception):
    """Prompt does not fit the context window and truncation wasn't requested."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afm",
        description="Run Apple's on-device Foundation Model from the shell.",
        epilog="Commands: 'afm status', 'afm chat'. Use 'afm -- status' to send "
        "the word as a prompt.",
    )
    parser.add_argument("prompt", nargs="*", help="prompt text (joined with spaces)")
    parser.add_argument("-s", "--system", help="system instructions for the session")
    parser.add_argument(
        "-t", "--temperature", type=float, help="sampling temperature, 0.0-2.0"
    )
    parser.add_argument(
        "-m",
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"maximum response tokens (default {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--schema",
        metavar="FILE",
        help="JSON Schema file; prints structured output as JSON",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="cut piped input to fit the context window instead of failing",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="print the full response at once"
    )
    return parser


def compose_prompt(prompt_words: List[str], piped: Optional[str]) -> str:
    """Join CLI words and piped stdin into one prompt."""
    prompt = " ".join(prompt_words).strip()
    piped = (piped or "").strip()
    if prompt and piped:
        return f"{prompt}\n\n{piped}"
    return prompt or piped


def fit_to_budget(
    prompt_words: List[str],
    piped: str,
    budget: int,
    count: Callable[[str], int],
) -> str:
    """Return the longest head of `piped` that keeps the prompt within `budget`.

    Binary search on character length, so it costs O(log n) token counts.
    Raises InputTooLargeError if even the instruction words alone don't fit.
    """
    if count(compose_prompt(prompt_words, piped)) <= budget:
        return piped
    if count(compose_prompt(prompt_words, "")) > budget:
        raise InputTooLargeError("the prompt alone exceeds the context window")
    lo, hi = 0, len(piped)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(compose_prompt(prompt_words, piped[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return piped[:lo]


def _token_counter() -> Optional[Callable[[str], int]]:
    """Session.token_count if this OS supports it (macOS 26.4+), else None."""
    try:
        Session.token_count("probe")
    except NotAvailableError:
        return None
    return Session.token_count


# The framework spends tokens that tokenCount(for:) doesn't report (prompt
# template / preamble): measured ~70-85 on macOS 27, so keep a margin above it.
FRAMEWORK_OVERHEAD_TOKENS = 128


def _input_budget(system: Optional[str], max_tokens: int, count) -> int:
    # max_tokens isn't reserved by the framework up front, but without reserving
    # it here a long response can overflow the window mid-generation.
    reserved = max_tokens + FRAMEWORK_OVERHEAD_TOKENS
    if system:
        reserved += count(system)
    return Session.context_size() - reserved


def check_fits(
    prompt_words: List[str],
    piped: Optional[str],
    system: Optional[str],
    max_tokens: int,
    truncate: bool,
    err: TextIO,
) -> str:
    """Pre-flight the prompt against the context window; return the final prompt."""
    count = _token_counter()
    prompt = compose_prompt(prompt_words, piped)
    if count is None:
        return prompt  # older OS: rely on ContextWindowExceededError instead

    budget = _input_budget(system, max_tokens, count)
    used = count(prompt)
    if used <= budget:
        return prompt
    if not (truncate and piped):
        raise InputTooLargeError(
            f"input is {used} tokens but only {budget} fit "
            f"(context window {Session.context_size()}; {max_tokens} reserved "
            f"for the response, {FRAMEWORK_OVERHEAD_TOKENS} for framework overhead). "
            "Shorten it, lower --max-tokens, or pass "
            "--truncate to cut piped input."
        )
    kept = fit_to_budget(prompt_words, piped or "", budget, count)
    original = len((piped or "").strip())
    err.write(
        f"afm: input truncated to {len(kept)} of {original} characters "
        "to fit the context window\n"
    )
    return compose_prompt(prompt_words, kept)


def _write_stream(session: Session, prompt: str, args, out: TextIO) -> None:
    last = ""
    for chunk in session.generate(
        prompt,
        stream=True,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ):
        if chunk.content:
            out.write(chunk.content)
            out.flush()
            last = chunk.content
    if not last.endswith("\n"):
        out.write("\n")


def run_prompt(args, piped: Optional[str], out: TextIO, err: TextIO) -> int:
    if not args.prompt and not (piped or "").strip():
        err.write("afm: no prompt given (pass text or pipe input; see afm -h)\n")
        return EXIT_USAGE

    schema = None
    if args.schema:
        try:
            with open(args.schema, encoding="utf-8") as f:
                schema = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            err.write(f"afm: can't read schema {args.schema}: {e}\n")
            return EXIT_USAGE

    try:
        prompt = check_fits(
            args.prompt, piped, args.system, args.max_tokens, args.truncate, err
        )
    except InputTooLargeError as e:
        err.write(f"afm: {e}\n")
        return EXIT_USAGE

    with Session(instructions=args.system) as session:
        if schema is not None:
            response = session.generate(
                prompt,
                schema=schema,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            out.write(json.dumps(response.parsed, indent=2, ensure_ascii=False))
            out.write("\n")
        elif args.no_stream:
            response = session.generate(
                prompt, temperature=args.temperature, max_tokens=args.max_tokens
            )
            out.write(response.text.rstrip("\n") + "\n")
        else:
            _write_stream(session, prompt, args, out)
    return EXIT_OK


def run_status(out: TextIO) -> int:
    status = Session.check_availability()
    out.write(f"availability:   {status.name}\n")
    reason = Session.get_availability_reason()
    if reason:
        out.write(f"reason:         {reason}\n")
    out.write(f"package:        {__version__}\n")
    out.write(f"native library: {Session.get_version()}\n")
    if status == Availability.AVAILABLE:
        out.write(f"context window: {Session.context_size()} tokens\n")
        counting = "yes" if _token_counter() else "no (needs macOS 26.4+)"
        out.write(f"token counting: {counting}\n")
    return EXIT_OK if status == Availability.AVAILABLE else EXIT_ERROR


CHAT_HELP = "Commands: /reset clears history, /exit quits (or Ctrl-D)."


def run_chat(args, inp: TextIO, out: TextIO, err: TextIO) -> int:
    def new_session() -> Session:
        return Session(instructions=args.system)

    interactive = inp.isatty()
    session = new_session()
    if interactive:
        err.write(f"afm chat — on-device model. {CHAT_HELP}\n")
    try:
        while True:
            if interactive:
                err.write("> ")
                err.flush()
            line = inp.readline()
            if not line:  # EOF
                if interactive:
                    err.write("\n")
                return EXIT_OK
            prompt = line.strip()
            if not prompt:
                continue
            if prompt in ("/exit", "/quit"):
                return EXIT_OK
            if prompt == "/reset":
                session.close()
                session = new_session()
                err.write("(history cleared)\n")
                continue
            if prompt.startswith("/"):
                err.write(f"unknown command {prompt}. {CHAT_HELP}\n")
                continue

            try:
                _write_stream(session, prompt, args, out)
            except ContextWindowExceededError:
                # History filled the window: start fresh and retry this turn once
                session.close()
                session = new_session()
                err.write("(context window full — history cleared, retrying)\n")
                try:
                    _write_stream(session, prompt, args, out)
                except ContextWindowExceededError:
                    err.write("afm: this message alone is too large for the model\n")
            except KeyboardInterrupt:
                out.write("\n")
                err.write("(interrupted)\n")
    finally:
        session.close()


def _read_piped(inp: TextIO) -> Optional[str]:
    if inp.isatty():
        return None
    return inp.read()


def main(
    argv: Optional[List[str]] = None,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    inp = stdin or sys.stdin
    out = stdout or sys.stdout
    err = stderr or sys.stderr

    command = argv[0] if argv and argv[0] in COMMANDS else None
    if command:
        argv = argv[1:]
    args = build_parser().parse_args(argv)

    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        err.write("afm: --temperature must be between 0.0 and 2.0\n")
        return EXIT_USAGE

    try:
        if command == "status":
            return run_status(out)

        if Session.check_availability() != Availability.AVAILABLE:
            reason = Session.get_availability_reason() or "unknown reason"
            err.write(f"afm: Apple Intelligence is not available: {reason}\n")
            return EXIT_ERROR

        if command == "chat":
            return run_chat(args, inp, out, err)
        return run_prompt(args, _read_piped(inp), out, err)
    except KeyboardInterrupt:
        err.write("\n")
        return EXIT_INTERRUPTED
    except ContextWindowExceededError:
        err.write(
            "afm: input exceeds the model's context window "
            f"({Session.context_size()} tokens). Shorten it or lower --max-tokens.\n"
        )
        return EXIT_USAGE
    except FoundationModelsError as e:
        err.write(f"afm: {type(e).__name__}: {e}\n")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

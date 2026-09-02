# Test Suite

## Overview

The test suite for `apple-foundation-models` is organized into unit tests and one integration test module. All tests exercise the real Swift/FoundationModels FFI layer — there is no mock/stub layer, so most tests still require Apple Intelligence to be available and enabled on the machine running them.

## Test Files

### `conftest.py`
Pytest configuration and shared fixtures:
- `check_availability`: session-scoped fixture that skips a test if Apple Intelligence isn't available (`Session.check_availability() != Availability.AVAILABLE`)
- `session`: yields a `Session` instance (no instructions, to avoid transcript pollution), closes it on teardown
- `async_session`: yields an `AsyncSession` instance the same way, `await`s `aclose()` on teardown
- `assert_valid_response` / `assert_valid_chunks`: helpers for validating `GenerationResponse` / `StreamChunk` results

### `test_session.py` (Unit Tests)
Tests for `applefoundationmodels.Session`:
- **Text generation**: basic generation, temperature variation, `max_tokens` limiting
- **Streaming**: sync streaming, temperature variation
- **History**: `get_history()`, `clear_history()`
- **Lifecycle**: context manager, explicit `close()`
- **Structured output**: JSON Schema dicts and Pydantic models (`schema=`, `.parsed`, `.parse_as()`)
- **Transcript tracking**: `transcript` vs. `last_generation_transcript` across single/multiple/structured/streaming generations, after `clear_history()`, and with tool calling

### `test_async_session.py` (Unit Tests)
Mirrors `test_session.py` for `applefoundationmodels.AsyncSession`: async generation, async streaming, async history management, `async with` / `aclose()` lifecycle (including that sync `close()` refuses to run inside an active event loop but works outside one), and async structured output.

### `test_session_isolation.py` (Unit Tests)
Regression tests for session isolation: distinct session ids, per-session instructions/history/tools not leaking across sessions, `clear_history()` scoped to one session, and closing one session leaving others usable. These exist because the native layer used to keep exactly one global session under the hood — see `CLAUDE.md`'s architecture section.

### `test_session_lifecycle.py` (Unit Tests)
Tests for static/lifecycle surface shared across sessions:
- **Availability**: `check_availability()`, `get_availability_reason()`, `is_ready()`
- **Info**: `get_version()`
- **Lifecycle**: context manager, `close()`, multiple concurrent sessions
- **Closed-session errors**: every method that should raise `RuntimeError("Session is closed")` after `close()` (`generate()`, `generate(stream=True)`, `get_history()`, `clear_history()`, `transcript`) does, and `close()` itself is idempotent
- **Creation**: with/without instructions, multiple sessions with different instructions

### `test_tools.py` (Unit Tests)
Tests for tool calling (`tools.py` schema generation + FFI tool registration/execution):
- Schema generation from function signatures (no params, single/multiple/mixed-type params, `Optional[...]`)
- Multiple tools registered on one session, tool return type handling
- `response.tool_calls` presence/absence and structure
- `transcript` shape for `tool_call`/`tool_output` entries
- End-to-end tool calling and large tool output

Note: a couple of these tests assert on the *exact* argument values the on-device model chooses to pass to a tool (e.g. that it calls `multiply` with `operation="multiply"` rather than `"*"`). Those can fail on model nondeterminism without indicating a code regression — check the assertion against what the model actually returned before assuming a real bug.

### `test_exceptions.py` (Unit Tests)
Tests that every error code in `error_codes.json` round-trips correctly: code → exception class mapping, `raise_for_error_code()`, exception hierarchy (all inherit `FoundationModelsError`; generation errors inherit `GenerationError`), and the Swift → Cython → Python error JSON contract (including malformed/missing-field edge cases). Runs without requiring Apple Intelligence to be available, since it simulates the Cython error-handling logic directly rather than calling into the FFI.

### `test_integration.py` (Integration Tests, `@pytest.mark.integration`)
End-to-end smoke tests, run as a sequence: availability, version, basic generation (math/knowledge/creative prompts), multi-turn conversation with context, async streaming, temperature variations, multiple session management, error handling (empty/long prompts), and context manager cleanup.

## Running Tests

### Run all tests:
```bash
pytest
```

### Run only unit tests:
```bash
pytest -m "not integration"
```

### Run only integration tests:
```bash
pytest -m integration
```
(equivalently: `pytest tests/test_integration.py`)

### Run a single file or test:
```bash
pytest tests/test_session.py
pytest tests/test_session.py -k test_generate_basic
```

### Run with coverage:
```bash
pip install pytest-cov
pytest --cov=applefoundationmodels --cov-report=html
```

### Skip tests if Apple Intelligence unavailable:
Tests automatically skip if Apple Intelligence is not available, using the `check_availability` fixture.

## Test Coverage Summary

### ✅ Well-Covered Areas:
- Session and AsyncSession creation, lifecycle, and closed-session error behavior
- Session isolation: independent instructions, history, and tool catalogs across concurrently-alive sessions
- Text generation (sync + async), temperature control, `max_tokens`
- Streaming generation (sync + async)
- Structured output (JSON Schema and Pydantic)
- Tool calling: schema generation, registration, execution, transcript shape
- Transcript / `last_generation_transcript` tracking across all generation modes
- Every error code's Python exception mapping and the Swift→Python error JSON contract
- Availability checking

### ⚠️ Limited Coverage:
- Tool-call argument correctness is asserted loosely in places because the on-device model's exact choice of argument values isn't fully deterministic
- Concurrent use of a single `Session`/`AsyncSession` from multiple threads/tasks at once (the framework itself raises `ConcurrentRequestsError` for this — see `exceptions.py` — but there's no test exercising that path)

### ❌ Not Covered (and not currently planned):
- Performance benchmarks, memory leak detection, stress/load testing
- Cross-platform compatibility — this project only supports Apple Silicon macOS 26+, so this is out of scope by design, not a gap
- Compatibility across different FoundationModels/Swift dylib versions

## Requirements

Tests require:
- macOS 26.0+ with Apple Intelligence enabled
- `pytest>=7.0`
- `pytest-asyncio>=0.20`

Install dev dependencies:
```bash
uv sync --extra dev
# or
pip install -e ".[dev]"
```

## CI/CD Considerations

Since most tests require Apple Intelligence on macOS 26.0+:
- Cannot run on standard (non-macOS, non-Apple-Silicon) CI runners
- `test_exceptions.py` is the exception — it doesn't call into the FFI, so it can run anywhere
- `.github/workflows/test.yml` runs on `macos-26` runners (Apple Intelligence available) across the Python 3.9-3.14 matrix, builds a wheel with `uv build --wheel`, installs it into a clean venv, and runs the *entire* `tests/` directory unfiltered (unit + integration together) — there's no separate fast/slow split in CI today
- A separate `lint` job runs `black --check` and `mypy` (mypy is `continue-on-error: true`, so it's advisory only)

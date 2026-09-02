# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Unofficial Python bindings for Apple's FoundationModels framework (on-device LLM, macOS 26+ / Apple Intelligence). Only builds/runs on Apple Silicon macOS 26.0+ with Xcode command line tools; there is no meaningful cross-platform CI path — GitHub Actions in `.github/workflows/` (`test.yml`, `publish-to-pypi.yml`) require macOS runners.

## Build

The package requires a compiled Swift dylib and a Cython extension; both are built automatically by `setup.py` on install — there is no separate manual build step for normal development.

```bash
uv sync --extra dev        # install dev deps (pytest, mypy, black, pre-commit)
pip install -e .           # (re)builds Swift dylib + Cython extension, then installs editable
```

Force a rebuild after touching Swift or Cython sources (the dylib build is skipped if source mtimes are older than the existing dylib):

```bash
pip install --force-reinstall --no-cache-dir -e .
```

`uv build --wheel` builds a wheel; wheels are architecture/version-specific (arm64 only — see `setup.py`'s `ARCH = "arm64"`; x86_64 Macs are not supported).

## Test

```bash
uv run pytest                                   # all tests
uv run pytest tests/test_session.py             # single file
uv run pytest tests/test_session.py -k test_foo # single test
uv run pytest -m "not integration"              # unit tests only, skip integration
uv run pytest --cov=applefoundationmodels --cov-report=html
```

Tests that need real Apple Intelligence auto-skip via the `check_availability` fixture in `tests/conftest.py` if it isn't available/enabled on the machine (see `Session.check_availability()` / `get_availability_reason()`). `tests/test_integration.py` is marked `integration` and exercises the framework end-to-end; the rest are unit tests. There is no mock/stub FFI layer — unit tests still call into the real Swift library, just without requiring generation to succeed.

## Lint / format / typecheck

```bash
uv run black applefoundationmodels examples   # formatting (pre-commit hook enforces this: .pre-commit-config.yaml)
uv run mypy applefoundationmodels             # type checking (config in pyproject.toml: python 3.10 target, untyped defs allowed)
```

## Architecture

Five-layer stack, each layer only talking to the one directly below it:

```
Python API      session.py / async_session.py / base_session.py
       ↓
Cython FFI      _foundationmodels.pyx  (compiled against _foundationmodels.pxd)
       ↓
C FFI layer     swift/foundation_models.h  (extern "C" declarations, ai_result_t / ai_availability_t enums)
       ↓
Swift impl      swift/foundation_models.swift  (actual calls into Apple's framework)
       ↓
FoundationModels framework (Apple Intelligence, on-device)
```

- **`base_session.py`** (`BaseSession`, abstract) holds essentially all session logic shared between sync and async: transcript tracking, generation planning (`_plan_generate_call` picks `text`/`structured`/`stream` mode), the streaming producer/consumer machinery (a background thread pushes chunks through a queue adapter; sync and async each supply their own `_StreamQueueAdapter`), tool-call extraction from the transcript, and the static availability/version methods. `session.py` and `async_session.py` are thin subclasses that implement `_call_ffi` (sync passthrough vs. `asyncio.to_thread`) and `_create_stream_queue_adapter`. When changing session behavior, change it in `base_session.py` unless it's genuinely sync- or async-specific.
- **Every session is independently identified by a native session id.** `apple_ai_create_session()` returns a positive integer handle; the Swift layer keeps a lock-guarded `[Int32: SessionEntry]` registry (`foundation_models.swift`) rather than a single global session, so each `Session`/`AsyncSession` object's instructions, conversation history, and tool catalog are fully isolated from every other one. Every per-session FFI call (`generate`, `generate_stream`, `generate_structured`, `get_transcript`, `get_history`, `clear_history`) takes that session id as its first argument all the way down through Cython (`_foundationmodels.pyx`) and the C header; `Session.close()`/`AsyncSession.aclose()` call `close_session(session_id)` to free the entry. Tool execution dispatch (`_tool_callback_wrapper` in `_foundationmodels.pyx`) is likewise keyed by session id, and the streaming callback registry (`_stream_callbacks`) is keyed by session id so two sessions can stream concurrently from separate threads without clobbering each other's chunks. **This wasn't always true** — before this design, the native layer held one process-wide `LanguageModelSession`, so creating a second `Session` silently replaced the first one's state everywhere (instructions, history, tool calls) without raising an error. `tests/test_session_isolation.py` is the regression suite for this; if you touch session lifecycle, run it.
- Streaming always runs the FFI call on a background `threading.Thread` regardless of sync/async — async streaming wraps queue reads in `asyncio.to_thread`/an async queue, it doesn't call the FFI stream function directly on the event loop.
- **`_foundationmodels.pyx`/`.pxd`**: the only place that talks to the C ABI. Functions return raw C structures/strings that Python-side code owns and must free (`apple_ai_free_string`); Cython release the GIL (`nogil`) around most calls.
- **Error codes are generated, not hand-duplicated**: `applefoundationmodels/error_codes.json` is the single source of truth for exception codes/names/parent classes. `setup.py`'s `generate_swift_error_code_file()` renders it into `swift/error_codes.generated.swift` (a `Do not edit manually` file) at build time, and `exceptions.py` maps the same codes to Python exception classes. To add/change an error code, edit `error_codes.json` and rebuild — don't hand-edit the generated Swift file or let Python/Swift definitions drift apart.
- **`tools.py`** builds JSON Schema for Python functions passed as `tools=[...]` (via `inspect`/`get_type_hints`) so the model can call them; `base_session.py._build_session_config` registers them with the FFI (`register_tools`) at session creation. Tool calls show up as `tool_call`/`tool_output` entries in `session.transcript`.
- **`pydantic_compat.py`** is an optional bridge so `generate(schema=SomeBaseModel)` accepts Pydantic v2 models in addition to raw JSON Schema dicts; only loaded/needed when `pydantic` is installed (`pip install apple-foundation-models[pydantic]`).
- **Transcript vs. last_generation_transcript**: `transcript` is the full FFI-backed conversation history (queried live each access, not cached in Python). `last_generation_transcript` is a Python-side slice from `_last_transcript_length`, set at the start of each `generate()` call — it exists so callers can inspect just what happened in the most recent call (e.g. which tools fired) without diffing the whole history themselves.
- Context window is 4096 tokens per session (instructions + prompts + outputs combined) — enforced by the framework, surfaced as `ContextWindowExceededError`.

## Key constraint when editing across layers

A change to the C surface touches four files in lockstep: `swift/foundation_models.h` (declaration) → `swift/foundation_models.swift` (implementation) → `_foundationmodels.pxd` (Cython `cdef extern` declaration) → `_foundationmodels.pyx` (Cython wrapper) → the Python-facing method in `base_session.py`/`session.py`/`async_session.py`. Missing any layer produces a link or import error, not a Python-level test failure, so check all four when adding/changing FFI-exposed functionality. Also update `_foundationmodels.pyi` (the type stub) — it isn't checked against the `.pyx` at build time, so it silently drifts if you forget it. Any new per-session FFI function needs a `session_id: Int32`/`int32_t` as its first parameter to stay consistent with the rest of the surface (see the session-isolation note above) — don't add a new global/singleton piece of native state.

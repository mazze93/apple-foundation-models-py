# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Unofficial Python bindings for Apple's FoundationModels framework (on-device LLM, macOS 26+ / Apple Intelligence). Only builds/runs on Apple Silicon macOS 26.0+ with Xcode command line tools; there is no meaningful cross-platform CI path — GitHub Actions in `.github/workflows/` (`test.yml`, `publish-to-pypi.yml`) require macOS runners.

**This is mazze's personal fork, developed and run on macOS 27, for their own on-device use.** It is intentionally diverging from upstream: don't constrain changes to what upstream would accept. Remotes: `origin` = `mazze93/apple-foundation-models-py` (push and PR here, against `main`); `upstream` = `btucker/...` — never push or open PRs there without asking.

Swift sources and the C header live in `applefoundationmodels/swift/` (there is no top-level `swift/` dir).

## Build

The package requires a compiled Swift dylib and a Cython extension; both are built automatically by `setup.py` on install — there is no separate manual build step for normal development. The uv-managed `.venv` has no `pip`; use uv.

```bash
uv sync --extra dev        # install dev deps (pytest, mypy, ruff, pre-commit) + editable build of the package
```

`uv run` silently rebuilds the package whenever `pyproject.toml`/`setup.py` change, so a syntax error in `setup.py` breaks *every* `uv run` command, not just installs. Force a rebuild after touching Swift or Cython sources (the dylib build is skipped if source mtimes are older than the existing dylib):

```bash
uv sync --extra dev --reinstall-package apple-foundation-models
```

**`afm` CLI** (`applefoundationmodels/cli.py`, entry point in `[project.scripts]`) is installed machine-wide with `uv tool install --editable .`. The tool env imports from this source tree and shares its compiled `.so`/`.dylib` (both envs are Python 3.13), so Python edits and `--reinstall-package` rebuilds go live in `afm` immediately; if the two Python versions ever diverge, rerun `uv tool install --editable --reinstall .`. Its context pre-flight reserves `--max-tokens` plus `FRAMEWORK_OVERHEAD_TOKENS` (128): the framework uses ~70–85 tokens that `token_count` doesn't report — measured, not guessed; re-measure if a new OS changes it.

`uv build --wheel` builds a wheel; wheels are architecture/version-specific (arm64 only — see `setup.py`'s `ARCH = "arm64"`; x86_64 Macs are not supported).

**Python floor is 3.9** (`requires-python`, CI matrix 3.9–3.14) while the local venv is newer, so nothing local catches 3.10+ syntax: no `X | Y` annotations or `match` in runtime code — use `Optional`/`Union`. (This is also why ruff's `UP` rules are off.)

## Test

```bash
uv run pytest                                   # all tests
uv run pytest tests/test_session.py             # single file
uv run pytest tests/test_session.py -k test_foo # single test
uv run pytest -m "not integration"              # unit tests only, skip integration
uv run pytest --cov=applefoundationmodels --cov-report=html
```

Tests that need real Apple Intelligence auto-skip via the `check_availability` fixture in `tests/conftest.py` if it isn't available/enabled on the machine (see `Session.check_availability()` / `get_availability_reason()`). `tests/test_integration.py` is marked `integration` and exercises the framework end-to-end; the rest are unit tests. There is no mock/stub FFI layer — unit tests still call into the real Swift library, just without requiring generation to succeed.

CI tests the **installed wheel**, not the source tree (`uv build --wheel`, install, run pytest from `/tmp`). A new non-`.py` runtime file must match `[tool.setuptools.package-data]` in `pyproject.toml` or CI fails even though the editable install works locally.

## Lint / format / typecheck

ruff is the single linter + formatter (replaced black; config in `pyproject.toml`, rules `E,F,W,I,B`, `*.md` excluded). It's enforced by the pre-commit hook and the CI `lint` job; mypy is `continue-on-error` in CI, so it's advisory.

```bash
uv run ruff check --fix . && uv run ruff format .
uv run mypy applefoundationmodels             # config in pyproject.toml: python 3.10 target, untyped defs allowed
```

## Architecture

Five-layer stack, each layer only talking to the one directly below it:

```
Python API      session.py / async_session.py / base_session.py
       ↓
Cython FFI      _foundationmodels.pyx  (compiled against _foundationmodels.pxd)
       ↓
C FFI layer     applefoundationmodels/swift/foundation_models.h  (extern "C" declarations, ai_result_t / ai_availability_t enums)
       ↓
Swift impl      applefoundationmodels/swift/foundation_models.swift  (actual calls into Apple's framework)
       ↓
FoundationModels framework (Apple Intelligence, on-device)
```

- **`base_session.py`** (`BaseSession`, abstract) holds essentially all session logic shared between sync and async: transcript tracking, generation planning (`_plan_generate_call` picks `text`/`structured`/`stream` mode), the streaming producer/consumer machinery (a background thread pushes chunks through a queue adapter; sync and async each supply their own `_StreamQueueAdapter`), tool-call extraction from the transcript, and the static availability/version methods. `session.py` and `async_session.py` are thin subclasses that implement `_call_ffi` (sync passthrough vs. `asyncio.to_thread`) and `_create_stream_queue_adapter`. When changing session behavior, change it in `base_session.py` unless it's genuinely sync- or async-specific.
- **Every session is independently identified by a native session id.** `apple_ai_create_session()` returns a positive integer handle; the Swift layer keeps a lock-guarded `[Int32: SessionEntry]` registry (`foundation_models.swift`) rather than a single global session, so each `Session`/`AsyncSession` object's instructions, conversation history, and tool catalog are fully isolated from every other one. Every per-session FFI call (`generate`, `generate_stream`, `generate_structured`, `get_transcript`, `get_history`, `clear_history`) takes that session id as its first argument all the way down through Cython (`_foundationmodels.pyx`) and the C header; `Session.close()`/`AsyncSession.aclose()` call `close_session(session_id)` to free the entry. Tool execution dispatch (`_tool_callback_wrapper` in `_foundationmodels.pyx`) is likewise keyed by session id, and the streaming callback registry (`_stream_callbacks`) is keyed by session id so two sessions can stream concurrently from separate threads without clobbering each other's chunks. **This wasn't always true** — before this design, the native layer held one process-wide `LanguageModelSession`, so creating a second `Session` silently replaced the first one's state everywhere (instructions, history, tool calls) without raising an error. `tests/test_session_isolation.py` is the regression suite for this; if you touch session lifecycle, run it.
- Streaming always runs the FFI call on a background `threading.Thread` regardless of sync/async — async streaming wraps queue reads in `asyncio.to_thread`/an async queue, it doesn't call the FFI stream function directly on the event loop.
- **`_foundationmodels.pyx`/`.pxd`**: the only place that talks to the C ABI. Functions return raw C structures/strings that Python-side code owns and must free (`apple_ai_free_string`); Cython release the GIL (`nogil`) around most calls.
- **Error codes are generated, not hand-duplicated**: `applefoundationmodels/error_codes.json` is the single source of truth for exception codes/names/parent classes. `setup.py`'s `generate_swift_error_code_file()` renders it into `applefoundationmodels/swift/error_codes.generated.swift` (a `Do not edit manually` file) at build time, and `exceptions.py` maps the same codes to Python exception classes. To add/change an error code, edit `error_codes.json` and rebuild — don't hand-edit the generated Swift file or let Python/Swift definitions drift apart.
- **`tools.py`** builds JSON Schema for Python functions passed as `tools=[...]` (via `inspect`/`get_type_hints`) so the model can call them; `base_session.py._build_session_config` registers them with the FFI (`register_tools`) at session creation. Tool calls show up as `tool_call`/`tool_output` entries in `session.transcript`. `required` semantics are load-bearing: the Swift converter (`convertObjectType`) makes properties not in `required` optional, but treats a schema with **no** `required` key as all-required (so old structured-output schemas keep their meaning). Tool schemas therefore always emit `required` (even `[]`); dropping it forces the model to invent values for defaulted params or stall asking for them instead of calling the tool. `Literal["a", "b"]` maps to a string `enum` (→ `anyOf` natively).
- **`pydantic_compat.py`** is an optional bridge so `generate(schema=SomeBaseModel)` accepts Pydantic v2 models in addition to raw JSON Schema dicts; only loaded/needed when `pydantic` is installed (`pip install apple-foundation-models[pydantic]`).
- **Transcript vs. last_generation_transcript**: `transcript` is the full FFI-backed conversation history (queried live each access, not cached in Python). `last_generation_transcript` is a Python-side slice from `_last_transcript_length`, set at the start of each `generate()` call — it exists so callers can inspect just what happened in the most recent call (e.g. which tools fired) without diffing the whole history themselves.
- The context window is per session (instructions + history + prompt + output combined) and **model-dependent** — 8192 tokens on macOS 27 (it was 4096 on macOS 26). Never hard-code it: use `Session.context_size()`, and `Session.token_count(text)` (macOS 26.4+) to check input first. Overflow raises `ContextWindowExceededError`.

## Key constraint when editing across layers

A change to the C surface touches every layer in lockstep: `applefoundationmodels/swift/foundation_models.h` (declaration) → `applefoundationmodels/swift/foundation_models.swift` (implementation) → `_foundationmodels.pxd` (Cython `cdef extern` declaration) → `_foundationmodels.pyx` (Cython wrapper) → the Python-facing method in `base_session.py`/`session.py`/`async_session.py`. Missing any layer produces a link or import error, not a Python-level test failure, so check every layer when adding/changing FFI-exposed functionality. Also update `_foundationmodels.pyi` (the type stub) — it isn't checked against the `.pyx` at build time, so it silently drifts if you forget it. Any new per-session FFI function needs a `session_id: Int32`/`int32_t` as its first parameter to stay consistent with the rest of the surface (see the session-isolation note above) — don't add a new global/singleton piece of native state.

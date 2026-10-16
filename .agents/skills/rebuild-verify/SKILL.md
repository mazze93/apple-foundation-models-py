---
name: rebuild-verify
description: Rebuild the Swift dylib + Cython extension and run the full local gate (ruff, mypy, unit tests, session-isolation suite). Use after touching Swift, Cython, the C header, setup.py, or session/streaming code, and before committing.
---

Run these from the repo root, in order. Stop at the first hard failure and report it with the output; don't paper over it.

1. **Rebuild.** The dylib build is skipped when source mtimes are older than the existing dylib, so force it:
   ```bash
   uv sync --extra dev --reinstall-package apple-foundation-models
   ```
   Then confirm the native layer loads and report availability (tests auto-skip if it isn't `AVAILABLE`, so say so if it isn't — a green run with everything skipped is not a pass):
   ```bash
   uv run python -c "import applefoundationmodels as a; print(a.Session.check_availability(), a.Session.get_availability_reason())"
   ```

2. **Lint + format (hard gate, matches CI):**
   ```bash
   uv run ruff check . && uv run ruff format --check .
   ```

3. **Types (advisory — CI runs mypy with `continue-on-error`):** `uv run mypy applefoundationmodels`. Report new errors in files you changed; don't block on pre-existing ones.

4. **Tests:**
   ```bash
   uv run pytest -m "not integration" -ra
   uv run pytest tests/test_session_isolation.py -ra
   ```
   Report the skip count from `-ra`.

5. **Concurrency probe (only if the change touched session lifecycle, streaming, tool dispatch, or the Swift session registry).** Sequential tests in this repo have missed a real concurrent-streaming hang before. Write a throwaway script in the scratchpad (not the repo) that, under a hard timeout, runs two `Session`s streaming concurrently from separate threads, and two `AsyncSession`s via `asyncio.gather`, and asserts each stream's chunks stay with its own session and nothing hangs. If it surfaces a bug, add it as a regression test in `tests/test_session_isolation.py`.

6. **If the C surface changed**, check every layer is in step: `applefoundationmodels/swift/foundation_models.h`, `foundation_models.swift`, `_foundationmodels.pxd`, `_foundationmodels.pyx`, `_foundationmodels.pyi` (the stub drifts silently), and the Python method in `base_session.py`/`session.py`/`async_session.py`.

7. **If you added a non-`.py` runtime file**, confirm it's covered by `[tool.setuptools.package-data]` — CI tests the built wheel, not the source tree. Optionally prove it: `uv build --wheel && unzip -l dist/*.whl | grep <file>`.

Finish with a short pass/fail table per step.

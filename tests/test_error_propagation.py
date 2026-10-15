"""
Regression tests: native generation errors must raise, not come back as text.

Swift reports failures as a ``{"error": ..., "error_code": ...}`` payload over
the same channel as model output. Two bugs once let that payload leak through
as a "successful" response:

- ``generate()`` only recognised the payload by a ``'{"error"'`` string prefix,
  but JSONSerialization doesn't fix key order, so ``{"error_code": ...}``-first
  payloads were returned as response text.
- Streaming forwarded the payload as an ordinary chunk followed by the
  end-of-stream signal, so consumers stopped before the exception arrived.

Context-window overflow is the easiest way to trigger a real native error.
"""

import pytest

from applefoundationmodels import AsyncSession, Session
from applefoundationmodels._foundationmodels import _parse_error_payload
from applefoundationmodels.exceptions import ContextWindowExceededError

# Far beyond the on-device context window
OVERSIZED_PROMPT = (
    "Summarize: " + "The quick brown fox jumps over the lazy dog. " * 2000
)


class TestParseErrorPayload:
    @pytest.mark.parametrize(
        "payload",
        [
            '{"error":"boom","error_code":-14}',
            '{"error_code":-14,"error":"boom"}',
        ],
    )
    def test_detects_payload_in_either_key_order(self, payload):
        assert _parse_error_payload(payload) == (-14, "boom")

    @pytest.mark.parametrize(
        "text",
        [
            "plain model output",
            '{"error": "a field the model chose to emit"}',
            '{"error":"x","error_code":-1,"extra":1}',
            "[1, 2, 3]",
            "{not json",
        ],
    )
    def test_ignores_non_payloads(self, text):
        assert _parse_error_payload(text) is None


class TestContextOverflowRaises:
    def test_generate_raises(self, check_availability):
        with Session() as session:
            with pytest.raises(ContextWindowExceededError):
                session.generate(OVERSIZED_PROMPT)

    def test_stream_raises_without_yielding_error_text(self, check_availability):
        chunks = []
        with Session() as session:
            with pytest.raises(ContextWindowExceededError):
                for chunk in session.generate(OVERSIZED_PROMPT, stream=True):
                    chunks.append(chunk.content)
        assert not any("error_code" in c for c in chunks)

    async def test_async_stream_raises(self, check_availability):
        async with AsyncSession() as session:
            with pytest.raises(ContextWindowExceededError):
                async for _ in session.generate(OVERSIZED_PROMPT, stream=True):
                    pass

    def test_session_still_usable_after_overflow(self, check_availability):
        with Session() as session:
            with pytest.raises(ContextWindowExceededError):
                session.generate(OVERSIZED_PROMPT)
            assert session.generate("Say hi.", max_tokens=20).text

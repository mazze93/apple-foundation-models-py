"""
Unit tests for applefoundationmodels.AsyncSession

Mirrors tests/test_session.py so AsyncSession gets the same unit-level
coverage as Session, independent of the end-to-end integration test's
single streaming smoke test.
"""

import asyncio

import pytest
import applefoundationmodels
from applefoundationmodels import AsyncSession
from conftest import assert_valid_response, assert_valid_chunks


class TestAsyncSessionGeneration:
    """Tests for async text generation."""

    async def test_generate_basic(self, async_session, check_availability):
        """Test basic async text generation."""
        response = await async_session.generate("What is 2 + 2?", temperature=0.3)
        assert isinstance(response.text, str), "Response should have text property"
        assert (
            "4" in response.text or "four" in response.text.lower()
        ), "Response should contain the answer to 2+2"

    async def test_generate_with_temperature(self, async_session, check_availability):
        """Test async generation with different temperatures."""
        prompt = "Complete: The sky is"

        response1 = await async_session.generate(prompt, temperature=0.1)
        assert isinstance(response1.text, str), "Response should have text property"

        response2 = await async_session.generate(prompt, temperature=1.5)
        assert isinstance(response2.text, str), "Response should have text property"

    async def test_generate_with_max_tokens(self, async_session, check_availability):
        """Test async generation with token limit."""
        response_short = await async_session.generate(
            "Write a long story about space exploration", max_tokens=20, temperature=0.5
        )
        assert isinstance(
            response_short.text, str
        ), "Response should have text property"

        response_long = await async_session.generate(
            "Write a long story about space exploration",
            max_tokens=200,
            temperature=0.5,
        )
        assert isinstance(response_long.text, str), "Response should have text property"
        assert len(response_long.text) > len(response_short.text), (
            f"Higher max_tokens should produce longer response: "
            f"short={len(response_short.text)} chars, long={len(response_long.text)} chars"
        )


class TestAsyncSessionStreaming:
    """Tests for async streaming generation."""

    async def test_generate_stream_basic(self, async_session, check_availability):
        """Test basic async streaming generation."""
        chunks = []
        async for chunk in async_session.generate(
            "Count to 5", stream=True, temperature=0.3
        ):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive at least one chunk"
        for chunk in chunks:
            assert hasattr(chunk, "content"), "Chunk should have content attribute"
            assert isinstance(chunk.content, str), "Chunk content should be string"

    async def test_generate_stream_with_temperature(
        self, async_session, check_availability
    ):
        """Test async streaming with a non-default temperature."""
        chunks = []
        async for chunk in async_session.generate(
            "Say hello", stream=True, temperature=1.0
        ):
            chunks.append(chunk)

        assert len(chunks) > 0, "Should receive at least one chunk"
        for chunk in chunks:
            assert hasattr(chunk, "content"), "Chunk should have content attribute"


class TestAsyncSessionHistory:
    """Tests for async conversation history."""

    async def test_get_history(self, async_session, check_availability):
        """Test getting conversation history asynchronously."""
        history = await async_session.get_history()
        assert isinstance(history, list)

    async def test_clear_history(self, async_session, check_availability):
        """Test clearing conversation history asynchronously."""
        response = await async_session.generate("Hello", temperature=0.5)
        assert isinstance(response.text, str), "Response should have text property"

        history_before = await async_session.get_history()
        assert isinstance(history_before, list)

        await async_session.clear_history()

        history_after = await async_session.get_history()
        assert isinstance(history_after, list)
        assert len(history_after) == 0, "History should be empty after clearing"


class TestAsyncSessionLifecycle:
    """Tests for async session lifecycle."""

    async def test_session_async_context_manager(self, check_availability):
        """Test AsyncSession works as an async context manager."""
        async with AsyncSession() as session:
            assert session is not None
            response = await session.generate("Hello", temperature=0.5)
            assert isinstance(response.text, str), "Response should have text property"

    async def test_session_aclose(self, check_availability):
        """Test explicit async close."""
        session = AsyncSession()
        response = await session.generate("Hello", temperature=0.5)
        assert isinstance(response.text, str), "Response should have text property"
        await session.aclose()
        # aclose should complete without error

    async def test_double_aclose_is_idempotent(self, check_availability):
        """Calling aclose() twice should not raise."""
        session = AsyncSession()
        await session.aclose()
        await session.aclose()  # should be a no-op, not an error

    def test_close_raises_inside_running_loop(self, check_availability):
        """Sync close() must refuse to run while an event loop is active."""

        async def use_sync_close_from_coroutine():
            session = AsyncSession()
            try:
                with pytest.raises(RuntimeError, match="event loop"):
                    session.close()
            finally:
                await session.aclose()

        asyncio.run(use_sync_close_from_coroutine())

    def test_close_outside_loop_drives_aclose(self, check_availability):
        """Sync close() should work when no event loop is running."""
        session = AsyncSession()
        session.close()  # drives asyncio.run(self.aclose()) internally


class TestAsyncClosedSessionErrors:
    """Tests for operations on a closed AsyncSession."""

    async def test_generate_after_close_raises(self, check_availability):
        """generate() on a closed async session should raise RuntimeError."""
        session = AsyncSession()
        await session.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await session.generate("Hello")

    async def test_get_history_after_close_raises(self, check_availability):
        """get_history() on a closed async session should raise RuntimeError."""
        session = AsyncSession()
        await session.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await session.get_history()

    async def test_transcript_after_close_raises(self, check_availability):
        """transcript property on a closed async session should raise RuntimeError."""
        session = AsyncSession()
        await session.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            _ = session.transcript


class TestAsyncStructuredOutput:
    """Tests for async structured output generation."""

    async def test_generate_structured_basic(self, async_session):
        """Test basic async structured output generation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        response = await async_session.generate(
            "Extract information: John is 30 years old", schema=schema
        )

        result = response.parsed
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "name" in result, "Result should have 'name' field"
        assert "age" in result, "Result should have 'age' field"
        assert isinstance(result["name"], str), "Name should be a string"
        assert isinstance(result["age"], int), "Age should be an integer"

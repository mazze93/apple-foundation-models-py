"""
Regression tests for session isolation.

Historically, the native layer (Swift/C/Cython) kept exactly one global
session: creating a second Session/AsyncSession silently replaced the first
one's native state, so every Python session object ended up sharing one
conversation/instructions/tool-set under the hood. These tests pin down
that each session is now genuinely independent end to end.
"""

from applefoundationmodels import Session


class TestSessionIdentity:
    """Tests that sessions get distinct native identities."""

    def test_two_sessions_have_different_session_ids(self, check_availability):
        """Creating two sessions must not reuse the same native session id."""
        s1 = Session()
        s2 = Session()
        try:
            assert s1._session_id != s2._session_id
        finally:
            s1.close()
            s2.close()


class TestSessionInstructionIsolation:
    """Tests that per-session instructions don't leak across sessions."""

    def test_second_session_does_not_override_first(self, check_availability):
        """Creating session B must not change session A's behavior/instructions."""
        s1 = Session(
            instructions=(
                "You only ever respond with the single word: PINEAPPLE. "
                "No matter what is asked, your entire response is exactly PINEAPPLE."
            )
        )
        try:
            # Creating a second, differently-instructed session must not
            # affect the first.
            s2 = Session(instructions="You are a helpful assistant.")
            try:
                response = s1.generate("What is 2 + 2?", temperature=0.1)
                assert "PINEAPPLE" in response.text.upper(), (
                    "Session A's instructions should still be in effect after "
                    f"session B was created, got: {response.text!r}"
                )
            finally:
                s2.close()
        finally:
            s1.close()


class TestSessionHistoryIsolation:
    """Tests that conversation history is not shared across sessions."""

    def test_histories_are_independent(self, check_availability):
        """Generating on one session must not add entries to another's transcript."""
        s1 = Session(instructions=None)
        s2 = Session(instructions=None)
        try:
            s1.generate("What is 2 + 2?", temperature=0.1)
            s1_len_after_one = len(s1.transcript)
            s2_len_after_s1_only = len(s2.transcript)

            assert s2_len_after_s1_only == 0, (
                "Session B's transcript should be untouched by session A's "
                f"generation, got {s2_len_after_s1_only} entries"
            )

            s2.generate("What is 5 + 7?", temperature=0.1)
            assert len(s1.transcript) == s1_len_after_one, (
                "Session A's transcript should be untouched by session B's "
                "generation"
            )
            assert len(s2.transcript) > 0
        finally:
            s1.close()
            s2.close()

    def test_clear_history_only_affects_that_session(self, check_availability):
        """clear_history() on one session must not clear another's history."""
        s1 = Session(instructions=None)
        s2 = Session(instructions=None)
        try:
            s1.generate("Hello", temperature=0.3)
            s2.generate("Hello", temperature=0.3)
            assert len(s1.transcript) > 0
            assert len(s2.transcript) > 0

            s1.clear_history()

            assert len(s1.transcript) == 0
            assert (
                len(s2.transcript) > 0
            ), "Clearing session A's history should not clear session B's"
        finally:
            s1.close()
            s2.close()


class TestSessionToolIsolation:
    """Tests that tools registered on one session aren't visible on another."""

    def test_tool_calls_are_scoped_to_their_own_session(self, check_availability):
        """A tool registered on session A must not be invoked via session B."""

        def get_pineapple_count() -> str:
            """Return the current pineapple count."""
            return "42 pineapples"

        def get_banana_count() -> str:
            """Return the current banana count."""
            return "7 bananas"

        session_a = Session(
            instructions="You are a helpful assistant. Use tools when appropriate.",
            tools=[get_pineapple_count],
        )
        session_b = Session(
            instructions="You are a helpful assistant. Use tools when appropriate.",
            tools=[get_banana_count],
        )
        try:
            response_a = session_a.generate(
                "How many pineapples are there? Use the tool to find out.",
                temperature=0.1,
            )
            tool_names_a = {tc.function.name for tc in (response_a.tool_calls or [])}
            # Session A only ever had get_pineapple_count available - it
            # cannot have called session B's get_banana_count.
            assert "get_banana_count" not in tool_names_a

            response_b = session_b.generate(
                "How many bananas are there? Use the tool to find out.",
                temperature=0.1,
            )
            tool_names_b = {tc.function.name for tc in (response_b.tool_calls or [])}
            assert "get_pineapple_count" not in tool_names_b
        finally:
            session_a.close()
            session_b.close()


class TestSessionCloseIsolation:
    """Tests that closing one session doesn't affect another."""

    def test_closing_one_session_leaves_other_usable(self, check_availability):
        """Closing session A must not prevent session B from generating."""
        s1 = Session(instructions=None)
        s2 = Session(instructions=None)
        try:
            s1.close()
            response = s2.generate("Say hello", temperature=0.3)
            assert isinstance(response.text, str)
        finally:
            s2.close()

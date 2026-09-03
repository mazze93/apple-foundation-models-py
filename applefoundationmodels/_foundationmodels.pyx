# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Cython implementation of Apple AI Python bindings.

This module provides low-level Python wrappers around the Swift FoundationModels
C API, handling memory management, error conversion, and callback marshalling.
"""

import json
import threading
from typing import Optional, Callable, Any, Dict
from .exceptions import raise_for_error_code
from .types import Result, Availability
from libc.string cimport memcpy


# ============================================================================
# Helper functions
# ============================================================================

cdef bytes _encode_string(s):
    """Convert Python string to bytes for C."""
    if s is None:
        return None
    if isinstance(s, bytes):
        return s
    return s.encode('utf-8')


cdef str _decode_string(const char *s):
    """Convert C string to Python string."""
    if s == NULL:
        return None
    return s.decode('utf-8')


cdef void _check_result(int32_t result):
    """Check result code and raise exception if error."""
    if result != 0:  # AI_SUCCESS = 0
        raise_for_error_code(result, "Operation failed")


# ============================================================================
# Core library functions
# ============================================================================

def init() -> None:
    """
    Initialize the Apple AI library.

    Must be called before using any other library functions.

    Raises:
        InitializationError: If initialization fails
        NotAvailableError: If Apple Intelligence is not available
    """
    cdef int32_t result
    with nogil:
        result = apple_ai_init()
    _check_result(result)


def cleanup() -> None:
    """
    Cleanup and shutdown the Apple AI library.

    Should be called when the application is done using the library.
    """
    with nogil:
        apple_ai_cleanup()


def get_version() -> str:
    """
    Get library version string.

    Returns:
        Version string in format "major.minor.patch"
    """
    cdef const char *version
    with nogil:
        version = apple_ai_get_version()
    return _decode_string(version)


# ============================================================================
# Availability functions
# ============================================================================

def check_availability() -> int:
    """
    Check Apple Intelligence availability on this device.

    Returns:
        Availability status code from Availability enum
    """
    cdef int32_t status
    with nogil:
        status = apple_ai_check_availability()
    return status


def get_availability_reason() -> Optional[str]:
    """
    Get detailed availability status message.

    Returns:
        Detailed status description, or None if library not initialized
    """
    cdef char *reason
    with nogil:
        reason = apple_ai_get_availability_reason()
    if reason == NULL:
        return None
    try:
        return _decode_string(reason)
    finally:
        apple_ai_free_string(reason)


def is_ready() -> bool:
    """
    Check if Apple Intelligence is ready for immediate use.

    Returns:
        True if ready for use, False otherwise
    """
    cdef int32_t status
    with nogil:
        status = apple_ai_check_availability()
    return status == 1  # AI_AVAILABLE = 1


# ============================================================================
# Tool calling
# ============================================================================

# Tool functions registered per-session: {session_id: {tool_name: callable}}.
# Populated by create_session() before the session handle is even known to
# Python isn't possible, so the entry is added right after the FFI call
# returns a session_id (see create_session below).
cdef object _registered_tools = {}


cdef int32_t _tool_callback_wrapper(
    int32_t session_id,
    const char *tool_name,
    const char *arguments_json,
    char *result_buffer,
    int32_t buffer_size
) noexcept with gil:
    """
    C callback wrapper for tool execution.

    Called from Swift when the model wants to execute a tool. session_id
    scopes the lookup to that session's own registered tools, so two
    sessions with same-named tools never cross-execute each other's
    functions.
    """
    global _registered_tools

    if tool_name == NULL or arguments_json == NULL or result_buffer == NULL:
        return -3  # AI_ERROR_INVALID_PARAMS

    try:
        # Decode inputs
        name_str = tool_name.decode('utf-8')
        args_str = arguments_json.decode('utf-8')

        # Look up tool function, scoped to this session
        session_tools = _registered_tools.get(session_id, {})
        if name_str not in session_tools:
            error_msg = f"Tool '{name_str}' not found"
            error_bytes = error_msg.encode('utf-8')
            if len(error_bytes) < buffer_size:
                for i, b in enumerate(error_bytes):
                    result_buffer[i] = b
                result_buffer[len(error_bytes)] = 0
            return -11  # AI_ERROR_TOOL_NOT_FOUND

        tool_func = session_tools[name_str]

        # Parse arguments
        args_dict = json.loads(args_str)

        # Execute tool
        try:
            result = tool_func(**args_dict)

            # Convert result to string
            if result is None:
                result_str = ""
            elif isinstance(result, str):
                result_str = result
            else:
                result_str = json.dumps(result)

            # Write to buffer using memcpy
            result_bytes = result_str.encode('utf-8')
            if len(result_bytes) >= buffer_size:
                # Buffer too small - signal retry needed
                error_msg = f"Result too large: {len(result_bytes)} bytes (buffer: {buffer_size} bytes)"
                error_bytes = error_msg.encode('utf-8')
                if len(error_bytes) >= buffer_size:
                    error_bytes = error_bytes[:buffer_size-1]
                memcpy(result_buffer, <char*>error_bytes, len(error_bytes))
                result_buffer[len(error_bytes)] = 0
                return -13  # AI_ERROR_BUFFER_TOO_SMALL

            memcpy(result_buffer, <char*>result_bytes, len(result_bytes))
            result_buffer[len(result_bytes)] = 0

            return 0  # AI_SUCCESS

        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            error_bytes = error_msg.encode('utf-8')
            if len(error_bytes) >= buffer_size:
                error_bytes = error_bytes[:buffer_size-1]
            memcpy(result_buffer, <char*>error_bytes, len(error_bytes))
            result_buffer[len(error_bytes)] = 0
            return -12  # AI_ERROR_TOOL_EXECUTION

    except Exception as e:
        error_msg = f"Callback error: {str(e)}"
        error_bytes = error_msg.encode('utf-8')
        if len(error_bytes) >= buffer_size:
            error_bytes = error_bytes[:buffer_size-1]
        memcpy(result_buffer, <char*>error_bytes, len(error_bytes))
        result_buffer[len(error_bytes)] = 0
        return -99  # AI_ERROR_UNKNOWN


# ============================================================================
# Session management
# ============================================================================

def create_session(
    config: Optional[Dict[str, Any]] = None,
    tools: Optional[Dict[str, Callable]] = None,
) -> int:
    """
    Create a new, fully independent AI session.

    Args:
        config: Optional configuration dictionary with 'instructions' key
        tools: Optional dictionary mapping tool names to callable functions,
            bound to this session only - other sessions never see them and
            are never affected by them.

    Returns:
        A positive session ID uniquely identifying this session. Pass it to
        every other function in this module that takes a session_id.

    Raises:
        InitializationError: If session creation fails
    """
    global _registered_tools

    cdef bytes config_json_bytes = None
    cdef const char *config_json = NULL
    cdef bytes tools_json_bytes = None
    cdef const char *tools_json = NULL
    cdef int32_t result

    if config:
        config_json_str = json.dumps(config)
        config_json_bytes = _encode_string(config_json_str)
        config_json = config_json_bytes

    if tools:
        tools_list = [
            {
                "name": name,
                "description": getattr(func, "_tool_description", ""),
                "parameters": getattr(func, "_tool_parameters", {}),
            }
            for name, func in tools.items()
        ]
        tools_json_str = json.dumps(tools_list)
        tools_json_bytes = _encode_string(tools_json_str)
        tools_json = tools_json_bytes

    with nogil:
        result = apple_ai_create_session(config_json, tools_json, _tool_callback_wrapper)

    if result <= 0:
        raise_for_error_code(result, "Session creation failed")

    if tools:
        _registered_tools[result] = tools.copy()

    return result


def close_session(session_id: int) -> None:
    """
    Release a session's native resources. Safe to call more than once.

    Args:
        session_id: The session ID returned by create_session()
    """
    global _registered_tools

    cdef int32_t sid = session_id
    with nogil:
        apple_ai_close_session(sid)

    _registered_tools.pop(session_id, None)


def get_transcript(session_id: int) -> list:
    """
    Get a session's transcript including tool calls.

    Args:
        session_id: The session ID returned by create_session()

    Returns:
        List of transcript entries

    Raises:
        GenerationError: If transcript retrieval fails
    """
    cdef int32_t sid = session_id
    cdef char *transcript_json

    with nogil:
        transcript_json = apple_ai_get_transcript(sid)

    if transcript_json == NULL:
        raise_for_error_code(-6, "Failed to get transcript")

    try:
        transcript_str = _decode_string(transcript_json)
        return json.loads(transcript_str)
    finally:
        apple_ai_free_string(transcript_json)


# ============================================================================
# Text generation
# ============================================================================

def generate(
    session_id: int,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 1024
) -> str:
    """
    Generate text response for a prompt on a specific session.

    Args:
        session_id: The session ID returned by create_session()
        prompt: Input text prompt
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response

    Raises:
        GenerationError: If generation fails
        InvalidParametersError: If parameters are invalid
        SessionNotFoundError: If session_id does not identify a live session
    """
    cdef int32_t sid = session_id
    cdef bytes prompt_bytes = _encode_string(prompt)
    cdef const char *prompt_c = prompt_bytes
    cdef char *result_c
    cdef str result_str
    cdef double temp_c = temperature
    cdef int32_t tokens_c = max_tokens

    with nogil:
        result_c = apple_ai_generate(sid, prompt_c, temp_c, tokens_c)

    if result_c == NULL:
        raise RuntimeError("Generation returned NULL")

    # Check if result is an error JSON
    result_str = _decode_string(result_c)
    apple_ai_free_string(result_c)

    # Check for error in response
    if result_str.startswith('{"error"'):
        try:
            error_data = json.loads(result_str)
            error_msg = error_data.get('error', 'Unknown error')
            error_code = error_data.get('error_code', -6)  # Default to GenerationError
            raise_for_error_code(error_code, error_msg)
        except json.JSONDecodeError:
            pass  # Not JSON, treat as normal response

    return result_str


# ============================================================================
# Structured generation
# ============================================================================

def generate_structured(
    session_id: int,
    prompt: str,
    schema: Dict[str, Any],
    temperature: float = 1.0,
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Generate structured JSON output matching a schema, on a specific session.

    Args:
        session_id: The session ID returned by create_session()
        prompt: Input text prompt
        schema: JSON schema the output must conform to
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary with 'object' key containing parsed JSON matching schema

    Raises:
        GenerationError: If generation fails
        InvalidParametersError: If parameters are invalid
        JSONParseError: If schema or response is invalid JSON
        SessionNotFoundError: If session_id does not identify a live session
    """
    cdef int32_t sid = session_id
    cdef bytes prompt_bytes = _encode_string(prompt)
    cdef const char *prompt_c = prompt_bytes

    # Convert schema dict to JSON string
    schema_json_str = json.dumps(schema)
    cdef bytes schema_bytes = _encode_string(schema_json_str)
    cdef const char *schema_c = schema_bytes

    cdef char *result_c
    cdef str result_str
    cdef double temp_c = temperature
    cdef int32_t tokens_c = max_tokens

    with nogil:
        result_c = apple_ai_generate_structured(sid, prompt_c, schema_c, temp_c, tokens_c)

    if result_c == NULL:
        raise RuntimeError("Structured generation returned NULL")

    # Get result string
    result_str = _decode_string(result_c)
    apple_ai_free_string(result_c)

    # Parse JSON response
    try:
        result_data = json.loads(result_str)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse response JSON: {e}")

    # Check for error in response
    if "error" in result_data:
        error_msg = result_data["error"]
        error_code = result_data.get("error_code", -6)  # Default to GenerationError
        raise_for_error_code(error_code, error_msg)

    return result_data


# ============================================================================
# Streaming generation
# ============================================================================

# Callback storage for streaming, keyed by session_id so two sessions can
# stream concurrently (e.g. on separate background threads) without
# clobbering each other's callback. _stream_callbacks_lock makes the
# "only register if nothing is registered for this session_id yet" check
# in generate_stream() atomic - see the comment there for why that matters.
cdef object _stream_callbacks = {}
cdef object _stream_callbacks_lock = threading.Lock()


cdef void _stream_callback_wrapper(int32_t session_id, const char *chunk) noexcept with gil:
    """
    C callback wrapper that calls the Python callback for `session_id`.

    This is called from Swift/C code during streaming generation.
    """
    global _stream_callbacks

    callback = _stream_callbacks.get(session_id)
    if callback is None:
        return

    if chunk == NULL:
        # End of stream signal
        try:
            callback(None)
        except:
            pass  # Ignore exceptions in callback at end of stream
        return

    try:
        chunk_str = chunk.decode('utf-8')
        callback(chunk_str)
    except Exception as e:
        print(f"Error in stream callback: {e}")


def generate_stream(
    session_id: int,
    prompt: str,
    callback: Callable[[Optional[str]], None],
    temperature: float = 1.0,
    max_tokens: int = 1024
) -> None:
    """
    Generate text response with streaming chunks, on a specific session.

    Args:
        session_id: The session ID returned by create_session()
        prompt: Input text prompt
        callback: Function called with each text chunk (None signals end)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate

    Raises:
        GenerationError: If generation fails
        InvalidParametersError: If parameters are invalid
        SessionNotFoundError: If session_id does not identify a live session
    """
    global _stream_callbacks

    cdef int32_t sid = session_id
    cdef bytes prompt_bytes = _encode_string(prompt)
    cdef const char *prompt_c = prompt_bytes
    cdef int32_t result
    cdef double temp_c = temperature
    cdef int32_t tokens_c = max_tokens

    # Only register if no stream is already registered for this session_id.
    # Swift's own per-session busy flag (apple_ai_generate_stream rejects a
    # second concurrent call with errorConcurrentRequests before ever
    # touching the model) is the real serialization point, but a rejected
    # call still runs this wrapper - if it unconditionally overwrote the
    # dict entry, and then unconditionally popped it in `finally`, it would
    # rip the *active* call's callback out from under it mid-stream: every
    # subsequent chunk for the active call would silently find no callback
    # registered and be dropped, and its consumer would block forever
    # waiting for a chunk (or an end-of-stream signal) that will never
    # arrive. Guard both the write and the pop so a call that never became
    # "the" registered stream can't disturb the one that did. The lock
    # makes the check-then-set atomic - without it, two threads could both
    # observe "nothing registered" before either writes.
    with _stream_callbacks_lock:
        registered_here = session_id not in _stream_callbacks
        if registered_here:
            _stream_callbacks[session_id] = callback

    try:
        with nogil:
            result = apple_ai_generate_stream(
                sid,
                prompt_c,
                temp_c,
                tokens_c,
                _stream_callback_wrapper
            )
        _check_result(result)
    finally:
        if registered_here:
            with _stream_callbacks_lock:
                if _stream_callbacks.get(session_id) is callback:
                    _stream_callbacks.pop(session_id, None)


# ============================================================================
# History management
# ============================================================================

def get_history(session_id: int) -> list:
    """
    Get conversation history.

    Args:
        session_id: The session ID returned by create_session()

    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    cdef int32_t sid = session_id
    cdef char *history_json

    with nogil:
        history_json = apple_ai_get_history(sid)

    if history_json == NULL:
        return []

    try:
        history_str = _decode_string(history_json)
        return json.loads(history_str)
    finally:
        apple_ai_free_string(history_json)


def clear_history(session_id: int) -> None:
    """
    Clear a session's conversation history, keeping its instructions/tools.

    Args:
        session_id: The session ID returned by create_session()
    """
    cdef int32_t sid = session_id
    with nogil:
        apple_ai_clear_history(sid)


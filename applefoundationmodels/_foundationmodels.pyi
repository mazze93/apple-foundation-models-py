"""Type stubs for _foundationmodels Cython extension."""

from typing import Any, Callable, Dict, List, Optional

# Initialization and cleanup
def init() -> None: ...
def cleanup() -> None: ...
def get_version() -> str: ...

# Availability functions
def check_availability() -> int: ...
def get_availability_reason() -> Optional[str]: ...
def is_ready() -> bool: ...

# Session management
def create_session(
    config: Optional[Dict[str, Any]] = None,
    tools: Optional[Dict[str, Callable]] = None,
) -> int: ...
def close_session(session_id: int) -> None: ...
def get_transcript(session_id: int) -> List[Dict[str, Any]]: ...

# Text generation
def generate(
    session_id: int, prompt: str, temperature: float = 1.0, max_tokens: int = 1024
) -> str: ...

# Structured generation
def generate_structured(
    session_id: int,
    prompt: str,
    schema: Dict[str, Any],
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]: ...

# Streaming generation
def generate_stream(
    session_id: int,
    prompt: str,
    callback: Callable[[Optional[str]], None],
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> None: ...

# History management
def get_history(session_id: int) -> List[Any]: ...
def clear_history(session_id: int) -> None: ...

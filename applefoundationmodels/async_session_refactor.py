"""
AsyncSession API for applefoundationmodels Python bindings.

Provides async session management, text generation, and async streaming support.
"""

import asyncio
import inspect
import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
    overload,
)
from collections.abc import AsyncIterator, Coroutine

from typing_extensions import Literal

from .base import AsyncContextManagedResource
from .base_session import BaseSession, StreamQueueItem
from .pydantic_compat import normalize_schema
from .types import (
    GenerationResponse,
    StreamChunk,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Global registry for tools decorated with @tool
_TOOL_REGISTRY: List[Dict[str, Any]] = []


def tool(func: Callable) -> Callable:
    """
    Decorator to register a Python function as a tool for the LLM to call.

    The decorated function's name, docstring, and type hints are extracted
    to create a JSON-serializable schema that describes the tool to the model.

    Example:
        @tool
        def get_weather(location: str) -> str:
            '''Get current weather for a location.'''
            return f"Weather in {location}: 22°C"
    """
    # Extract function signature and docstring
    sig = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""
    
    # Build parameters schema from type hints
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # Skip 'self' for methods
        if param_name == 'self':
            continue
            
        # Get type annotation, default to str if not provided
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            annotation = str
            
        # Map Python types to JSON Schema types (simplified)
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        json_type = type_map.get(annotation, "string")
        
        properties[param_name] = {"type": json_type}
        
        # Check if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    # Create tool definition
    tool_def = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": docstring.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
        "_callable": func,  # Store the actual function for execution
    }
    
    # Register the tool
    _TOOL_REGISTRY.append(tool_def)
    logger.debug(f"Registered tool: {func.__name__}")
    
    return func


def get_registered_tools() -> List[Dict[str, Any]]:
    """Return a copy of the currently registered tools (without the callable)."""
    # Return tools without the internal '_callable' key for serialization
    return [
        {k: v for k, v in tool.items() if k != "_callable"}
        for tool in _TOOL_REGISTRY
    ]


class AsyncSession(BaseSession, AsyncContextManagedResource):
    """
    Async AI session for maintaining conversation state.

    AsyncSession provides async/await support for all operations including
    streaming. Use this for async applications. Sessions maintain conversation
    history and can be configured with tools and instructions.

    Usage:
        async with AsyncSession() as session:
            response = await session.generate("Hello!")
            print(response.text)

            # Async streaming
            async for chunk in session.generate("Story", stream=True):
                print(chunk.content, end='', flush=True)

        # With configuration:
        def get_weather(location: str) -> str:
            '''Get current weather for a location.'''
            return f"Weather in {location}: 22°C"

        session = AsyncSession(
            instructions="You are a helpful assistant.",
            tools=[get_weather]
        )
        response = await session.generate("What's the weather in Paris?")
        await session.aclose()
    """

    def __init__(
        self,
        instructions: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
    ):
        """
        Initialize an async session.

        Args:
            instructions: Initial system instructions for the model.
            tools: List of Python callables (functions) decorated with @tool
                   to be made available to the model.
        """
        # Process and register tools if provided
        if tools:
            for tool_func in tools:
                tool(tool_func)  # Register via decorator
        
        # Extract the actual callable functions from the registry to pass to BaseSession
        callable_tools = [tool_info["_callable"] for tool_info in _TOOL_REGISTRY]
        
        # Call parent init with the list of callable tools (maintaining existing contract)
        super().__init__(
            instructions=instructions,
            tools=callable_tools,
        )

    async def _call_ffi(self, func, *args, **kwargs):
        """Execute FFI call asynchronously via a worker thread."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def _create_stream_queue_adapter(self) -> BaseSession._StreamQueueAdapter:
        """Return the async-aware queue adapter used for streaming."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[StreamQueueItem] = asyncio.Queue()

        def push(item: StreamQueueItem) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(item), loop)

        async def get_async() -> StreamQueueItem:
            return await queue.get()

        def get_sync() -> StreamQueueItem:
            raise RuntimeError(
                "Synchronous queue access is not supported for AsyncSession"
            )

        return BaseSession._StreamQueueAdapter(
            push=push,
            get_sync=get_sync,
            get_async=get_async,
        )

    def close(self) -> None:
        """Close the session synchronously.

        When no event loop is running, this method drives the async cleanup via
        asyncio.run(). If called while an event loop is active, a RuntimeError is
        raised to avoid nested event loops—the caller should instead await
        `aclose()` or use `async with`.
        """
        if self._closed:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return

        raise RuntimeError(
            "AsyncSession.close() cannot be called while an event loop is running. "
            "Use await session.aclose() or 'async with AsyncSession()'."
        )

    async def aclose(self) -> None:
        """Asynchronously close the session and cleanup resources."""
        if self._closed:
            return
        self._mark_closed()

    # ========================================================================
    # Type overloads for generate() method
    #
    # IMPORTANT: These overloads must be kept in sync with Session.generate()
    # in session.py. The signatures are identical except for:
    # - async keyword (Session: def generate() vs AsyncSession: async def generate())
    # - Return type for streaming (Iterator vs AsyncIterator)
    #
    # When modifying these overloads, update both files to maintain consistency.
    # ========================================================================

    # Type overload for non-streaming text generation
    @overload
    def generate(
        self,
        prompt: str,
        schema: None = None,
        stream: Literal[False] = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Coroutine[Any, Any, GenerationResponse]: ...

    # Type overload for non-streaming structured generation
    @overload
    def generate(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        stream: Literal[False] = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Coroutine[Any, Any, GenerationResponse]: ...

    # Type overload for async streaming generation (text only)
    @overload
    def generate(
        self,
        prompt: str,
        schema: None = None,
        stream: Literal[True] = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[StreamChunk]: ...

    def generate(
        self,
        prompt: str,
        schema: Optional[Union[Dict[str, Any], Type["BaseModel"]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[Coroutine[Any, Any, GenerationResponse], AsyncIterator[StreamChunk]]:
        """
        Generate text or structured output asynchronously, with optional streaming
        and automatic tool calling.

        This unified async method supports three generation modes:
        1. Text generation (schema=None, stream=False) -> GenerationResponse
        2. Structured generation (schema=dict/model, stream=False) -> GenerationResponse
        3. Async streaming (schema=None, stream=True) -> AsyncIterator[StreamChunk]

        When tools are registered, the model can request to call them. The session
        will automatically execute the requested tool and feed the result back
        into the conversation for the model's next turn.

        Args:
            prompt: Input text prompt
            schema: Optional JSON schema dict or Pydantic model for structured output
            stream: If True, return an async iterator of chunks
            temperature: Sampling temperature (0.0-2.0, default: DEFAULT_TEMPERATURE)
            max_tokens: Maximum tokens to generate (default: DEFAULT_MAX_TOKENS)

        Returns:
            - GenerationResponse with .text or .parsed property (if stream=False)
            - AsyncIterator[StreamChunk] yielding content deltas (if stream=True)

        Raises:
            RuntimeError: If session is closed
            GenerationError: If generation fails
            ValueError: If schema is provided with stream=True

        Examples:
            Text generation:
                >>> response = await session.generate("What is Python?")
                >>> print(response.text)

            Structured generation:
                >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
                >>> response = await session.generate("Extract name: John", schema=schema)
                >>> print(response.parsed)

            Async streaming:
                >>> async for chunk in session.generate("Tell a story", stream=True):
                ...     print(chunk.content, end='', flush=True)
        """
        plan = self._plan_generate_call(stream, schema, temperature, max_tokens)

        if plan.mode == "stream":
            return self._generate_stream_impl(prompt, plan.temperature, plan.max_tokens)
        if plan.mode == "structured" and schema is not None:
            return self._generate_structured_impl(
                prompt, schema, plan.temperature, plan.max_tokens
            )

        return self._generate_text_with_tools_impl(prompt, plan.temperature, plan.max_tokens)

    async def _generate_text_with_tools_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> GenerationResponse:
        """
        Internal implementation for async text generation with tool calling loop.
        """
        # Start generation context to track transcript length
        async with self._async_generation_context() as start_length:
            # Tool calling loop - prevents infinite loops with a reasonable limit
            max_tool_iterations = 10
            iteration_count = 0
            
            while iteration_count < max_tool_iterations:
                iteration_count += 1
                
                # Generate response from model
                text = await self._call_ffi(
                    self._ffi.generate,
                    self._session_id,
                    prompt,
                    temperature,
                    max_tokens,
                )
                
                # Check if the response contains a tool call request
                tool_call = self._parse_tool_call(text)
                
                if tool_call is not None:
                    # Execute the requested tool
                    tool_result = await self._execute_tool(tool_call)
                    
                    # Append tool result to conversation history for next iteration
                    # Format as a tool message for the model
                    tool_message = f"\n\nTool {tool_call['name']} returned: {json.dumps(tool_result)}\n\n"
                    prompt = prompt + tool_message
                    
                    # Continue loop to let model process the tool result
                    continue
                else:
                    # No tool call detected, return final response
                    return self._build_generation_response(text, False, start_length)
            
            # If we exceeded max iterations, return what we have
            logger.warning(f"Tool calling loop exceeded {max_tool_iterations} iterations")
            return self._build_generation_response(text, False, start_length)

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse model output to detect a tool call request.
        
        Expected format: JSON object with "name" and "arguments" fields.
        Example: {"name": "get_weather", "arguments": {"location": "Paris"}}
        """
        try:
            # Look for JSON-like patterns in the text
            # This is a simple implementation - could be enhanced with more robust parsing
            text = text.strip()
            if text.startswith('{') and text.endswith('}'):
                data = json.loads(text)
                if isinstance(data, dict) and 'name' in data and 'arguments' in data:
                    # Validate that the tool exists in our registry
                    tool_name = data['name']
                    for tool in _TOOL_REGISTRY:
                        if tool['function']['name'] == tool_name:
                            return {
                                'name': tool_name,
                                'arguments': data['arguments']
                            }
                    logger.warning(f"Unknown tool requested: {tool_name}")
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        """
        Execute a registered tool with the given arguments.
        """
        tool_name = tool_call['name']
        arguments = tool_call['arguments']
        
        # Find the tool in registry
        for tool in _TOOL_REGISTRY:
            if tool['function']['name'] == tool_name:
                func = tool['_callable']
                try:
                    # Call the function with provided arguments
                    if inspect.iscoroutinefunction(func):
                        result = await func(**arguments)
                    else:
                        result = func(**arguments)
                    logger.debug(f"Tool {tool_name} executed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    return {"error": str(e)}
        
        raise ValueError(f"Tool {tool_name} not found in registry")

    async def _generate_text_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> GenerationResponse:
        """Internal implementation for async text generation."""
        async with self._async_generation_context() as start_length:
            text = await self._call_ffi(
                self._ffi.generate,
                self._session_id,
                prompt,
                temperature,
                max_tokens,
            )
            return self._build_generation_response(text, False, start_length)

    async def _generate_structured_impl(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"]],
        temperature: float,
        max_tokens: int,
    ) -> GenerationResponse:
        """Internal implementation for async structured generation."""
        async with self._async_generation_context() as start_length:
            json_schema = normalize_schema(schema)
            result = await self._call_ffi(
                self._ffi.generate_structured,
                self._session_id,
                prompt,
                json_schema,
                temperature,
                max_tokens,
            )
            return self._build_generation_response(result, True, start_length)

    async def _generate_stream_impl(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> AsyncIterator[StreamChunk]:
        """Internal implementation for async streaming generation."""
        start_length = self._begin_generation()
        adapter = self._create_stream_queue_adapter()
        try:
            async for chunk in self._stream_chunks_async_impl(
                prompt, temperature, max_tokens, adapter
            ):
                yield chunk
        finally:
            self._end_generation(start_length)

    async def get_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history asynchronously.

        Returns:
            List of message dictionaries with 'role' and 'content' keys

        Example:
            >>> history = await session.get_history()
            >>> for msg in history:
            ...     print(f"{msg['role']}: {msg['content']}")
        """
        self._check_closed()
        result = await self._call_ffi(self._ffi.get_history, self._session_id)
        return cast(List[Dict[str, Any]], result)

    async def clear_history(self) -> None:
        """
        Clear conversation history asynchronously.

        Removes all messages from the session while keeping the session active.
        """
        self._check_closed()
        await self._call_ffi(self._ffi.clear_history, self._session_id)
        # Reset to current transcript length (may include persistent instructions)
        self._last_transcript_length = len(self.transcript)
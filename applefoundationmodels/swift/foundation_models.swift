/**
 * foundation_models.swift
 *
 * Swift bindings for FoundationModels framework
 * Exports C-compatible API for Python/Cython FFI
 */

import Foundation

#if canImport(FoundationModels)
import FoundationModels
#endif

// MARK: - Global State

private var isInitialized = false

/// A single native session's identity and the parameters it was created
/// with, kept alongside the session so `clear_history` can rebuild an
/// equivalent session (Apple's API has no in-place history reset) without
/// losing that session's own instructions/tools.
@available(macOS 26.0, *)
private struct SessionEntry {
    var session: LanguageModelSession
    var instructions: String?
    var tools: [any Tool]
    /// True while a generate/generate_stream/generate_structured call is
    /// in flight on this session. Apple's `LanguageModelSession` itself
    /// guards a *second* concurrent `respond()` cleanly (raises
    /// `concurrentRequests`), but a second concurrent `streamResponse()`
    /// was found to hang the process instead of erroring - rejecting the
    /// second call before it ever reaches the session avoids that
    /// regardless of which call kind race occurs.
    var busy: Bool = false
}

/// Registry of every live session, keyed by the id handed back from
/// `apple_ai_create_session`. Each entry is fully independent: no state is
/// shared between sessions. Guarded by `sessionRegistryLock` since Python
/// can call in from multiple OS threads (sync streaming runs on a
/// background thread per call).
@available(macOS 26.0, *)
private var sessionRegistry: [Int32: SessionEntry] = [:]
private let sessionRegistryLock = NSLock()
private var nextSessionID: Int32 = 1

@available(macOS 26.0, *)
private func allocateSessionID(for entry: SessionEntry) -> Int32 {
    sessionRegistryLock.lock()
    defer { sessionRegistryLock.unlock() }
    let id = nextSessionID
    nextSessionID += 1
    sessionRegistry[id] = entry
    return id
}

@available(macOS 26.0, *)
private func getSessionEntry(_ sessionID: Int32) -> SessionEntry? {
    sessionRegistryLock.lock()
    defer { sessionRegistryLock.unlock() }
    return sessionRegistry[sessionID]
}

/// Atomically check-and-set a session's `busy` flag before starting a
/// generate/generate_stream/generate_structured call, so two concurrent
/// calls on the *same* session id can never both proceed. Pair with
/// `endRequest(_:)` (typically via `defer`) once the call completes,
/// success or failure.
@available(macOS 26.0, *)
private func beginRequest(_ sessionID: Int32) -> Result<LanguageModelSession, AIResult> {
    sessionRegistryLock.lock()
    defer { sessionRegistryLock.unlock() }
    guard var entry = sessionRegistry[sessionID] else {
        return .failure(.errorSessionNotFound)
    }
    if entry.busy {
        return .failure(.errorConcurrentRequests)
    }
    entry.busy = true
    sessionRegistry[sessionID] = entry
    return .success(entry.session)
}

@available(macOS 26.0, *)
private func endRequest(_ sessionID: Int32) {
    sessionRegistryLock.lock()
    defer { sessionRegistryLock.unlock() }
    if var entry = sessionRegistry[sessionID] {
        entry.busy = false
        sessionRegistry[sessionID] = entry
    }
}

@available(macOS 26.0, *)
private func replaceSessionEntry(_ sessionID: Int32, with entry: SessionEntry) {
    sessionRegistryLock.lock()
    defer { sessionRegistryLock.unlock() }
    sessionRegistry[sessionID] = entry
}

private func removeSessionEntry(_ sessionID: Int32) {
    sessionRegistryLock.lock()
    defer { sessionRegistryLock.unlock() }
    sessionRegistry.removeValue(forKey: sessionID)
}

private func clearSessionRegistry() {
    sessionRegistryLock.lock()
    defer { sessionRegistryLock.unlock() }
    sessionRegistry.removeAll()
    nextSessionID = 1
}

// MARK: - Error Codes

public enum AIResult: Int32, CaseIterable, Error {
    case success = 0
    case errorInitFailed = -1
    case errorNotAvailable = -2
    case errorInvalidParams = -3
    case errorMemory = -4
    case errorJSONParse = -5
    case errorGeneration = -6
    case errorTimeout = -7
    case errorSessionNotFound = -8
    case errorGuardrailViolation = -10
    case errorToolNotFound = -11
    case errorToolExecution = -12
    case errorBufferTooSmall = -13
    case errorContextWindowExceeded = -14
    case errorDecodingFailure = -15
    case errorRateLimited = -16
    case errorRefusal = -17
    case errorConcurrentRequests = -18
    case errorUnsupportedGuide = -19
    case errorUnsupportedLanguage = -20
    case errorAssetsUnavailable = -21
    case errorUnknown = -99
}

public enum AIAvailability: Int32 {
    case available = 1
    case deviceNotEligible = -1
    case notEnabled = -2
    case modelNotReady = -3
    case unknown = -99
}

// MARK: - Tool Calling Infrastructure

/// C-compatible callback for Python tool execution
public typealias ToolCallback = @convention(c) (
    Int32,  // session_id
    UnsafePointer<CChar>?,  // tool_name
    UnsafePointer<CChar>?,  // arguments_json
    UnsafeMutablePointer<CChar>?,  // result_buffer
    Int32  // buffer_size
) -> Int32

/// Store tool callback globally with thread-safe access
private let toolCallbackLock = NSLock()
private var _toolCallback: ToolCallback?

private var toolCallback: ToolCallback? {
    get {
        toolCallbackLock.lock()
        defer { toolCallbackLock.unlock() }
        return _toolCallback
    }
    set {
        toolCallbackLock.lock()
        defer { toolCallbackLock.unlock() }
        _toolCallback = newValue
    }
}

/// Python tool wrapper that bridges Swift Tool protocol to Python callbacks
@available(macOS 26.0, *)
struct PythonToolWrapper: Tool, Sendable {
    let sessionID: Int32
    let toolName: String
    let toolDescription: String
    let dynamicSchema: DynamicGenerationSchema

    var name: String { toolName }
    var description: String { toolDescription }

    // Use GeneratedContent as Arguments - it's already Generable and supports dynamic schemas!
    typealias Arguments = GeneratedContent
    typealias Output = String

    // Override parameters with our dynamic schema
    var parameters: GenerationSchema {
        // Must not throw - create schema or use empty one on error
        (try? GenerationSchema(root: dynamicSchema, dependencies: [])) ?? GenerationSchema(
            type: GeneratedContent.self,
            properties: []
        )
    }

    nonisolated func call(arguments: Arguments) async throws -> Output {
        // Extract JSON directly from GeneratedContent!
        let argsJson = arguments.jsonString

        // Call Python callback
        guard let callback = toolCallback else {
            throw NSError(domain: "ToolError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Tool callback not set"])
        }

        // Buffer size configuration
        let initialBufferSize: Int32 = 16384  // 16KB initial
        let maxBufferSize: Int32 = 1048576    // 1MB max cap
        var bufferSize: Int32 = initialBufferSize

        // Retry loop with progressively larger buffers
        while bufferSize <= maxBufferSize {
            let resultBuffer = UnsafeMutablePointer<CChar>.allocate(capacity: Int(bufferSize))
            resultBuffer.initialize(repeating: 0, count: Int(bufferSize))
            defer { resultBuffer.deallocate() }

            let result = argsJson.withCString { argsPtr in
                toolName.withCString { namePtr in
                    callback(sessionID, namePtr, argsPtr, resultBuffer, bufferSize)
                }
            }

            // Success - return result
            if result == 0 {
                return String(cString: resultBuffer)
            }

            // Buffer too small - retry with larger buffer
            if result == AIResult.errorBufferTooSmall.rawValue {
                // Double the buffer size and retry
                let newSize = bufferSize * 2
                if newSize > maxBufferSize {
                    throw NSError(domain: "ToolError", code: Int(result), userInfo: [
                        NSLocalizedDescriptionKey: "Tool '\(toolName)' output exceeds maximum buffer size (\(maxBufferSize) bytes)"
                    ])
                }
                bufferSize = newSize
                continue
            }

            // Other error - extract message and throw
            let errorMsg = String(cString: resultBuffer)
            throw NSError(domain: "ToolError", code: Int(result), userInfo: [
                NSLocalizedDescriptionKey: "Tool '\(toolName)' failed: \(errorMsg)"
            ])
        }

        // Should never reach here due to buffer size check above
        throw NSError(domain: "ToolError", code: -4, userInfo: [
            NSLocalizedDescriptionKey: "Tool '\(toolName)' output exceeds maximum buffer size"
        ])
    }
}

// MARK: - Helper Functions

/// Create a fallback error JSON string
private func fallbackErrorJSON(code: AIResult = .errorUnknown) -> String {
    return "{\"error\":\"An error occurred\",\"error_code\":\(code.rawValue)}"
}

/// Create an error response in JSON format using safe serialization
private func createErrorResponse(_ message: String, errorCode: AIResult = .errorGeneration) -> UnsafeMutablePointer<CChar>? {
    let errorDict: [String: Any] = [
        "error": message,
        "error_code": errorCode.rawValue
    ]

    do {
        let jsonData = try JSONSerialization.data(withJSONObject: errorDict, options: [])
        if let jsonString = String(data: jsonData, encoding: .utf8) {
            return strdup(jsonString)
        }
    } catch {
        // Fallback to generic error message if serialization fails
        return strdup(fallbackErrorJSON())
    }

    // If UTF-8 encoding fails, return generic error
    return strdup(fallbackErrorJSON())
}

#if canImport(FoundationModels)
/// Map LanguageModelSession.GenerationError to our error codes
@available(macOS 26.0, *)
private func mapGenerationErrorToCode(_ error: Error) -> AIResult {
    guard let genError = error as? LanguageModelSession.GenerationError else {
        return .errorGeneration
    }

    switch genError {
    case .exceededContextWindowSize:
        return .errorContextWindowExceeded
    case .guardrailViolation:
        return .errorGuardrailViolation
    case .assetsUnavailable:
        return .errorAssetsUnavailable
    case .decodingFailure:
        return .errorDecodingFailure
    case .rateLimited:
        return .errorRateLimited
    case .refusal:
        return .errorRefusal
    case .concurrentRequests:
        return .errorConcurrentRequests
    case .unsupportedGuide:
        return .errorUnsupportedGuide
    case .unsupportedLanguageOrLocale:
        return .errorUnsupportedLanguage
    @unknown default:
        return .errorGeneration
    }
}
#endif

/// Build a brand-new, independent `LanguageModelSession` for the given
/// instructions/tools. Pure - never touches the session registry, so it's
/// safe to use both for first creation and for rebuilding a session with
/// the same parameters (e.g. `clear_history`).
@available(macOS 26.0, *)
private func buildSession(
    instructions: String?,
    tools: [any Tool]
) -> LanguageModelSession {
    let session: LanguageModelSession
    switch (instructions, tools.isEmpty) {
    case (let inst?, false):
        session = LanguageModelSession(
            model: SystemLanguageModel.default,
            tools: tools,
            instructions: { inst }
        )
    case (let inst?, true):
        session = LanguageModelSession(
            model: SystemLanguageModel.default,
            instructions: { inst }
        )
    case (nil, false):
        session = LanguageModelSession(
            model: SystemLanguageModel.default,
            tools: tools
        )
    case (nil, true):
        session = LanguageModelSession(
            model: SystemLanguageModel.default
        )
    }
    return session
}

// MARK: - Initialization

@_cdecl("apple_ai_init")
public func appleAIInit() -> Int32 {
    guard !isInitialized else {
        return AIResult.success.rawValue
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        // Check if model is available
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            isInitialized = true
            return AIResult.success.rawValue
        case .unavailable:
            return AIResult.errorNotAvailable.rawValue
        }
    }
    #endif

    return AIResult.errorNotAvailable.rawValue
}

@_cdecl("apple_ai_cleanup")
public func appleAICleanup() {
    clearSessionRegistry()
    isInitialized = false
}

// MARK: - Availability Check

@_cdecl("apple_ai_check_availability")
public func appleAICheckAvailability() -> Int32 {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            return AIAvailability.available.rawValue
        case .unavailable(let reason):
            // Map unavailability reason to status code
            let description = String(describing: reason)
            if description.contains("not enabled") || description.contains("disabled") {
                return AIAvailability.notEnabled.rawValue
            } else if description.contains("downloading") || description.contains("not ready") {
                return AIAvailability.modelNotReady.rawValue
            } else {
                return AIAvailability.deviceNotEligible.rawValue
            }
        }
    } else {
        return AIAvailability.deviceNotEligible.rawValue
    }
    #else
    return AIAvailability.deviceNotEligible.rawValue
    #endif
}

@_cdecl("apple_ai_get_availability_reason")
public func appleAIGetAvailabilityReason() -> UnsafeMutablePointer<CChar>? {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            return strdup("Apple Intelligence is available and ready")
        case .unavailable(let reason):
            return strdup("Apple Intelligence is unavailable: \(reason)")
        }
    } else {
        return strdup("Device does not support Apple Intelligence (requires macOS 26.0+)")
    }
    #else
    return strdup("FoundationModels framework not available")
    #endif
}

@_cdecl("apple_ai_get_version")
public func appleAIGetVersion() -> UnsafeMutablePointer<CChar>? {
    return strdup("1.0.0-foundationmodels")
}

// MARK: - Session Management

/// Parse a JSON array of `{name, description, parameters}` tool
/// definitions into `PythonToolWrapper`s bound to `sessionID`. Returns
/// `.success([])` for a nil/empty `toolsJson`.
@available(macOS 26.0, *)
private func parseTools(
    toolsJson: UnsafePointer<CChar>?,
    sessionID: Int32
) -> Result<[any Tool], AIResult> {
    guard let jsonPtr = toolsJson else {
        return .success([])
    }

    let jsonString = String(cString: jsonPtr)
    guard let jsonData = jsonString.data(using: .utf8),
          let toolsArray = try? JSONSerialization.jsonObject(with: jsonData, options: []) as? [[String: Any]] else {
        return .failure(.errorJSONParse)
    }

    var tools: [any Tool] = []
    for (index, toolDef) in toolsArray.enumerated() {
        guard let name = toolDef["name"] as? String else {
            print("ERROR: Tool at index \(index) missing required 'name' field")
            return .failure(.errorInvalidParams)
        }
        guard let description = toolDef["description"] as? String else {
            print("ERROR: Tool '\(name)' at index \(index) missing required 'description' field")
            return .failure(.errorInvalidParams)
        }
        guard let parameters = toolDef["parameters"] as? [String: Any] else {
            print("ERROR: Tool '\(name)' at index \(index) missing required 'parameters' field")
            return .failure(.errorInvalidParams)
        }

        let conversionResult = convertJSONSchemaToDynamic(parameters, name: "\(name)_params")
        guard case .success(let dynamicSchema) = conversionResult else {
            if case .failure(let error) = conversionResult {
                print("ERROR: \(error.message)")
            }
            return .failure(.errorJSONParse)
        }

        tools.append(PythonToolWrapper(
            sessionID: sessionID,
            toolName: name,
            toolDescription: description,
            dynamicSchema: dynamicSchema
        ))
    }

    return .success(tools)
}

/// Create a brand-new, fully independent session. Unlike the pre-registry
/// design, nothing here is shared with any other session: its instructions,
/// its tools, and its conversation state all live only under the returned
/// session_id.
/// - Returns: A positive session_id on success, or a negative AIResult error code.
@_cdecl("apple_ai_create_session")
public func appleAICreateSession(
    instructionsJson: UnsafePointer<CChar>?,
    toolsJson: UnsafePointer<CChar>?,
    callback: ToolCallback?
) -> Int32 {
    guard isInitialized else {
        return AIResult.errorInitFailed.rawValue
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        var instructions: String? = nil
        if let jsonPtr = instructionsJson {
            let jsonString = String(cString: jsonPtr)
            if let jsonData = jsonString.data(using: .utf8),
               let config = try? JSONDecoder().decode([String: String].self, from: jsonData),
               let inst = config["instructions"] {
                instructions = inst
            }
        }

        if toolsJson != nil {
            guard let callback = callback else {
                return AIResult.errorInvalidParams.rawValue
            }
            // The dispatcher itself is stateless (session_id disambiguates
            // on every call), so one process-wide function pointer is fine.
            toolCallback = callback
        }

        // Reserve the id first (tools need it to route callbacks), then
        // parse tools against it, then register the entry only once the
        // session is fully built - a failure never leaves a partial entry.
        let sessionID = allocateSessionID(for: SessionEntry(
            session: buildSession(instructions: instructions, tools: []),
            instructions: instructions,
            tools: []
        ))

        let toolsResult = parseTools(toolsJson: toolsJson, sessionID: sessionID)
        guard case .success(let tools) = toolsResult else {
            removeSessionEntry(sessionID)
            if case .failure(let code) = toolsResult {
                return code.rawValue
            }
            return AIResult.errorInvalidParams.rawValue
        }

        let session = tools.isEmpty
            ? getSessionEntry(sessionID)!.session
            : buildSession(instructions: instructions, tools: tools)
        replaceSessionEntry(sessionID, with: SessionEntry(
            session: session,
            instructions: instructions,
            tools: tools
        ))

        return sessionID
    }
    #endif

    return AIResult.errorNotAvailable.rawValue
}

/// Release a session's native resources. Safe to call more than once or on
/// an id that was never valid (both are no-ops that report success).
@_cdecl("apple_ai_close_session")
public func appleAICloseSession(sessionID: Int32) -> Int32 {
    removeSessionEntry(sessionID)
    return AIResult.success.rawValue
}

// MARK: - Generation

/// Turn a `beginRequest` failure into the message for `createErrorResponse`.
private func beginRequestErrorMessage(_ code: AIResult, sessionID: Int32) -> String {
    switch code {
    case .errorConcurrentRequests:
        return "Session \(sessionID) is already responding to a previous prompt. " +
            "Wait for it to finish before starting another generate() call on the same session."
    default:
        return "Session \(sessionID) not found"
    }
}

/// Generate text response
/// - Parameters:
///   - prompt: User prompt as C string
///   - temperature: Sampling temperature (0.0 to 2.0)
///   - maxTokens: Maximum tokens to generate
/// - Returns: JSON response or error message
@_cdecl("apple_ai_generate")
public func appleAIGenerate(
    sessionID: Int32,
    prompt: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int32
) -> UnsafeMutablePointer<CChar>? {
    guard isInitialized else {
        return createErrorResponse("Not initialized")
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let requestResult = beginRequest(sessionID)
        guard case .success(let session) = requestResult else {
            guard case .failure(let code) = requestResult else { return createErrorResponse("Unknown error") }
            return createErrorResponse(beginRequestErrorMessage(code, sessionID: sessionID), errorCode: code)
        }
        defer { endRequest(sessionID) }

        let promptString = String(cString: prompt)

        // Use semaphore for async coordination
        let semaphore = DispatchSemaphore(value: 0)
        var result: String = ""

        Task {
            do {
                // Configure generation options
                let options = GenerationOptions(
                    temperature: temperature,
                    maximumResponseTokens: Int(maxTokens)
                )

                // Generate response
                let response = try await session.respond(
                    to: promptString,
                    options: options
                )

                result = response.content
            } catch {
                // Map error to specific error code
                let errorCode = mapGenerationErrorToCode(error)

                // Use safe JSON serialization for error messages
                if let errorJson = createErrorResponse(error.localizedDescription, errorCode: errorCode) {
                    result = String(cString: errorJson)
                    free(errorJson)
                } else {
                    result = fallbackErrorJSON(code: errorCode)
                }
            }
            semaphore.signal()
        }

        semaphore.wait()
        return strdup(result)
    }
    #endif

    return createErrorResponse("FoundationModels not available")
}

// Streaming callback type. session_id lets a caller running several
// concurrently-streaming sessions route each chunk to the right consumer.
public typealias StreamCallback = @convention(c) (Int32, UnsafePointer<CChar>?) -> Void

/// Generate streaming text response
/// - Parameters:
///   - sessionID: The session to generate on
///   - prompt: User prompt as C string
///   - temperature: Sampling temperature (0.0 to 2.0)
///   - maxTokens: Maximum tokens to generate
///   - callback: Callback function to receive text chunks (receives nil to signal end)
/// - Returns: Result code (0 = success, negative = error)
@_cdecl("apple_ai_generate_stream")
public func appleAIGenerateStream(
    sessionID: Int32,
    prompt: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int32,
    callback: StreamCallback?
) -> Int32 {
    guard isInitialized, let cb = callback else {
        return AIResult.errorInvalidParams.rawValue
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let requestResult = beginRequest(sessionID)
        guard case .success(let session) = requestResult else {
            // Deliberately don't invoke the callback here: the Cython
            // wrapper raises from this function's *return code* via
            // _check_result(), and the caller only distinguishes "stream
            // ended cleanly" (a nil chunk) from "stream errored" by whether
            // an exception ever reaches it - sending a nil chunk first
            // would make the generator stop before that exception surfaces,
            // turning a real error into a silently-truncated success.
            if case .failure(let code) = requestResult {
                return code.rawValue
            }
            return AIResult.errorUnknown.rawValue
        }
        defer { endRequest(sessionID) }

        let promptString = String(cString: prompt)

        let semaphore = DispatchSemaphore(value: 0)
        var resultCode = AIResult.success

        Task {
            do {
                // Configure generation options
                let options = GenerationOptions(
                    temperature: temperature,
                    maximumResponseTokens: Int(maxTokens)
                )

                // Stream response
                let stream = try await session.streamResponse(
                    options: options
                ) {
                    promptString
                }

                var previousContent = ""
                for try await partial in stream {
                    let currentContent = partial.content

                    // Calculate delta from previous snapshot
                    if currentContent.count > previousContent.count {
                        let delta = String(currentContent.dropFirst(previousContent.count))
                        if !delta.isEmpty {
                            cb(sessionID, strdup(delta))
                        }
                    }

                    previousContent = currentContent
                }

                // Signal end of stream
                cb(sessionID, nil)

            } catch {
                // Map error to specific error code
                let errorCode = mapGenerationErrorToCode(error)

                // Use safe JSON serialization for error messages
                let errorMessage = "Error: \(error.localizedDescription)"
                if let errorJson = createErrorResponse(errorMessage, errorCode: errorCode) {
                    cb(sessionID, errorJson)
                    // Note: callback takes ownership, will be freed by caller
                } else {
                    cb(sessionID, strdup(fallbackErrorJSON(code: errorCode)))
                }
                cb(sessionID, nil)
                resultCode = errorCode
            }
            semaphore.signal()
        }

        semaphore.wait()
        return resultCode.rawValue
    }
    #endif

    cb(sessionID, strdup("FoundationModels not available"))
    cb(sessionID, nil)
    return AIResult.errorNotAvailable.rawValue
}

// MARK: - Transcript Access

/// Get the session transcript
/// - Returns: JSON array of transcript entries or error message
@_cdecl("apple_ai_get_transcript")
public func appleAIGetTranscript(sessionID: Int32) -> UnsafeMutablePointer<CChar>? {
    guard isInitialized else {
        return createErrorResponse("Not initialized")
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        guard let entry = getSessionEntry(sessionID) else {
            return createErrorResponse("Session \(sessionID) not found", errorCode: .errorSessionNotFound)
        }
        let session = entry.session

        // Use semaphore for async coordination
        let semaphore = DispatchSemaphore(value: 0)
        var result: String = ""

        Task {
            do {
                let transcript = session.transcript
                var entries: [NSDictionary] = []

                for entry in transcript {
                    var entryDict: [String: Any] = [:]

                    switch entry {
                    case .instructions(let text):
                        entryDict["type"] = "instructions" as NSString
                        entryDict["content"] = String(describing: text) as NSString

                    case .prompt(let text):
                        entryDict["type"] = "prompt" as NSString
                        entryDict["content"] = String(describing: text) as NSString

                    case .response(let text):
                        entryDict["type"] = "response" as NSString
                        entryDict["content"] = String(describing: text) as NSString

                    case .toolCalls(let toolCalls):
                        // Create individual entries for each tool call
                        // API: ToolCall exposes id, toolName, and arguments (GeneratedContent)
                        for call in toolCalls {
                            var callDict: [String: Any] = [:]
                            callDict["type"] = "tool_call" as NSString
                            callDict["tool_id"] = String(describing: call.id) as NSString
                            callDict["tool_name"] = call.toolName as NSString
                            // Serialize arguments to JSON string via GeneratedContent.jsonString
                            callDict["arguments"] = call.arguments.jsonString as NSString
                            entries.append(callDict as NSDictionary)
                        }
                        // Skip adding entryDict since we added individual entries
                        continue

                    case .toolOutput(let output):
                        entryDict["type"] = "tool_output" as NSString
                        entryDict["tool_id"] = String(describing: output.id) as NSString

                        // Extract content from segments
                        // API: ToolOutput exposes id, toolName, and segments (array of Segment)
                        // Note: FoundationModels API uses segments rather than direct content property
                        var contentParts: [String] = []
                        for segment in output.segments {
                            switch segment {
                            case .text(let textSegment):
                                contentParts.append(textSegment.content)
                            case .structure(let structuredSegment):
                                // For structured segments, use the JSON representation
                                contentParts.append(structuredSegment.content.jsonString)
                            @unknown default:
                                break
                            }
                        }
                        entryDict["content"] = contentParts.joined() as NSString

                    @unknown default:
                        entryDict["type"] = "unknown" as NSString
                    }

                    entries.append(entryDict as NSDictionary)
                }

                // Convert to JSON
                let jsonData = try JSONSerialization.data(withJSONObject: entries, options: .prettyPrinted)
                result = String(data: jsonData, encoding: .utf8) ?? "{\"error\":\"Failed to encode transcript\"}"
            } catch {
                if let errorJson = createErrorResponse(error.localizedDescription) {
                    result = String(cString: errorJson)
                    free(errorJson)
                } else {
                    result = "{\"error\":\"An error occurred\"}"
                }
            }
            semaphore.signal()
        }

        semaphore.wait()
        return strdup(result)
    }
    #endif

    return createErrorResponse("FoundationModels not available")
}

// MARK: - Structured Generation

// Error type for schema conversion with full context
struct SchemaConversionError: Error {
    let path: [String]
    let reason: String

    var message: String {
        let pathStr = path.isEmpty ? "root" : path.joined(separator: ".")
        return "Schema conversion failed at '\(pathStr)': \(reason)"
    }
}

// Helper to convert primitive type to DynamicGenerationSchema
@available(macOS 26.0, *)
private func convertPrimitiveType(
    _ type: String,
    schema: [String: Any],
    name: String
) -> DynamicGenerationSchema? {
    switch type {
    case "string":
        if let enumValues = schema["enum"] as? [String] {
            return DynamicGenerationSchema(name: name, anyOf: enumValues)
        }
        return DynamicGenerationSchema(type: String.self)
    case "integer", "number":
        return DynamicGenerationSchema(type: Double.self)
    case "boolean":
        return DynamicGenerationSchema(type: Bool.self)
    default:
        return nil
    }
}

// Helper to convert object type to DynamicGenerationSchema
@available(macOS 26.0, *)
private func convertObjectType(
    schema: [String: Any],
    name: String,
    path: [String]
) -> Result<DynamicGenerationSchema, SchemaConversionError> {
    guard let properties = schema["properties"] as? [String: [String: Any]] else {
        return .failure(SchemaConversionError(
            path: path,
            reason: "Object type missing 'properties' field"
        ))
    }

    var dynamicProperties: [DynamicGenerationSchema.Property] = []

    for (propName, propSchema) in properties {
        let propPath = path + [propName]

        // Recursive call with proper error propagation
        let result = convertJSONSchemaToDynamic(propSchema, name: propName, path: propPath)

        switch result {
        case .success(let propDynamicSchema):
            let description = propSchema["description"] as? String
            dynamicProperties.append(
                DynamicGenerationSchema.Property(
                    name: propName,
                    description: description,
                    schema: propDynamicSchema
                )
            )
        case .failure(let error):
            // Propagate error with full context
            return .failure(error)
        }
    }

    return .success(DynamicGenerationSchema(
        name: name,
        description: schema["description"] as? String,
        properties: dynamicProperties
    ))
}

// Helper to convert array type to DynamicGenerationSchema
@available(macOS 26.0, *)
private func convertArrayType(
    schema: [String: Any],
    name: String,
    path: [String]
) -> Result<DynamicGenerationSchema, SchemaConversionError> {
    guard let items = schema["items"] as? [String: Any] else {
        return .failure(SchemaConversionError(
            path: path + ["items"],
            reason: "Array type missing 'items' specification"
        ))
    }

    let itemPath = path + ["items"]
    let result = convertJSONSchemaToDynamic(items, name: "\(name)Item", path: itemPath)

    switch result {
    case .success(let itemSchema):
        let minItems = schema["minItems"] as? Int
        let maxItems = schema["maxItems"] as? Int

        return .success(DynamicGenerationSchema(
            arrayOf: itemSchema,
            minimumElements: minItems,
            maximumElements: maxItems
        ))
    case .failure(let error):
        return .failure(error)
    }
}

// Helper to convert JSON Schema dictionary to DynamicGenerationSchema with error context
@available(macOS 26.0, *)
private func convertJSONSchemaToDynamic(
    _ schema: [String: Any],
    name: String = "root",
    path: [String] = []
) -> Result<DynamicGenerationSchema, SchemaConversionError> {
    // Validate type exists
    guard let type = schema["type"] as? String else {
        return .failure(SchemaConversionError(
            path: path,
            reason: "Missing required 'type' field"
        ))
    }

    // Handle primitive types first
    if let primitiveSchema = convertPrimitiveType(type, schema: schema, name: name) {
        return .success(primitiveSchema)
    }

    // Handle complex types
    switch type {
    case "object":
        return convertObjectType(schema: schema, name: name, path: path)
    case "array":
        return convertArrayType(schema: schema, name: name, path: path)
    default:
        return .failure(SchemaConversionError(
            path: path,
            reason: "Unsupported type '\(type)'"
        ))
    }
}

// Helper to extract structure (dictionary) from GeneratedContent properties
@available(macOS 26.0, *)
private func extractStructure(from properties: [String: GeneratedContent]) throws -> [String: Any] {
    var result: [String: Any] = [:]
    for (key, value) in properties {
        result[key] = try extractValue(from: value)
    }
    return result
}

// Helper to extract array from GeneratedContent items
@available(macOS 26.0, *)
private func extractArray(from items: [GeneratedContent]) throws -> [Any] {
    return try items.map { try extractValue(from: $0) }
}

// Helper to extract Any value from GeneratedContent
@available(macOS 26.0, *)
private func extractValue(from content: GeneratedContent) throws -> Any {
    switch content.kind {
    case .string(let str):
        return str
    case .number(let num):
        return num
    case .bool(let bool):
        return bool
    case .null:
        return NSNull()
    case .structure(let properties, _):
        return try extractStructure(from: properties)
    case .array(let items):
        return try extractArray(from: items)
    @unknown default:
        throw NSError(domain: "FoundationModels", code: -1, userInfo: [NSLocalizedDescriptionKey: "Unsupported GeneratedContent kind"])
    }
}

/// Generate structured output conforming to JSON Schema
/// - Parameters:
///   - prompt: User prompt as C string
///   - schemaJson: JSON Schema as C string
///   - temperature: Sampling temperature (0.0 to 2.0)
///   - maxTokens: Maximum tokens to generate
/// - Returns: JSON object conforming to schema, or error message
@_cdecl("apple_ai_generate_structured")
public func appleAIGenerateStructured(
    sessionID: Int32,
    prompt: UnsafePointer<CChar>,
    schemaJson: UnsafePointer<CChar>,
    temperature: Double,
    maxTokens: Int32
) -> UnsafeMutablePointer<CChar>? {
    guard isInitialized else {
        return createErrorResponse("Not initialized")
    }

    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let requestResult = beginRequest(sessionID)
        guard case .success(let session) = requestResult else {
            guard case .failure(let code) = requestResult else { return createErrorResponse("Unknown error") }
            return createErrorResponse(beginRequestErrorMessage(code, sessionID: sessionID), errorCode: code)
        }
        defer { endRequest(sessionID) }

        let promptString = String(cString: prompt)
        let schemaString = String(cString: schemaJson)

        // Parse schema JSON to dictionary
        guard let schemaData = schemaString.data(using: .utf8),
              let schemaDict = try? JSONSerialization.jsonObject(with: schemaData) as? [String: Any] else {
            return createErrorResponse("Invalid schema JSON")
        }

        // Convert JSON Schema to DynamicGenerationSchema
        let conversionResult = convertJSONSchemaToDynamic(schemaDict)
        guard case .success(let dynamicSchema) = conversionResult else {
            if case .failure(let error) = conversionResult {
                return createErrorResponse(error.message)
            }
            return createErrorResponse("Failed to convert schema")
        }

        // Use semaphore for async coordination
        let semaphore = DispatchSemaphore(value: 0)
        var result: String = ""

        Task {
            do {
                // Configure generation options
                let options = GenerationOptions(
                    temperature: temperature,
                    maximumResponseTokens: Int(maxTokens)
                )

                // Create GenerationSchema from DynamicGenerationSchema
                let generationSchema = try GenerationSchema(root: dynamicSchema, dependencies: [])

                // Generate response with proper schema
                let response = try await session.respond(
                    to: promptString,
                    schema: generationSchema,
                    options: options
                )

                // Extract JSON from GeneratedContent
                guard let jsonObject = try extractValue(from: response.content) as? [String: Any] else {
                    throw NSError(domain: "FoundationModels", code: -1, userInfo: [NSLocalizedDescriptionKey: "Expected structure content"])
                }
                let jsonData = try JSONSerialization.data(withJSONObject: jsonObject)
                if let jsonString = String(data: jsonData, encoding: .utf8) {
                    result = jsonString
                } else {
                    // Use fallback error JSON for encoding failure
                    result = fallbackErrorJSON(code: .errorGeneration)
                }
            } catch {
                // Map error to specific error code
                let errorCode = mapGenerationErrorToCode(error)
                // Use safe JSON serialization for error response
                if let errorJson = createErrorResponse(error.localizedDescription, errorCode: errorCode) {
                    result = String(cString: errorJson)
                    free(errorJson)
                } else {
                    result = fallbackErrorJSON(code: errorCode)
                }
            }
            semaphore.signal()
        }

        semaphore.wait()
        return strdup(result)
    }
    #endif

    return createErrorResponse("FoundationModels not available")
}

// MARK: - Memory Management

@_cdecl("apple_ai_free_string")
public func appleAIFreeString(ptr: UnsafeMutablePointer<CChar>?) {
    guard let ptr = ptr else { return }
    free(ptr)
}

// MARK: - History Management

@_cdecl("apple_ai_get_history")
public func appleAIGetHistory(sessionID: Int32) -> UnsafeMutablePointer<CChar>? {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        guard getSessionEntry(sessionID) != nil else {
            return strdup("[]")
        }

        // The FoundationModels framework doesn't expose history directly
        // This is a limitation of the framework
        return strdup("[]")
    }
    #endif

    return strdup("[]")
}

@_cdecl("apple_ai_clear_history")
public func appleAIClearHistory(sessionID: Int32) {
    // Apple's API has no in-place history reset, so rebuild a fresh session
    // with this session's own instructions/tools and keep the same id -
    // other sessions are untouched.
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        guard let entry = getSessionEntry(sessionID) else { return }
        let newSession = buildSession(instructions: entry.instructions, tools: entry.tools)
        replaceSessionEntry(sessionID, with: SessionEntry(
            session: newSession,
            instructions: entry.instructions,
            tools: entry.tools
        ))
    }
    #endif
}

// MARK: - Statistics (Stubs)

@_cdecl("apple_ai_get_stats")
public func appleAIGetStats() -> UnsafeMutablePointer<CChar>? {
    let stats = """
    {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_tokens_generated": 0,
        "average_response_time": 0.0,
        "total_processing_time": 0.0
    }
    """
    return strdup(stats)
}

@_cdecl("apple_ai_reset_stats")
public func appleAIResetStats() {
    // Stub for compatibility
}

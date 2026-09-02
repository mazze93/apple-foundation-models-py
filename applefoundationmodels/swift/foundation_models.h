/**
 * foundation_models.h
 *
 * C header for Swift FoundationModels bindings
 * Declares C-compatible functions exported from foundation_models.swift
 */

#ifndef FOUNDATION_MODELS_H
#define FOUNDATION_MODELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Callback type for streaming. session_id identifies which session this chunk
// belongs to, so a process with multiple concurrently-streaming sessions can
// route chunks to the right consumer.
typedef void (*ai_stream_callback_t)(int32_t session_id, const char *chunk);

// Callback type for tool execution. session_id identifies which session's
// tool catalog `tool_name` should be resolved against.
typedef int32_t (*ai_tool_callback_t)(int32_t session_id,
                                       const char *tool_name,
                                       const char *arguments_json,
                                       char *result_buffer,
                                       int32_t buffer_size);

// Core library functions
int32_t apple_ai_init(void);
void apple_ai_cleanup(void);
const char *apple_ai_get_version(void);

// Availability functions
int32_t apple_ai_check_availability(void);
char *apple_ai_get_availability_reason(void);

// Session management.
// Creates an independent, isolated session and returns its session_id
// (a positive integer) on success, or a negative AIResult error code.
// tools_json (may be NULL) is a JSON array of {name, description, parameters}
// tool definitions bound to *this* session only; callback is required
// whenever tools_json is non-NULL.
int32_t apple_ai_create_session(const char *instructions_json,
                                 const char *tools_json,
                                 ai_tool_callback_t callback);

// Releases a session's native resources. Safe to call more than once.
int32_t apple_ai_close_session(int32_t session_id);

char *apple_ai_get_transcript(int32_t session_id);

// Text generation
char *apple_ai_generate(int32_t session_id,
                       const char *prompt,
                       double temperature,
                       int32_t max_tokens);

// Streaming generation
int32_t apple_ai_generate_stream(int32_t session_id,
                                const char *prompt,
                                double temperature,
                                int32_t max_tokens,
                                ai_stream_callback_t callback);

// Structured generation
char *apple_ai_generate_structured(int32_t session_id,
                                   const char *prompt,
                                   const char *schema_json,
                                   double temperature,
                                   int32_t max_tokens);

// History management
char *apple_ai_get_history(int32_t session_id);
void apple_ai_clear_history(int32_t session_id);

// Statistics
char *apple_ai_get_stats(void);
void apple_ai_reset_stats(void);

// Memory management
void apple_ai_free_string(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDATION_MODELS_H */

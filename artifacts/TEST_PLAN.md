# TEST_PLAN.md — ChannelAgent Test Suite

## Scope

Tests cover every public and private method of the `ChannelAgent` class in
`channel_agent.py`, plus the module-level `_normalize_handle()` helper. The
goal is 100% statement coverage with no reliance on live external services
(OpenAI, ChromaDB, filesystem outside `tmp_path`).

Out of scope: integration tests requiring a real OpenAI key, a real ChromaDB
on disk, or actual YouTube transcript data. Those belong behind
`@pytest.mark.integration`.

---

## Assumptions

| # | Assumption | Basis |
|---|-----------|-------|
| A1 | `_load_session` is expected to raise `json.JSONDecodeError` when the session file contains malformed JSON — there is no `try/except` guard in the source. Tests document this as a known limitation (see Suggested Improvements). | Source inspection of `channel_agent.py` line 69. |
| A2 | `chat()` is expected to raise `json.JSONDecodeError` when the model returns malformed tool-call `arguments` — `json.loads()` is called without a guard. Same as A1. | Source inspection of line 247. |
| A3 | `chat()` propagates `RuntimeError` (or any exception) from the OpenAI client — there is no retry or catch logic. | Source inspection of `chat()` loop. |
| A4 | When `tool_name` is anything other than `"get_channel_stats"`, the code routes to `_search_videos`. The `getattr(tc.function, "name", "search_videos")` fallback means a missing `name` attribute is treated as `search_videos`. | Source inspection of lines 248-249. |
| A5 | The sliding-window trim keeps the **last** `_MAX_HISTORY` messages (most recent). Oldest messages are discarded. | Source inspection of line 281: `self.messages[-_MAX_HISTORY:]`. |
| A6 | `_get_channel_stats()` prefers `state_info["total_videos_indexed"]` over `vs_stats["total_videos"]` when state info is available. | Source inspection of line 175. |
| A7 | `ensure_session_file` writes a `created_at` ISO timestamp field that `_save_session` does not write. These are intentionally different schemas. | Source inspection of lines 89-96 vs 79-84. |

---

## Test Matrix

| Source File | Source Function | Test Class | Test Name | Type |
|-------------|----------------|------------|-----------|------|
| `channel_agent.py` | `_normalize_handle` | `TestNormalizeHandle` | `test_at_handle_strips_at_sign` | unit |
| | | | `test_plain_lowercase_handle_unchanged` | unit |
| | | | `test_strips_full_youtube_url_with_www` | unit |
| | | | `test_strips_youtube_url_without_www` | unit |
| | | | `test_leading_nonalnum_triggers_yt_prefix` | unit |
| | | | `test_trailing_nonalnum_triggers_col_suffix` | unit |
| | | | `test_both_bad_ends_get_both_fixes` | unit |
| | | | `test_internal_special_chars_replaced_with_underscore` | unit |
| | | | `test_truncated_to_50_chars` | unit |
| | | | `test_empty_string_gets_yt_prefix` | adversarial |
| | | | `test_whitespace_only_treated_as_empty` | adversarial |
| | | | `test_uppercase_lowercased` | unit |
| | | | `test_numeric_only_handle_valid` | unit |
| | | | `test_result_always_starts_with_alnum` | adversarial |
| | | | `test_result_always_ends_with_alnum` | adversarial |
| | | | `test_http_url_also_stripped` | unit |
| | | | `test_unicode_channel_name_replaced` | adversarial |
| | | | `test_channel_url_with_path_suffix` | unit |
| | `_system_prompt` | `TestSystemPrompt` | `test_no_persona_mentions_channel_input` | unit |
| | | | `test_no_persona_uses_first_person` | unit |
| | | | `test_with_persona_uses_display_name` | unit |
| | | | `test_with_persona_includes_topics` | unit |
| | | | `test_with_persona_includes_phrases` | unit |
| | | | `test_with_persona_limits_phrases_to_five` | unit |
| | | | `test_with_persona_no_phrases_no_phrase_block` | unit |
| | | | `test_partial_persona_missing_display_name_uses_channel_input` | unit |
| | | | `test_partial_persona_missing_topics_no_crash` | adversarial |
| | | | `test_all_empty_persona_fields_no_crash` | adversarial |
| | | | `test_system_prompt_is_a_string` | unit |
| | `_search_videos` | `TestSearchVideos` | `test_empty_documents_returns_no_results_message` | unit |
| | | | `test_none_documents_returns_no_results_message` | adversarial |
| | | | `test_formats_result_number` | unit |
| | | | `test_formats_title_in_from_block` | unit |
| | | | `test_includes_url_in_output` | unit |
| | | | `test_includes_chunk_index_in_output` | unit |
| | | | `test_missing_optional_metadata_fields_use_defaults` | adversarial |
| | | | `test_multiple_results_separated_by_divider` | unit |
| | | | `test_query_forwarded_to_store` | unit |
| | | | `test_channel_name_forwarded_to_store` | unit |
| | | | `test_unicode_query_handled` | adversarial |
| | `_get_channel_stats` | `TestGetChannelStats` | `test_returns_json_string` | unit |
| | | | `test_includes_channel_field` | unit |
| | | | `test_videos_indexed_from_state_info_takes_priority` | unit |
| | | | `test_videos_indexed_falls_back_to_vs_stats_when_state_none` | unit |
| | | | `test_chunks_indexed_comes_from_vs_stats` | unit |
| | | | `test_known_video_ids_comes_from_vs_stats` | unit |
| | | | `test_last_checked_from_state_info` | unit |
| | | | `test_last_checked_none_when_channel_missing` | unit |
| | | | `test_get_stats_called_with_channel_input` | unit |
| | | | `test_transcript_videos_in_vector_store_field_present` | unit |
| | `chat` | `TestChat` | `test_no_tool_call_returns_content` | unit |
| | | | `test_no_tool_call_appends_user_and_assistant_messages` | unit |
| | | | `test_empty_user_message_accepted` | adversarial |
| | | | `test_single_search_videos_tool_call_resolves` | unit |
| | | | `test_single_tool_call_history_has_two_messages` | unit |
| | | | `test_get_channel_stats_tool_dispatch_returns_json` | unit |
| | | | `test_get_channel_stats_tool_calls_store_get_stats` | unit |
| | | | `test_on_tool_call_fired_for_search_videos` | unit |
| | | | `test_on_tool_call_fired_for_get_channel_stats` | unit |
| | | | `test_on_tool_call_not_set_does_not_raise` | unit |
| | | | `test_multiple_tool_calls_in_single_response` | unit |
| | | | `test_max_loop_exhaustion_returns_fallback` | unit |
| | | | `test_max_loop_exhaustion_still_saves_session` | unit |
| | | | `test_max_loop_exhaustion_appends_assistant_fallback_message` | unit |
| | | | `test_openai_exception_propagates` | adversarial |
| | | | `test_multi_turn_context_preserved` | unit |
| | | | `test_search_videos_called_with_empty_query_when_key_missing` | adversarial |
| | `_load_session` / `_save_session` / `ensure_session_file` | `TestSessionPersistence` | `test_load_session_returns_empty_list_when_no_file` | unit |
| | | | `test_load_session_reads_existing_messages` | unit |
| | | | `test_load_session_corrupt_json_raises` | adversarial |
| | | | `test_save_session_writes_channel_input` | unit |
| | | | `test_save_session_writes_session_id` | unit |
| | | | `test_save_session_writes_last_active` | unit |
| | | | `test_save_session_writes_messages` | unit |
| | | | `test_save_session_valid_utf8_unicode` | adversarial |
| | | | `test_ensure_session_file_creates_if_missing` | unit |
| | | | `test_ensure_session_file_does_not_overwrite_existing` | unit |
| | | | `test_ensure_session_file_writes_empty_messages` | unit |
| | | | `test_ensure_session_file_has_created_at_field` | unit |
| | | | `test_session_reload_across_agent_instances` | unit |
| | | | `test_session_path_uses_session_id_as_filename` | unit |
| | sliding-window logic | `TestSlidingWindow` | `test_trims_exactly_at_boundary` | unit |
| | | | `test_does_not_trim_below_boundary` | unit |
| | | | `test_repeated_calls_never_exceed_max_history` | unit |
| | | | `test_trimmed_messages_are_oldest` | unit |
| | adversarial | `TestAdversarial` | `test_malformed_tool_call_arguments_json_raises` | adversarial |
| | | | `test_very_long_user_message` | adversarial |
| | | | `test_unicode_user_message` | adversarial |
| | | | `test_very_long_channel_name_truncated` | adversarial |
| | | | `test_channel_name_with_only_special_chars` | adversarial |
| | | | `test_search_videos_store_returns_mismatched_lengths` | adversarial |
| | | | `test_chat_with_none_tool_call_name_defaults_to_search_videos` | adversarial |

---

## Coverage Mapping

```
channel_agent.py  —  110 statements  —  100% covered (0 missing)
```

Every branch covered:

| Branch | Covered by |
|--------|-----------|
| `_normalize_handle`: empty/whitespace input | `test_empty_string_gets_yt_prefix`, `test_whitespace_only_treated_as_empty` |
| `_normalize_handle`: leading non-alnum → `yt_` prefix | `test_leading_nonalnum_triggers_yt_prefix` |
| `_normalize_handle`: trailing non-alnum → `_col` suffix | `test_trailing_nonalnum_triggers_col_suffix` |
| `_normalize_handle`: YouTube URL stripping | `test_strips_full_youtube_url_with_www`, `test_strips_youtube_url_without_www` |
| `_system_prompt`: no persona branch | `TestSystemPrompt::test_no_persona_*` |
| `_system_prompt`: persona with phrases | `test_with_persona_includes_phrases` |
| `_system_prompt`: persona without phrases (`phrase_block = ""`) | `test_with_persona_no_phrases_no_phrase_block` |
| `_search_videos`: empty/None documents → no-results message | `test_empty_documents_returns_no_results_message`, `test_none_documents_returns_no_results_message` |
| `_search_videos`: result formatting loop | `test_formats_result_number`, `test_includes_url_in_output`, etc. |
| `_get_channel_stats`: state_info is None (fallback to vs_stats) | `test_videos_indexed_falls_back_to_vs_stats_when_state_none` |
| `_get_channel_stats`: state_info present (priority path) | `test_videos_indexed_from_state_info_takes_priority` |
| `chat()`: no tool calls → direct reply | `test_no_tool_call_returns_content` |
| `chat()`: tool call → `search_videos` dispatch | `test_single_search_videos_tool_call_resolves` |
| `chat()`: tool call → `get_channel_stats` dispatch (lines 251-254) | `test_get_channel_stats_tool_dispatch_returns_json` |
| `chat()`: `on_tool_call` callback (search, line 260) | `test_on_tool_call_fired_for_search_videos` |
| `chat()`: `on_tool_call` callback (stats, lines 253-254) | `test_on_tool_call_fired_for_get_channel_stats` |
| `chat()`: `on_tool_call` is None (no-op path) | `test_on_tool_call_not_set_does_not_raise` |
| `chat()`: max loop exhaustion → fallback message | `test_max_loop_exhaustion_returns_fallback` |
| `chat()`: sliding window trim | `TestSlidingWindow` |

---

## How to Run

```bash
# Activate virtualenv
source .venv/bin/activate

# Run all channel_agent tests
python -m pytest tests/test_channel_agent.py -v

# Run with coverage
python -m pytest tests/test_channel_agent.py --cov=channel_agent --cov-report=term-missing

# Run only a specific class
python -m pytest tests/test_channel_agent.py::TestGetChannelStats -v

# Run the full unit suite (excludes integration + playwright tests)
python -m pytest --ignore=tests/test_integration.py --ignore=tests/test_chainlit_app.py -v
```

---

## Missing Coverage

| Area | Reason not tested |
|------|------------------|
| `_load_session` with a corrupted-but-valid-JSON file (e.g., `{}` missing `messages` key) | `data.get("messages", [])` safely returns `[]`; the behavior is correct and trivially obvious. Adding a test would be noise rather than signal. |
| `_save_session` filesystem write failure (disk full, permission error) | Would require OS-level mocking (`builtins.open` raising `OSError`). The method has no error handling, so the exception propagates to the caller — no special behavior to verify. |
| Live OpenAI responses with actual tool-call JSON | Requires `OPENAI_API_KEY` and network access. Gate with `@pytest.mark.integration`. |
| ChromaDB persistence across process restarts | Requires a real ChromaDB on disk. Gate with `@pytest.mark.integration`. |
| `_normalize_handle` in `persona_builder.py` (separate copy) | Out of scope for this test file; covered by `tests/test_persona_builder.py`. |
| Concurrent `chat()` calls from multiple threads | `ChannelAgent` has no thread-safety mechanism; concurrent tests would be non-deterministic. Document as a known limitation. |

---

## Suggested Code Improvements

These are observations from writing tests, not changes made to source code.

1. **Silent session corruption.** `_load_session()` calls `json.loads(path.read_text())` without a `try/except`. A corrupted session file causes `ChannelAgent.__init__()` to raise `JSONDecodeError`, making the agent unusable. Consider catching the error, logging a warning, and returning `[]` (empty history) as a safe fallback.

2. **Unguarded tool-call argument parsing.** `json.loads(tc.function.arguments or "{}")` in `chat()` will raise `JSONDecodeError` for malformed model output. Consider wrapping with `try/except json.JSONDecodeError` and treating the call as a no-op or falling back to `{}`.

3. **`_MAX_HISTORY` constant is accessible but undocumented.** It is `20` messages (10 conversation turns). A docstring or comment on the sliding-window logic explaining the intent (token budget, not just count) would help maintainers.

4. **`_get_channel_stats` instantiates `PipelineState()` on every call.** This reads the state JSON file from disk each time. If `chat()` triggers multiple `get_channel_stats` tool calls in one turn, there will be multiple disk reads. Consider passing a `PipelineState` instance in at construction time (dependency injection), which also makes the class easier to test without monkey-patching the class.

5. **`on_tool_call` is an instance attribute set to `None` by default**, not declared in `__init__` with a type annotation. This makes IDE auto-complete and static type checkers unable to discover it. Declaring it as `self.on_tool_call: Optional[Callable[[str, str], None]] = None` in `__init__` would improve clarity.

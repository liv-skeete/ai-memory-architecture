# MRS Scratchpad

This file is a temporary workspace for planning, drafting, and storing intermediate results during development. It can be safely cleared as needed.

**--- clear below here when starting new project ---**


## Plan: “MRS v4 — Standardize LLM Access via Open WebUI Connector”

1) Background and current behavior
- MRS currently calls LLMs directly via its own HTTP client and retry logic in `Filter._query_api()`, orchestrated from `Filter._get_relevant_memories_llm()`.
- SMR already uses the Open WebUI connector via `generate_chat_completion()` inside `Filter._get_model_recommendation()`, passing the FastAPI Request and user context through `Filter.inlet()`.

2) Target behavior (parity with SMR)
- Route all MRS re-ranking LLM calls through Open WebUI’s `generate_chat_completion()` so provider selection, credentials, quotas, and telemetry are centralized.
- Avoid filter recursion by setting `bypass_filter=True` on connector calls (as SMR does), and ensure `stream=False` for deterministic, structured outputs.
- Include standardized metadata for attribution and audit similar to SMR (e.g., `metadata.user_id = user.email` when available, plus an “mrs” breadcrumb).

3) API and signature changes
- Modify MRS inlet signature to accept the platform Request:
  - From:
    - `async Filter.inlet(self, body, __event_emitter__=None, __user__=None) -> dict`
  - To:
    - `async Filter.inlet(self, body, __event_emitter__=None, __user__=None, __request__=None) -> dict`
- Thread `__request__` down to the re-ranking call path:
  - `Filter._get_relevant_memories_llm()` will accept `request: Request` and pass it into the connector wrapper.
  - Replace callsites currently constructing `api_task = self._query_api(...)` with connector calls.

4) New connector wrapper in MRS
- Add a private helper, similar in spirit to SMR’s approach:
  - `def _chat_via_connector(self, request, messages, model_id, temperature, max_tokens, user) -> str`
  - Implementation: `await generate_chat_completion(request, payload, user=user, bypass_filter=True)`
    - payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "metadata": {
          "user_id": getattr(user, "email", None),
          "mrs": {"purpose": "rerank", "version": "3.x"}
        },
      }
  - Parse `response["choices"][0]["message"]["content"]` into a string to preserve the existing contract expected by `Filter._get_relevant_memories_llm()`.

5) Valve changes (config)
- Add:
  - `reranker_model_id: str = "gpt-4.1-mini"` (or similar) — an Open WebUI model ID/alias
  - `llm_request_timeout_seconds: int = 10` (wrap `await generate_chat_completion` with `asyncio.wait_for`)
  - `provider_options: dict[str, Any] = {}` (optional passthrough for future per-provider knobs)
- Remove:
  - `openai_api_url`, `openai_api_key`, `openai_model`
  - `ollama_api_url`, `ollama_model`, `ollama_num_ctx`
  - `api_provider`

6) Replace direct calls in the re-ranking path
- In `Filter._get_relevant_memories_llm()`:
  - Build messages as today (system + user).
  - Schedule tasks directly with `self._chat_via_connector(__request__, messages, self.valves.reranker_model_id, self.valves.temperature, self.valves.max_tokens, user)`.
- The legacy `self._query_api(...)` method will be removed entirely.
- Continue the chunked parallel gather; only the call implementation changes.

7) Governance and recursion safety
- Set `bypass_filter=True` in `generate_chat_completion` to prevent the MRS call from re-entering the filter pipeline and recursively invoking memory.
- Ensure that `reranker_model_id` is NOT included in `Valves.memory_enabled_models` to avoid any indirect loops.
- Keep `“stream”: False` for structured JSON returns.

8) Timeouts, retries, and error handling
- Wrap each connector call in `asyncio.wait_for(..., timeout=self.valves.llm_request_timeout_seconds)` analogous to SMR.
- The platform connector is responsible for provider-level retries. The MRS module will not implement its own retry logic.

9) Telemetry and metadata alignment
- Add metadata to connector calls:
  - `"user_id": user.email` if available
  - `"mrs": {"purpose": "rerank", "version": self.version or module version}`
- Logging:
  - On success: log `elapsed_ms`, chunk index, and top parsed relevance items (sanitized).
  - On malformed JSON: log the truncated raw response.

10) Migration and docs
- Update in-code docstrings for `Filter._get_relevant_memories_llm()` and the new connector helper.
- Remove documentation related to direct provider configuration (`OpenAI`, `Ollama`, etc.).
- Add a short section in the module header describing the required centralized provider handling.

11) Acceptance criteria
- Functional:
  - Re-ranking works via the platform connector.
  - No recursion into memory filters (verified by logs).
- Operational:
  - Platform logs show MRS re-rank calls attributed to the user with `mrs` metadata.
  - No provider credentials are read from or stored in the MRS module.
- Performance:
  - End-to-end LLM re-ranking latency is within acceptable limits (e.g., ≤ 50 ms overhead per chunk compared to a direct call baseline).

12) References for consistency
- SMR connector usage:
  - `Filter._get_model_recommendation()`
  - `generate_chat_completion()` call
  - Timeout guard via `asyncio.wait_for`
  - Metadata with `user_id` (email) in payload
- MRS direct call sites to be removed:
  - `Filter._get_relevant_memories_llm()` scheduling of `_query_api`
  - The entire `Filter._query_api()` method

Mermaid overview
```mermaid
flowchart LR
  subgraph Old MRS (To Be Removed)
    A[MRS _get_relevant_memories_llm] --> B[_query_api (aiohttp)]
    B --> C[OpenAI/Ollama endpoints]
  end

  subgraph New MRS (Single Path)
    D[MRS _get_relevant_memories_llm] --> E[_chat_via_connector]
    E --> F[generate_chat_completion (Open WebUI)]
    F --> G[Provider(s)]
  end
```

Pseudocode sketch
- New helper:
  - `Filter._chat_via_connector()`
- Callsite change in re-ranking:
  - Replace `api_task = self._query_api(...)` with:
    - `api_task = self._chat_via_connector(__request__, messages, self.valves.reranker_model_id, self.valves.temperature, self.valves.max_tokens, user)`
- inlet signature:
  - Align with SMR’s `Filter.inlet()` by adding `__request__`

End of plan.
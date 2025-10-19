"""
title: AMM - Recall & Storage
description: Adaptive Memory Manager - Recall & Storage Module for Open WebUI - Combines synchronous memory retrieval with asynchronous memory storage
author: Cody
version: 2.0.8
date: 2025-07-10
dev_note: "AI Coding Assistant: When making non-trivial code changes, update 'version', 'date', and add an entry under 'changes'."
changes:
  - v2.0.8 (2025-07-10):
      - Refactored citation formatting into a centralized `_format_citation_line` function to ensure consistency.
      - Added '☐' symbol to "Memories Read" and ensured "Memories Deleted" citations retain their tags.
  - v2.0.7 (2025-07-08):
      - Simplified `_parse_json_response` to remove complex regex/string manipulation and parse the API response directly. Assumes the LLM provides clean, valid JSON.
  - v2.0.6 (2025-07-08):
      - Refactored API query logic to consolidate `query_openai_api` and `query_ollama_api` into a single `_query_api` method.
      - Removed obsolete `num_ctx` setting from Ollama API calls.
  - v2.0.5 (2025-07-08):
      - Added status emitter for memory purging (`_delete_excess_memories`).
      - Standardized async status emitters to use a shared `message_id` for sequential display.
      - Threaded `**kwargs` through async methods to support status emissions.
  - v2.0.4 (2025-07-07):
      - Created `_get_memory_importance` helper to robustly extract scores from the `importance` attribute or content string `[score]` / `(score)`.
      - Refactored `_calculate_retention_score` and `_delete_excess_memories` to use the new helper for consistent score handling.
  - v2.0.3 (2025-07-07):
      - Patched `_delete_excess_memories` logging to prevent `AttributeError` on memories with older schemas lacking an `importance` score.
  - v2.0.2 (2025-07-07):
      - Enhanced memory pruning to return and display the count of deleted memories.
      - Improved logging in `_delete_excess_memories` for better diagnostics.
  - v2.0.1 (2025-07-05):
      - Made `_strip_leading_tags` and `_strip_trailing_scores` iterative to handle multiple occurrences.
      - Enhanced `_strip_trailing_scores` to recognize both `()` and `[]` brackets.
  - v2.0.0 (2025-07-03):
      - Refactored core data flow to use a `ProcessingMemory` dataclass.
      - Decoupled memory content from metadata (relevance, importance, tags).
      - Replaced ambiguous "score" with specific "relevance" (Stage 1) and "importance" (Stage 2) fields.
"""

import aiohttp
import asyncio
import json
import logging
import os
import re
import time
import math

from pydantic import BaseModel, Field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

from open_webui.models.memories import Memories

# Logger configuration
logger = logging.getLogger("amm_mrs")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler once
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


class ProcessingMemory(BaseModel):
    """
    A data class to represent a memory during processing, keeping content
    separate from metadata like relevance and importance scores.
    """

    content: str = Field(description="The core text of the memory.")
    tag: Optional[str] = Field(
        default=None,
        description="An optional classification tag (e.g., 'Hobby', 'Work').",
    )
    relevance: Optional[float] = Field(
        default=None,
        description="The relevance of the memory to the current context (Stage 1 score).",
    )
    importance: Optional[float] = Field(
        default=None,
        description="The intrinsic importance of the memory (Stage 2 score).",
    )
    operation: Optional[Literal["NEW", "DELETE"]] = Field(
        default=None, description="The operation to perform on this memory."
    )


class Filter:
    """Memory Recall & Storage module combining synchronous memory retrieval with asynchronous memory storage."""

    class Valves(BaseModel):
        # Set processing priority
        priority: int = Field(
            default=2,
            description="Priority level for the filter operations.",
        )
        # UI settings
        show_status: bool = Field(
            default=True, description="Show memory operation status in chat"
        )
        # Debug settings
        verbose_logging: bool = Field(
            default=False,
            description="Enable detailed diagnostic logging",
        )
        max_log_lines: int = Field(
            default=10,
            description="Maximum number of lines to show in verbose logs for multi-line content",
        )
        # Memory retention settings
        max_memories: int = Field(
            default=100,
            description="Maximum number of memories to store",
        )
        selection_method: Literal["Oldest", "Hybrid"] = Field(
            default="Hybrid",
            description="Method for selecting which memories to drop",
        )
        max_age: int = Field(
            default=30,
            description="Half-life in days for exponential decay of memory recency",
        )
        importance_weight: float = Field(
            default=0.5,
            description="Weight for importance when pruning, between 0.0 and 1.0",
        )
        # API configuration
        api_provider: Literal["OpenAI API", "Ollama API"] = Field(
            default="OpenAI API",
            description="Choose LLM API provider for memory processing",
        )
        # OpenAI settings
        openai_api_url: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API endpoint",
        )
        openai_api_key: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""),
            description="OpenAI API key",
        )
        openai_model: str = Field(
            default="gpt-4.1-mini",
            description="OpenAI model to use for memory processing",
        )
        max_tokens: int = Field(
            default=750,
            description="Maximum tokens for API calls",
        )
        # Ollama settings
        ollama_api_url: str = Field(
            default="http://ollama:11434",
            description="Ollama API URL",
        )
        ollama_model: str = Field(
            default="qwen2.5:14b",
            description="Ollama model to use for memory processing",
        )
        # Common API settings
        request_timeout: int = Field(
            default=30,
            description="Timeout for API requests in seconds",
        )
        max_retries: int = Field(
            default=2,
            description="Maximum number of retries for API calls",
        )
        retry_delay: float = Field(
            default=1.0,
            description="Delay between retries (seconds)",
        )
        temperature: float = Field(
            default=0.3,
            description="Temperature for API calls",
        )
        # Stage 1: Memory Retrieval settings
        relevance_threshold: float = Field(
            default=0.5,
            description="Minimum relevance score (0.0-1.0) for memories to be included in context",
        )
        relevance_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory relevance prompt with preserved formatting",
        )
        # Stage 2: Memory Storage settings
        memory_importance_threshold: float = Field(
            default=0.5,
            description="Minimum importance threshold for identifying potential memories (0.0-1.0)",
        )
        identification_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory identification prompt with preserved formatting",
        )
        recent_messages_count: int = Field(
            default=1,
            description="Number of recent user messages to analyze for memory (1-10)",
        )
        # Asynchronous processing settings
        async_memory_storage: bool = Field(
            default=True,
            description="Process memory storage asynchronously after response is sent",
        )

    def __init__(self) -> None:
        """
        Initialize the Memory Recall & Storage module.
        """
        # Initialize with empty prompts - must be set via update_valves
        try:
            self.valves = self.Valves(
                relevance_prompt="",  # Empty string to start - must be set via update_valves
                identification_prompt="",  # Empty string to start - must be set via update_valves
            )
        except Exception as e:
            logger.error(f"Failed to initialize valves: {e}")
            raise

        # Configure aiohttp session with optimized connection pooling
        connector = aiohttp.TCPConnector(
            limit=10,  # Limit total number of connections
            limit_per_host=5,  # Limit connections per host
            enable_cleanup_closed=True,  # Clean up closed connections
            force_close=False,  # Keep connections alive between requests
            ttl_dns_cache=300,  # Cache DNS results for 5 minutes
        )
        self.session = aiohttp.ClientSession(connector=connector)

        # Initialize background task tracking
        self._background_tasks = set()

    async def close(self) -> None:
        """Close the aiohttp session and cancel any background tasks with timeouts."""
        # Cancel any running background tasks with timeout
        if hasattr(self, "_background_tasks"):
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
            self._background_tasks.clear()

        # Close the session with timeout
        if hasattr(self, "session") and self.session and not self.session.closed:
            try:
                await asyncio.wait_for(self.session.close(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Session close timed out")
            finally:
                self.session = None

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager with resource cleanup."""
        await self.close()

    def update_valves(self, new_valves: Dict[str, Any]) -> None:
        """
        Update valve settings.

        Args:
            new_valves: Dictionary of valve settings to update
        """
        # Only log configuration updates in verbose mode
        if self.valves.verbose_logging:
            logger.info("Updating module configuration")

        for key, value in new_valves.items():
            if hasattr(self.valves, key):
                # For prompt fields, log a truncated version
                if key.endswith("_prompt") and isinstance(value, str):
                    if self.valves.verbose_logging:
                        preview = value[:50] + "..." if len(value) > 50 else value
                        logger.info(f"Updating {key} with: {preview}")
                    setattr(self.valves, key, value)
                else:
                    if self.valves.verbose_logging:
                        logger.info(f"Updating {key} with: {value}")
                    setattr(self.valves, key, value)

    @staticmethod
    def _inv_age_exponential(updated_at: int, half_life_days: float) -> float:
        now = int(time.time())
        age_days = (now - updated_at) / 86400.0
        lam = math.log(2) / half_life_days
        return math.exp(-lam * age_days)

    @staticmethod
    def _get_memory_importance(m: Any) -> float:
        """
        Get the importance score from a memory, checking the attribute first
        and falling back to parsing the content string.
        Handles both `[score]` and `(score)` formats.
        """
        # Prioritize the dedicated 'importance' attribute if available
        if hasattr(m, "importance") and m.importance is not None:
            return m.importance

        # Fallback for older schemas: parse from content
        if hasattr(m, "content") and m.content:
            # Regex to find a score like [0.8] or (0.8) at the end of the string
            match = re.search(r"[\[\(](\d(?:\.\d+)?)[\)\]]$", m.content.strip())
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, TypeError):
                    pass

        # Default if no score can be determined
        return 0.5

    @staticmethod
    def _calculate_retention_score(
        m, importance_weight: float, half_life_days: float
    ) -> float:
        """Calculate the retention score based on importance and age."""
        inv_age = Filter._inv_age_exponential(m.updated_at, half_life_days)
        importance = Filter._get_memory_importance(m)
        return (importance_weight * importance) + ((1 - importance_weight) * inv_age)

    async def _delete_excess_memories(self, user_id: str, **kwargs) -> int:
        """Remove excess memories, emit status, and return deleted count."""
        loop = asyncio.get_event_loop()
        mems = (
            await loop.run_in_executor(None, Memories.get_memories_by_user_id, user_id)
            or []
        )

        num_to_prune = len(mems) - self.valves.max_memories
        if num_to_prune <= 0:
            if self.valves.verbose_logging:
                logger.info(f"No excess memories to delete for user {user_id}")
            return 0

        logger.info(
            f"User {user_id} has {len(mems)} memories; pruning to {self.valves.max_memories} using {self.valves.selection_method} method."
        )

        # Emit the new status message *before* purging.
        await self._safe_emit(
            "status", f"🗑️ Purging {num_to_prune} memories...", **kwargs
        )

        if self.valves.selection_method == "Oldest":
            mems.sort(key=lambda m: m.created_at)
        else:  # Hybrid
            half_life_days = self.valves.max_age
            mems.sort(
                key=lambda m: self._calculate_retention_score(
                    m, self.valves.importance_weight, half_life_days
                )
            )

        deleted_count = 0
        # Corrected loop logic to avoid deleting one too many.
        while len(mems) > self.valves.max_memories:
            m = mems.pop(0)
            if self.valves.verbose_logging:
                truncated_content = self._truncate_log_lines(m.content, max_lines=1)
                importance_score = self._get_memory_importance(m)
                logger.info(
                    f"Deleting memory ID {m.id} (Importance: {importance_score:.2f}): '{truncated_content}'"
                )
            await loop.run_in_executor(
                None, Memories.delete_memory_by_id_and_user_id, m.id, user_id
            )
            deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Purged {deleted_count} excess memories for user {user_id}")
            # Final status/citation is now handled by the calling function.

        return deleted_count

    def _truncate_log_lines(self, text: str, max_lines: int = None) -> str:
        """
        Truncate a multi-line string to a maximum number of lines.

        Args:
            text: The text to truncate
            max_lines: Maximum number of lines (defaults to self.valves.max_log_lines)

        Returns:
            Truncated text with indicator of how many lines were omitted
        """
        if not max_lines:
            max_lines = self.valves.max_log_lines

        lines = text.split("\n")
        if len(lines) <= max_lines:
            return text

        truncated = lines[:max_lines]
        omitted = len(lines) - max_lines
        truncated.append(f"... [truncated, {omitted} more lines omitted]")
        return "\n".join(truncated)

    def format_bulleted_list(self, items: list) -> str:
        """
        Format a list of strings as a bulleted list.

        Args:
            items: List of strings to format

        Returns:
            String with each item formatted as a bullet point, or empty string for empty list
        """
        if not items:
            return ""
        return "\n".join([f"- {item}" for item in items])

    # --------------------------------------------------------------------------
    # Stage 1: Memory Retrieval (Synchronous)
    # --------------------------------------------------------------------------

    async def _get_relevant_memories_llm(
        self, current_message: str, db_memories: List[Any]
    ) -> List[ProcessingMemory]:
        """
        Get memories relevant to the current context using LLM-based scoring.

        Args:
            current_message: The current user message
            db_memories: List of memories from the database

        Returns:
            List of ProcessingMemory objects with 'content' and 'relevance' populated.
        """
        if self.valves.verbose_logging or len(db_memories) > 10:
            logger.info("Getting relevant memories using LLM-based scoring")

        memory_contents = [
            mem.content
            for mem in db_memories
            if hasattr(mem, "content") and mem.content
        ]

        if not memory_contents:
            logger.info("No valid memory contents found in database")
            return []

        formatted_memories = self.format_bulleted_list(memory_contents)

        if self.valves.verbose_logging or len(memory_contents) > 10:
            logger.info(f"Processing {len(memory_contents)} memories from database")

        if not self.valves.relevance_prompt:
            logger.error("Relevance prompt is empty - cannot process memories")
            raise ValueError("Relevance prompt is empty - module cannot function")

        system_prompt = self.valves.relevance_prompt.format(
            current_message=self.format_bulleted_list([current_message]),
            memories=formatted_memories,
        )

        if self.valves.verbose_logging:
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(f"System prompt (truncated):\n{truncated_prompt}")

        if self.valves.verbose_logging:
            model = (
                self.valves.openai_model
                if self.valves.api_provider == "OpenAI API"
                else self.valves.ollama_model
            )
            logger.info(f"Querying {self.valves.api_provider} with model: {model}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_message},
        ]
        response = await self._query_api(self.valves.api_provider, messages)

        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info(f"Raw API response: {truncated_response}")

        prepared_response = self._prepare_json_response(response)
        try:
            memory_data = json.loads(prepared_response)
            valid_memories = []
            if isinstance(memory_data, list):
                for mem in memory_data:
                    if (
                        isinstance(mem, dict)
                        and "content" in mem
                        and "relevance" in mem
                    ):
                        try:
                            relevance = float(mem["relevance"])
                            valid_memories.append(
                                ProcessingMemory(
                                    content=str(mem["content"]), relevance=relevance
                                )
                            )
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid memory format: {mem}")
                            pass

            valid_memories.sort(key=lambda x: x.relevance, reverse=True)
            total_memories = len(valid_memories)
            threshold_filtered_memories = [
                mem
                for mem in valid_memories
                if mem.relevance >= self.valves.relevance_threshold
            ]

            filtered_count = total_memories - len(threshold_filtered_memories)
            if filtered_count > 0:
                logger.info(
                    f"Found {total_memories} memories, filtered {filtered_count} below threshold {self.valves.relevance_threshold:.2f}"
                )
            else:
                logger.info(f"Found {total_memories} relevant memories")

            if self.valves.verbose_logging and threshold_filtered_memories:
                for i, mem in enumerate(threshold_filtered_memories):
                    logger.info(
                        f"Memory {i+1}: {mem.content} [relevance: {mem.relevance:.2f}]"
                    )
                if filtered_count > 0:
                    logger.info(
                        f"Memories filtered out by threshold {self.valves.relevance_threshold:.2f}:"
                    )
                    for i, mem in enumerate(
                        [
                            m
                            for m in valid_memories
                            if m.relevance < self.valves.relevance_threshold
                        ]
                    ):
                        logger.info(
                            f"Filtered {i+1}: {mem.content} [relevance: {mem.relevance:.2f}]"
                        )
            return threshold_filtered_memories

        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            logger.error(f"Failed JSON content: {prepared_response}")
            return []

    async def get_relevant_memories(
        self,
        current_message: str,
        user_id: str,
        db_memories: Optional[List[Any]] = None,
    ) -> List[ProcessingMemory]:
        """
        Get memories relevant to the current context using LLM-based scoring.

        Args:
            current_message: The current user message
            user_id: The user ID
            db_memories: List of memories from the database

        Returns:
            List of ProcessingMemory objects containing relevant memories.
        """
        if not db_memories:
            return []

        try:
            return await self._get_relevant_memories_llm(current_message, db_memories)
        except Exception as e:
            logger.error(f"Error in get_relevant_memories: {str(e)}")
            return []

    def _format_citation_line(
        self,
        symbol: str,
        content: str,
        tag: Optional[str] = None,
        score: Optional[float] = None,
        score_label: Optional[str] = None,
    ) -> str:
        """
        Format a single memory item for citation with a symbol.

        Args:
            symbol: The leading symbol (e.g., '☐', '☑', '☒').
            content: The memory content.
            tag: An optional classification tag.
            score: An optional score (relevance or importance).
            score_label: The label for the score (e.g., 'relevance').

        Returns:
            A formatted citation string for a single memory.
        """
        tag_str = f"{tag} " if tag else ""
        score_str = (
            f" [{score_label}: {score:.1f}]" if score is not None and score_label else ""
        )
        return f"{symbol} {tag_str}{content}{score_str}"

    def _format_memories_for_context(
        self, relevant_memories: List[ProcessingMemory]
    ) -> tuple[str, str]:
        """
        Format relevant memories for inclusion in the context.

        Args:
            relevant_memories: List of ProcessingMemory objects from Stage 1.

        Returns:
            Tuple containing:
            - Formatted string of memories for inclusion in the context (without scores).
            - Formatted string of memories with scores for logging.
        """
        if not relevant_memories:
            return "", ""

        context_memories = "User Information (sorted by relevance):"
        citation_memories = "Memories Read (sorted by relevance):"

        if relevant_memories:
            context_memories += "\n" + self.format_bulleted_list(
                [mem.content for mem in relevant_memories]
            )
            citation_memories += "\n" + "\n".join(
                self._format_citation_line(
                    symbol="☐",
                    content=mem.content,
                    score=mem.relevance,
                    score_label="relevance",
                )
                for mem in relevant_memories
            )

        return context_memories, citation_memories

    def _format_relevant_memories(self, memories: List[ProcessingMemory]) -> str:
        """
        Format relevant memories for inclusion in the identification prompt.

        Args:
            memories: List of relevant ProcessingMemory objects from Stage 1.

        Returns:
            Formatted string of memory content, newest first.
        """
        if not memories:
            return "No relevant memories found."
        return self.format_bulleted_list([mem.content for mem in reversed(memories)])

    def _update_message_context(self, body: dict, formatted_memories: str) -> None:
        """
        Update message context with relevant memories.

        Args:
            body: The message body
            formatted_memories: Formatted string of memories
        """
        if not formatted_memories:
            return

        if "messages" in body and len(body["messages"]) > 0:
            body["messages"].insert(
                0, {"role": "system", "content": formatted_memories}
            )

    # --------------------------------------------------------------------------
    # Stage 2: Memory Storage (Asynchronous)
    # --------------------------------------------------------------------------

    async def _identify_potential_memories(
        self,
        current_message: str,
        relevant_memories: List[ProcessingMemory],
        **kwargs,
    ) -> List[ProcessingMemory]:
        """
        Identify potential memories from the current message and recent context.
        """
        await self._safe_emit(
            "status", "🧠 Identifying potential memories...", **kwargs
        )
        logger.info("Identifying potential memories from message")

        if not self.valves.identification_prompt:
            logger.error("Identification prompt is empty - cannot process memories")
            raise ValueError("Identification prompt is empty - module cannot function")

        formatted_relevant = self._format_relevant_memories(relevant_memories)
        system_prompt = self.valves.identification_prompt.format(
            current_message=current_message,
            relevant_memories=formatted_relevant,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": current_message},
        ]
        response = await self._query_api(self.valves.api_provider, messages)

        if self.valves.verbose_logging:
            logger.info(
                f"Identification: Raw API response: {self._truncate_log_lines(response)}"
            )

        raw_memories = self._parse_json_response(response)
        potential_memories = []

        for raw_mem in raw_memories:
            if (
                isinstance(raw_mem, dict)
                and "content" in raw_mem
                and "importance" in raw_mem
            ):
                try:
                    content_clean_score = self._strip_trailing_scores(
                        raw_mem["content"]
                    )
                    content_clean, tag = self._strip_leading_tags(content_clean_score)
                    importance = float(raw_mem["importance"])

                    mem_obj = ProcessingMemory(
                        content=content_clean, tag=tag, importance=importance
                    )
                    potential_memories.append(mem_obj)

                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Invalid memory format from LLM: {raw_mem} | Error: {e}"
                    )

        if not potential_memories:
            logger.info("No potential memories identified.")

        return potential_memories

    async def _identify_and_prepare_memories(
        self,
        current_message: str,
        relevant_memories: List[ProcessingMemory],
        **kwargs,
    ) -> List[ProcessingMemory]:
        """
        Identifies potential memories and prepares memory operations (NEW or DELETE).
        """
        potential_memories = await self._identify_potential_memories(
            current_message, relevant_memories, **kwargs
        )

        important_memories = [
            mem
            for mem in potential_memories
            if mem.importance
            and mem.importance >= self.valves.memory_importance_threshold
        ]

        if not important_memories:
            return []

        logger.info(
            f"Identified {len(important_memories)} important memories meeting threshold ({self.valves.memory_importance_threshold:.2f})."
        )

        for mem in important_memories:
            if mem.tag and mem.tag.lower() == "[delete]":
                mem.operation = "DELETE"
                # The content is already clean because [Delete] was parsed as the primary tag.
                # We need to find the *next* tag, if it exists, to formulate the final content for citation.
                content_to_delete, next_tag = self._strip_leading_tags(mem.content)
                mem.content = content_to_delete
                mem.tag = next_tag  # This will be the semantic tag (e.g., [Preference]) or None
            else:
                mem.operation = "NEW"

        return important_memories

    async def _execute_memory_operations(
        self,
        memory_operations: List[ProcessingMemory],
        user_id: str,
        **kwargs,
    ) -> Tuple[List[str], List[str]]:
        """
        Executes memory operations (creation and deletion) and returns lists of
        strings for citation.
        """
        await self._safe_emit("status", "💾 Storing and updating memories...", **kwargs)
        if not memory_operations:
            return [], []

        created_memories_for_citation = []
        deleted_memories_for_citation = []

        # Sort by importance for display, but process all
        sorted_ops = sorted(
            memory_operations, key=lambda x: x.importance or 0, reverse=True
        )

        for mem_op in sorted_ops:
            if mem_op.operation == "NEW" and mem_op.content:
                created_id = await self._create_memory(mem_op, user_id, **kwargs)
                if created_id:
                    citation_str = self._format_citation_line(
                        symbol="☑",
                        content=mem_op.content,
                        tag=mem_op.tag,
                        score=mem_op.importance,
                        score_label="importance",
                    )
                    created_memories_for_citation.append(citation_str)
                    logger.info(f"Successfully created memory: {citation_str}")

            elif mem_op.operation == "DELETE" and mem_op.content:
                deleted_content, deleted_tag = await self._delete_memory(
                    mem_op, user_id, **kwargs
                )
                if deleted_content:
                    citation_str = self._format_citation_line(
                        symbol="☒", content=deleted_content, tag=deleted_tag
                    )
                    deleted_memories_for_citation.append(citation_str)
                    logger.info(f"Successfully deleted memory: {citation_str}")

        return created_memories_for_citation, deleted_memories_for_citation

    async def _create_memory(
        self, memory_to_create: ProcessingMemory, user_id: str, **kwargs
    ) -> Optional[str]:
        """
        Create a new memory in the database from a ProcessingMemory object.
        """
        try:
            loop = asyncio.get_event_loop()

            # Format the memory back into a single string for storage.
            full_content = memory_to_create.content
            if memory_to_create.tag:
                full_content = f"{memory_to_create.tag} {full_content}"
            if memory_to_create.importance is not None:
                full_content = f"{full_content} [{memory_to_create.importance:.1f}]"

            memory = await loop.run_in_executor(
                None,
                Memories.insert_new_memory,
                user_id,
                full_content,
            )
            if memory and hasattr(memory, "id"):
                if self.valves.verbose_logging:
                    logger.info(f"Successfully created memory {memory.id}")
                return memory.id
            else:
                logger.warning("Memory creation returned None or object without ID.")
                return None
        except Exception as e:
            logger.error(f"Error creating memory: {e}", exc_info=True)
            return None

    @staticmethod
    def _strip_leading_tags(content: str) -> tuple[str, Optional[str]]:
        """
        Parses and removes one or more leading tags (e.g., `[Hobby]`) from raw LLM output.
        It iteratively strips tags to handle multiple formats.

        Args:
            content: The raw text from which to strip tags.

        Returns:
            A tuple containing the content fully cleaned of all leading tags,
            and the *first* tag that was extracted (or None).
        """
        tag_pattern = re.compile(r"^\s*(\[\w+\])\s*")
        cleaned_content = content.strip()
        first_tag = None

        # Loop to remove multiple tags and capture the first one
        while match := tag_pattern.match(cleaned_content):
            tag = match.group(1)
            if first_tag is None:
                first_tag = tag  # Save the first tag
            # Remove the matched tag and surrounding whitespace from the beginning of the string
            cleaned_content = cleaned_content[match.end() :].lstrip()

        return cleaned_content, first_tag

    @staticmethod
    def _strip_trailing_scores(content: str) -> str:
        """
        Removes one or more trailing scores (e.g., `[0.7]` or `(0.7)`) from raw LLM output.
        This is a defensive measure to clean up potential LLM inconsistencies.
        It iteratively removes scores to handle multiple appended formats.

        Args:
            content: The text from which to strip a score.

        Returns:
            The content with any trailing scores removed.
        """
        # This regex now handles both [score] and (score) formats.
        score_pattern = re.compile(r"\s*[\(\[]\s*\d+\.\d+\s*[\)\]]$")
        cleaned_content = content.strip()

        # Loop to remove multiple scores if they exist
        while score_pattern.search(cleaned_content):
            cleaned_content = score_pattern.sub("", cleaned_content).strip()

        return cleaned_content

    async def _delete_memory(
        self, memory_to_delete: ProcessingMemory, user_id: str, **kwargs
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Deletes a memory based on content, returning the original content and its tag.
        """
        try:
            if not user_id:
                logger.warning("Cannot delete memory: User ID is missing.")
                return None

            loop = asyncio.get_event_loop()
            all_memories = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, user_id
            )

            content_to_delete = memory_to_delete.content.strip()

            for mem in all_memories:
                # Fully normalize the database content by stripping tags and scores
                db_content_no_scores = self._strip_trailing_scores(mem.content)
                db_content_no_tags, db_tag = self._strip_leading_tags(
                    db_content_no_scores
                )

                if db_content_no_tags.strip() == content_to_delete:
                    logger.info(f"Found match for deletion: DB ID {mem.id}")
                    await loop.run_in_executor(
                        None, Memories.delete_memory_by_id_and_user_id, mem.id, user_id
                    )
                    # Return the matched content and the tag from the deleted DB record
                    return content_to_delete, db_tag

            logger.warning(
                f"Could not find a matching memory to delete for content: '{content_to_delete}'"
            )
            return None, None
        except Exception as e:
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            return None

    async def _process_memory_storage(
        self,
        user_id: str,
        body: dict,
        relevant_memories: List[ProcessingMemory],
        **kwargs,
    ) -> None:
        """
        Asynchronously process and store memories based on the conversation context.
        """
        try:
            logger.info(f"Starting async memory processing for user: {user_id}")

            await self._delete_excess_memories(user_id, **kwargs)

            # Determine message content to analyze
            history = body.get("messages", [])
            user_messages = [
                msg["content"] for msg in history if msg.get("role") == "user"
            ]
            recent_messages = user_messages[-self.valves.recent_messages_count :]
            current_message_content = self.format_bulleted_list(recent_messages)

            # Identify and prepare memories
            memories_to_process = await self._identify_and_prepare_memories(
                current_message_content, relevant_memories, **kwargs
            )

            # Execute operations
            if memories_to_process:
                created_citations, deleted_citations = (
                    await self._execute_memory_operations(
                        memories_to_process, user_id, **kwargs
                    )
                )

                # Consolidate citations for a single "Memories Processed" message
                all_citations = created_citations + deleted_citations
                if all_citations:
                    # The "processed" citation is the final async step.
                    await self._send_citation(
                        all_citations, user_id, "processed", **kwargs
                    )

            else:
                logger.info(f"No new memory operations to execute for user {user_id}.")
                await self._safe_emit(
                    "status", "✅ Memories are up-to-date", **kwargs, final=True
                )

        except Exception as e:
            logger.error(f"Error during async memory processing: {e}", exc_info=True)
            await self._safe_emit(
                "status", f"⚠️ Memory processing error", **kwargs, final=True
            )
        finally:
            # This block guarantees the final status message is always emitted after all operations.
            logger.info("Async memory processing finished.")
            await self._safe_emit("status", "🏁 Memory processing complete", **kwargs)

    # --------------------------------------------------------------------------
    # API Query Helpers
    # --------------------------------------------------------------------------

    def _prepare_json_response(self, response_text: str) -> str:
        """
        Prepare API response for JSON parsing with minimal cleaning.

        Args:
            response_text: The raw response text from the API

        Returns:
            A minimally cleaned string ready for JSON parsing
        """
        if not response_text or not response_text.strip():
            return "[]"  # Return empty array for empty responses

        # Remove leading/trailing whitespace and markdown code block markers
        cleaned = re.sub(r"```json|```", "", response_text.strip())

        return cleaned

    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parses a JSON response from the API, assuming the input is a clean JSON string.

        This function strips leading/trailing whitespace and then attempts to decode
        the string directly. It retains error handling for JSON decoding errors and
        can handle responses that are either a list of objects or a single object
        containing a list (e.g., `{"memories": [...]}`).

        Args:
                response_text: The raw JSON string from the API.

        Returns:
                A list of dictionaries from the JSON response, or an empty list if
                parsing fails or the structure is incorrect.
        """
        try:
            # Strip whitespace and attempt to parse directly.
            cleaned_str = response_text.strip()
            if not cleaned_str:
                return []

            parsed_data = json.loads(cleaned_str)

            # Handle dicts that might contain the list (e.g., {"memories": []})
            if isinstance(parsed_data, dict):
                for key in ["memories", "operations", "memory_operations"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        logger.debug(f"Extracted list from dictionary key '{key}'")
                        return parsed_data[key]
                logger.debug("API returned a single dictionary; wrapping it in a list.")
                return [parsed_data]

            # Handle responses that are already a list
            elif isinstance(parsed_data, list):
                if all(isinstance(item, dict) for item in parsed_data):
                    logger.debug("API returned a valid list of dictionaries.")
                    return parsed_data
                else:
                    logger.warning(
                        "API returned a list with non-dictionary items. Discarding."
                    )
                    return []

            # Handle any other unexpected a Eunexpected type
            else:
                logger.warning(
                    f"Parsed JSON is not a dictionary or list, but {type(parsed_data)}. Discarding."
                )
                return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            if self.valves.verbose_logging:
                logger.error(
                    f"Raw response causing error:\n{self._truncate_log_lines(response_text)}"
                )
            return []
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during JSON parsing: {e}", exc_info=True
            )
            return []

    async def _query_api(self, provider: str, messages: List[Dict[str, Any]]) -> str:
        """
        Query LLM API with retry logic.

        Args:
            provider: The API provider ("OpenAI API" or "Ollama API")
            messages: Array of message objects with role and content

        Returns:
            The API response content as a string
        """
        # Prepare request based on provider
        if provider == "OpenAI API":
            url = f"{self.valves.openai_api_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.valves.openai_api_key}",
            }
            payload = {
                "model": self.valves.openai_model,
                "messages": messages,
                "temperature": self.valves.temperature,
                "max_tokens": self.valves.max_tokens,
            }
        else:  # Ollama API
            url = f"{self.valves.ollama_api_url.rstrip('/')}/api/chat"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.valves.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.valves.temperature,
                },
            }
            # Per instructions, remove num_ctx if present as it's handled externally.
            if "num_ctx" in payload.get("options", {}):
                del payload["options"]["num_ctx"]

        # Try request with retries
        for attempt in range(self.valves.max_retries + 1):
            try:
                async with self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.valves.request_timeout,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Extract content based on provider
                    if provider == "OpenAI API":
                        return str(data["choices"][0]["message"]["content"])
                    else:  # Ollama API
                        return str(data["message"]["content"])

            except Exception as e:
                if attempt < self.valves.max_retries:
                    logger.error(
                        f"API error (attempt {attempt+1}/{self.valves.max_retries+1}): {str(e)}"
                    )
                    await asyncio.sleep(self.valves.retry_delay)
                else:
                    logger.error(f"Max retries reached: {str(e)}")
                    return ""

    # --------------------------------------------------------------------------
    # Event Emitter Helpers
    # --------------------------------------------------------------------------

    async def _safe_emit(self, event: str, payload: Any, **kwargs) -> None:
        """
        Safely emit an event if the emitter is available.
        """
        emitter = kwargs.get("__event_emitter__")
        if not emitter:
            return

        # Prepare the payload
        full_payload = {}
        if event == "status":
            full_payload = {
                "type": "status",
                "data": {"description": str(payload), "done": True},
            }
        else:  # citation
            full_payload = {"type": "citation", **payload}

        try:
            await emitter(full_payload)
        except Exception as e:
            logger.warning(f"Failed to emit event '{event}': {e}")

    async def _send_citation(
        self,
        status_updates: List[str],
        user_id: str,
        citation_type: str = "processed",
        **kwargs,
    ) -> None:
        """
        Send a citation message. The status banner is only cleared for the final
        "processed" citation type, not for the intermediary "read" citation.
        """
        if not status_updates or not self.valves.show_status:
            return

        # Determine citation title and source based on type
        if citation_type == "read":
            title = "Memories Read"
            header = "Memories Read (sorted by relevance):"
        else:  # "processed"
            title = "Memories Processed"
            header = "Memories Processed (sorted by importance):"

        source_path = f"module://mrs/memories/{title.lower().replace(' ', '_')}"
        citation_message = f"{header}\n" + "\n".join(status_updates)

        payload = {
            "data": {
                "document": [citation_message],
                "metadata": [{"source": source_path, "html": False}],
                "source": {"name": title},
            }
        }

        if self.valves.verbose_logging:
            logger.info(f"Sending '{title}' citation for user {user_id}")

        await self._safe_emit("citation", payload, **kwargs)

    # --------------------------------------------------------------------------
    # Main Entry Points (Inlet/Outlet)
    # --------------------------------------------------------------------------

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming messages, retrieve relevant memories, and trigger async storage.
        """
        if (
            not body
            or not isinstance(body, dict)
            or not __user__
            or not body.get("messages")
        ):
            return body

        user_id = __user__.get("id")
        if not user_id:
            logger.error("User ID not found in __user__ object.")
            return body

        try:
            logger.info(f"Processing inlet request for user {user_id}")

            user_messages = [m for m in body["messages"] if m.get("role") == "user"]
            if not user_messages:
                return body

            last_user_message_content = user_messages[-1].get("content", "")
            if not last_user_message_content:
                return body

            logger.info(
                f"Processing user message: {last_user_message_content[:100]}..."
            )

            # --- Stage 1: Memory Retrieval (Synchronous) ---
            loop = asyncio.get_event_loop()
            db_memories = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, user_id
            )
            logger.info(f"Found {len(db_memories or [])} memories for user {user_id}")

            # Prepare kwargs for event emitting, ensuring they are available for the async task.
            kwargs = {
                "__event_emitter__": __event_emitter__,
            }

            await self._safe_emit(
                "status", "💭 Retrieving relevant memories...", **kwargs
            )

            relevant_memories = await self.get_relevant_memories(
                last_user_message_content, user_id, db_memories
            )

            if relevant_memories:
                formatted_memories, citation_memories = (
                    self._format_memories_for_context(relevant_memories)
                )
                self._update_message_context(body, formatted_memories)
                logger.info(
                    f"Added {len(relevant_memories)} relevant memories to context."
                )

                await self._send_citation(
                    citation_memories.split("\n")[1:],
                    user_id,
                    citation_type="read",
                    **kwargs,
                )
            else:
                logger.info("No relevant memories found for the current message.")
                await self._safe_emit(
                    "status", "🚫 No relevant memories found", **kwargs, final=True
                )

            # --- Stage 2: Memory Storage (Asynchronous/Synchronous) ---
            if self.valves.async_memory_storage:
                logger.info("Creating background task for memory storage")
                task = asyncio.create_task(
                    self._process_memory_storage(
                        user_id, body, relevant_memories, **kwargs
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                await self._process_memory_storage(
                    user_id, body, relevant_memories, **kwargs
                )

            logger.info("Inlet processing complete.")

        except Exception as e:
            logger.error(f"Error in inlet: {str(e)}", exc_info=True)

        return body

    async def outlet(self, body: dict) -> dict:
        """
        Process outgoing messages (called by Open WebUI).
        This stage is a pass-through, as memory storage is now initiated from the inlet.
        """
        # Memory storage is now handled in the inlet to ensure data availability.
        # This function remains for pipeline compatibility.
        if self.valves.verbose_logging:
            logger.info("Outlet is now a pass-through. Skipping processing.")
        return body

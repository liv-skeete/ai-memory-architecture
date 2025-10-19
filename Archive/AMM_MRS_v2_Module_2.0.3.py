"""
title: AMM - Recall & Storage
description: Adaptive Memory Manager - Recall & Storage Module for Open WebUI - Combines synchronous memory retrieval with asynchronous memory storage
author: Cody
version: 2.0.3
date: 2025-07-07
dev_note: "AI Coding Assistant: When making non-trivial code changes, update 'version', 'date', and add an entry under 'changes'."
changes:
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
  - v1.0.22 (2025-07-02):
      - Patched `_create_memory` to use normalization, preventing duplicate scores.
  - v1.0.21 (2025-06-30):
      - Made normalization in `_normalize_memory_content` iterative to handle multiple tags/scores.
  - v1.0.20 (2025-06-30):
      - Enhanced `_delete_memory` to robustly handle score and tag variations.
      - Added `_normalize_memory_content` helper to strip leading tags and trailing scores for reliable matching.
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
from open_webui.models.users import Users

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

# Debug mode flag - controls cleanup warnings
DEBUG_MODE = False


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
    def _calculate_retention_score(
        m, importance_weight: float, half_life_days: float
    ) -> float:
        inv_age = Filter._inv_age_exponential(m.updated_at, half_life_days)
        # Importance is now stored in the database, but we fall back to parsing for old memories
        importance = m.importance if hasattr(m, "importance") and m.importance else 0.5
        if not hasattr(m, "importance"):  # Fallback for old schema
            match = re.search(r"\[(\d+\.\d+)\]$", m.content)
            if match:
                importance = float(match.group(1))
        return (importance_weight * importance) + ((1 - importance_weight) * inv_age)

    async def _delete_excess_memories(self, user_id: str) -> int:
        """Remove excess memories and return the count of deleted items."""
        loop = asyncio.get_event_loop()
        mems = (
            await loop.run_in_executor(None, Memories.get_memories_by_user_id, user_id)
            or []
        )

        if len(mems) < self.valves.max_memories:
            if self.valves.verbose_logging:
                logger.info(f"No excess memories to delete for user {user_id}")
            return 0

        logger.info(
            f"User {user_id} has {len(mems)} memories; pruning to {self.valves.max_memories} using {self.valves.selection_method} method."
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
        while len(mems) >= self.valves.max_memories:
            m = mems.pop(0)
            if self.valves.verbose_logging:
                truncated_content = self._truncate_log_lines(m.content, max_lines=1)
                importance_score = getattr(m, "importance", 0.0)
                logger.info(
                    f"Deleting memory ID {m.id} (Importance: {importance_score:.2f}): '{truncated_content}'"
                )
            await loop.run_in_executor(
                None, Memories.delete_memory_by_id_and_user_id, m.id, user_id
            )
            deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} excess memories for user {user_id}")
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

        if self.valves.api_provider == "OpenAI API":
            if self.valves.verbose_logging:
                logger.info(
                    f"Querying OpenAI API with model: {self.valves.openai_model}"
                )
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            if self.valves.verbose_logging:
                logger.info(
                    f"Querying Ollama API with model: {self.valves.ollama_model}"
                )
            response = await self.query_ollama_api(system_prompt, current_message)

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

    def _format_memory_item(
        self, content: str, value: float, label: str = "score"
    ) -> str:
        """
        Format a single memory item with consistent formatting.

        Args:
            content: The memory content
            value: The relevance or importance score
            label: Label for the score (e.g., "relevance" or "importance")

        Returns:
            Formatted memory string (without additional bullets)
        """
        # This function should only ever format the content and score, without any status symbols.
        return f"{content} [{label}: {value:.1f}]"

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
                self._format_memory_item(mem.content, mem.relevance, "relevance")
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
        recent_messages: List[Dict[str, Any]],
        relevant_memories: List[ProcessingMemory],
    ) -> List[ProcessingMemory]:
        """
        Identify potential memories from the current message and recent context.

        Args:
            current_message: The current user message.
            recent_messages: List of recent messages (not currently used).
            relevant_memories: List of ProcessingMemory objects from Stage 1.

        Returns:
            A list of ProcessingMemory objects with 'content', 'importance', and 'tag'.
        """
        logger.info("Identifying potential memories from message")

        if not self.valves.identification_prompt:
            logger.error("Identification prompt is empty - cannot process memories")
            raise ValueError("Identification prompt is empty - module cannot function")

        formatted_relevant = self._format_relevant_memories(relevant_memories)

        if self.valves.verbose_logging and relevant_memories:
            logger.info(
                "Relevant memories passed to identification stage: %s",
                [
                    f"'{mem.content}' (Relevance: {mem.relevance:.2f})"
                    for mem in relevant_memories
                ],
            )

        system_prompt = self.valves.identification_prompt.format(
            current_message=current_message,
            relevant_memories=formatted_relevant,
        )

        if self.valves.verbose_logging:
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(
                f"System prompt for Identification (truncated):\n{truncated_prompt}"
            )

        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, current_message)
        else:
            response = await self.query_ollama_api(system_prompt, current_message)

        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info(f"Identification: Raw API response: {truncated_response}")

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

                    if importance >= self.valves.memory_importance_threshold:
                        logger.info(
                            f"Identified potential memory: '{mem_obj.content}' [importance: {mem_obj.importance:.2f}, tag: {mem_obj.tag}]"
                        )
                except (ValueError, TypeError):
                    logger.warning(f"Invalid memory format from LLM: {raw_mem}")

        if not potential_memories:
            logger.info("No potential memories identified.")

        return potential_memories

    async def _identify_and_prepare_memories(
        self, current_message: str, relevant_memories: List[ProcessingMemory]
    ) -> List[ProcessingMemory]:
        """
        Identifies potential memories and prepares memory operations (NEW or DELETE).
        Args:
            current_message: The current user message.
            relevant_memories: List of ProcessingMemory objects from Stage 1.
        Returns:
            A list of ProcessingMemory objects with their `operation` field set.
        """
        potential_memories = await self._identify_potential_memories(
            current_message,
            [],
            relevant_memories,
        )

        important_memories = [
            mem
            for mem in potential_memories
            if mem.importance
            and mem.importance >= self.valves.memory_importance_threshold
        ]

        if not important_memories:
            if self.valves.verbose_logging:
                logger.info(
                    "No potential memories met the importance threshold (%.2f). No operations needed.",
                    self.valves.memory_importance_threshold,
                )
            return []

        logger.info(
            f"Identified {len(important_memories)} important memories meeting threshold ({self.valves.memory_importance_threshold:.2f})."
        )

        for mem in important_memories:
            # Operation is determined by the tag.
            if mem.tag and mem.tag.lower() == "[delete]":
                mem.operation = "DELETE"
                # The content is already clean because the [Delete] was parsed out as a tag.
                # We just need to find the *next* tag, if it exists, to formulate the final content.
                content_to_delete, next_tag = self._strip_leading_tags(mem.content)
                mem.content = content_to_delete
                mem.tag = (
                    next_tag  # The tag is now the *second* tag, e.g., [Preferences]
                )
            else:
                mem.operation = "NEW"

        memory_operations = important_memories

        new_ops = [op for op in memory_operations if op.operation == "NEW"]
        delete_ops = [op for op in memory_operations if op.operation == "DELETE"]

        op_summary_parts = []
        if new_ops:
            op_summary_parts.append(f"{len(new_ops)} NEW")
        if delete_ops:
            op_summary_parts.append(f"{len(delete_ops)} DELETE")

        op_summary = " | ".join(op_summary_parts)
        if op_summary:
            logger.info(f"Prepared memory operations: {op_summary}")

        return memory_operations

    async def _execute_memory_operations(
        self,
        memory_operations: List[ProcessingMemory],
        __user__: dict,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> None:
        """
        Executes memory operations (creation and deletion).

        Args:
            memory_operations: List of ProcessingMemory objects with operations set.
            __user__: The user dictionary object.
            __event_emitter__: Optional event emitter for sending status updates.
        """
        new_ops = [op for op in memory_operations if op.operation == "NEW"]
        delete_ops = [op for op in memory_operations if op.operation == "DELETE"]

        logger.info(
            f"Executing {len(new_ops)} NEW and {len(delete_ops)} DELETE operations."
        )
        if not memory_operations:
            return

        status_updates_with_importance = []

        for mem_op in memory_operations:
            if mem_op.operation == "NEW":
                if mem_op.content:
                    if self.valves.verbose_logging:
                        logger.info(f"Attempting to create memory: {mem_op.content}")

                    # The `mem_op` is already a correctly formed ProcessingMemory object.
                    # We pass it directly to the creation function.
                    created_memory = await self._create_memory(mem_op, __user__)

                    if created_memory:
                        # The `created_memory` variable is just the ID.
                        # We must use the original `mem_op` object to build the citation.
                        tag_str = f"{mem_op.tag} " if mem_op.tag else ""
                        status = f"☑ {tag_str}{mem_op.content} [importance: {mem_op.importance:.1f}]"
                        logger.info(f"Successfully created memory: {status}")
                        status_updates_with_importance.append(
                            (status, mem_op.importance)
                        )
                    else:
                        status = f"☒ Failed to create memory: {mem_op.content}"
                        logger.warning(status)
                        status_updates_with_importance.append((status, 0))
                else:
                    logger.warning("Skipping NEW operation: Content is missing.")
            elif mem_op.operation == "DELETE":
                if mem_op.content:
                    if self.valves.verbose_logging:
                        logger.info(f"Attempting to delete memory: {mem_op.content}")
                    success = await self._delete_memory(mem_op, __user__)
                    if success:
                        # Ensure the tag is included in the deletion status.
                        tag_str = f"{mem_op.tag} " if mem_op.tag else ""
                        status = f"☒ {tag_str}{mem_op.content}"
                        logger.info(f"Successfully deleted memory: {status}")
                        status_updates_with_importance.append(
                            (status, mem_op.importance)
                        )
                    else:
                        status = f"☒ Failed to delete memory: {mem_op.content}"
                        logger.warning(status)
                        status_updates_with_importance.append((status, 0))
                else:
                    logger.warning("Skipping DELETE operation: Content is missing.")
            else:
                logger.warning(f"Skipping unknown operation: {mem_op.operation}")

        # Sort updates by importance, descending to prioritize more important memories
        sorted_status_updates = [
            status
            for status, _ in sorted(
                status_updates_with_importance, key=lambda x: x[1], reverse=True
            )
        ]

        # Send citation if there are any updates
        if self.valves.show_status and __event_emitter__ and sorted_status_updates:
            asyncio.create_task(
                self._send_citation(
                    __event_emitter__,
                    sorted_status_updates,
                    __user__.get("id"),
                    "Memories Processed",
                )
            )

    async def _create_memory(
        self, memory_to_create: ProcessingMemory, __user__: dict
    ) -> Optional[str]:
        """
        Create a new memory in the database from a ProcessingMemory object.

        Args:
            memory_to_create: The ProcessingMemory object to be saved.
            __user__: The user dictionary object.

        Returns:
            The ID of the created memory, or None if creation failed.
        """
        try:
            loop = asyncio.get_event_loop()

            # Format the memory back into a single string for the legacy database.
            full_content = memory_to_create.content
            if memory_to_create.tag:
                full_content = f"{memory_to_create.tag} {full_content}"
            if memory_to_create.importance is not None:
                full_content = f"{full_content} [{memory_to_create.importance:.1f}]"

            memory = await loop.run_in_executor(
                None,
                Memories.insert_new_memory,
                __user__["id"],
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
        while (match := tag_pattern.match(cleaned_content)):
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
        self, memory_to_delete: ProcessingMemory, __user__: dict
    ) -> bool:
        """
        Deletes a memory based on its content by finding a matching memory in the DB.

        Args:
            memory_to_delete: The ProcessingMemory object to delete. Its 'content'
                              should be the pure, untagged text.
            __user__: The user dictionary object.

        Returns:
            True if deletion was successful, False otherwise.
        """
        try:
            user_id = __user__.get("id")
            if not user_id:
                logger.warning("Cannot delete memory: User ID is missing.")
                return False

            loop = asyncio.get_event_loop()
            all_memories = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, user_id
            )

            content_to_delete = memory_to_delete.content.strip()

            for mem in all_memories:
                # Fully normalize the database content by stripping potential tags and scores
                db_content_no_scores = self._strip_trailing_scores(mem.content)
                db_content_no_tags, _ = self._strip_leading_tags(db_content_no_scores)

                if db_content_no_tags.strip() == content_to_delete:
                    logger.info(f"Found match for deletion: DB ID {mem.id}")
                    await loop.run_in_executor(
                        None, Memories.delete_memory_by_id_and_user_id, mem.id, user_id
                    )
                    return True  # Successfully found and deleted

            logger.warning(
                f"Could not find a matching memory to delete for content: '{content_to_delete}'"
            )
            return False  # No matching memory was found
        except Exception as e:
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            return False

    async def _process_memory_storage(
        self,
        current_message_content: str,
        __user__: dict,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
        relevant_memories: List[ProcessingMemory] = None,
    ) -> None:
        """
        Process memory storage asynchronously.
        Args:
            current_message_content: The current user message content.
            __user__: The user dictionary object.
            __event_emitter__: Optional event emitter for sending status updates.
            relevant_memories: Optional list of ProcessingMemory objects from Stage 1.
        """
        if relevant_memories is None:
            relevant_memories = []

        memory_operations = []

        try:
            logger.info("Processing memory storage")
            await self._delete_excess_memories(__user__.get("id"))
            memory_operations = await self._identify_and_prepare_memories(
                current_message_content, relevant_memories
            )
            if memory_operations:
                await self._execute_memory_operations(
                    memory_operations, __user__, __event_emitter__
                )
            else:
                logger.info("No memory operations to execute")
        except Exception as e:
            logger.error(
                f"Error during asynchronous memory storage: {str(e)}", exc_info=True
            )
        finally:
            logger.info("Memory storage processing completed")
            if self.valves.show_status and __event_emitter__:
                new_ops_count = len(
                    [op for op in memory_operations if op.operation == "NEW"]
                )
                delete_ops_count = len(
                    [op for op in memory_operations if op.operation == "DELETE"]
                )
                description = "☒ No important memories identified for storage"

                if new_ops_count > 0 and delete_ops_count > 0:
                    description = f"☑ Saved {new_ops_count} and deleted {delete_ops_count} memories"
                elif new_ops_count > 0:
                    description = f"☑ Saved {new_ops_count} important memories"
                elif delete_ops_count > 0:
                    description = f"☑ Deleted {delete_ops_count} memories"

                asyncio.create_task(
                    self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {"description": description, "done": True},
                        },
                    )
                )

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
        Parse the JSON response from the API, handling potential errors.

        Args:
            response_text: The raw response text from the API

        Returns:
            List of dictionaries representing memories or operations, or empty list on error.
        """
        try:
            # Attempt to find JSON within potential markdown code blocks
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            if match:
                json_str = match.group(1).strip()
                logger.debug("Extracted JSON from markdown block")
            else:
                # Assume the whole response might be JSON, try cleaning it
                json_str = response_text.strip()
                # This is a basic cleanup, might need refinement
                if not json_str.startswith(("[", "{")):
                    json_str = (
                        json_str[json_str.find("[") :]
                        if "[" in json_str
                        else json_str[json_str.find("{") :]
                    )
                if not json_str.endswith(("]", "}")):
                    json_str = (
                        json_str[: json_str.rfind("]") + 1]
                        if "]" in json_str
                        else json_str[: json_str.rfind("}") + 1]
                    )

                logger.debug("Attempting to parse response directly as JSON")

            if not json_str:
                logger.warning("Could not extract valid JSON content from response.")
                return []

            parsed_data = json.loads(json_str)

            if isinstance(parsed_data, dict):
                for key in ["memories", "operations", "memory_operations"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        logger.debug(f"Found list under key '{key}'")
                        return parsed_data[key]
                logger.debug("API returned a single dictionary, wrapping in list")
                return [parsed_data]
            elif isinstance(parsed_data, list):
                if all(isinstance(item, dict) for item in parsed_data):
                    logger.debug("API returned a list of dictionaries")
                    return parsed_data
                else:
                    logger.warning(
                        "API returned a list containing non-dictionary items. Discarding."
                    )
                    return []
            else:
                logger.warning(
                    f"Parsed JSON is not a dictionary or list, but {type(parsed_data)}. Discarding."
                )
                return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            if self.valves.verbose_logging:
                logger.error(
                    "Raw response causing error:\n%s",
                    self._truncate_log_lines(response_text),
                )
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}", exc_info=True)
            if self.valves.verbose_logging:
                logger.error(
                    "Raw response causing error:\n%s",
                    self._truncate_log_lines(response_text),
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

    async def query_openai_api(self, system_prompt: str, prompt: str) -> str:
        """
        Query OpenAI API with system prompt and user prompt.

        Args:
            system_prompt: The system prompt
            prompt: The user prompt

        Returns:
            The API response content as a string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._query_api("OpenAI API", messages)

    async def query_ollama_api(self, system_prompt: str, prompt: str) -> str:
        """
        Query Ollama API with system prompt and user prompt.

        Args:
            system_prompt: The system prompt
            prompt: The user prompt

        Returns:
            The API response content as a string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._query_api("Ollama API", messages)

    # --------------------------------------------------------------------------
    # Event Emitter Helpers
    # --------------------------------------------------------------------------

    async def _safe_emit(
        self,
        emitter: Callable[[Any], Awaitable[None]],
        data: Dict[str, Any],
    ) -> None:
        """
        Safely emit an event, handling missing emitter.

        Args:
            emitter: The event emitter function
            data: The data to emit
        """
        if not emitter:
            logger.debug("Event emitter not available")
            return

        try:
            await emitter(data)
            if isinstance(data, dict) and data.get("type") == "status":
                asyncio.create_task(self._delayed_clear_status(emitter, 7.0))
        except Exception as e:
            logger.error(f"Error in event emitter: {e}")

    async def _delayed_clear_status(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        delay_seconds: float,
    ) -> None:
        """
        Clear status after a delay, without blocking the main execution flow.

        Args:
            event_emitter: The event emitter function
            delay_seconds: Delay in seconds before clearing
        """
        if not event_emitter:
            return

        try:
            # Wait for the specified delay
            await asyncio.sleep(delay_seconds)

            # Clear the status
            asyncio.create_task(
                self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "",
                            "done": True,
                        },
                    },
                )
            )
        except Exception as e:
            logger.error(f"Error in delayed status clearing: {e}")
        finally:
            pass

    async def _send_citation(
        self,
        emitter: Callable[..., Awaitable[None]],
        status_updates: List[str],
        user_id: Optional[str],
        citation_type: str = "saved",
    ) -> None:
        """
        Send a citation message with memory information.

        Args:
            emitter: The event emitter function
            status_updates: List of status update strings
            user_id: The user ID
            citation_type: Type of citation ("saved", "read", "deleted", or custom title)
        """
        if not status_updates or not emitter:
            return

        # Determine citation title and source based on type
        if citation_type == "saved":
            # Categorize updates by icon
            deletion_updates = [u for u in status_updates if u.startswith("☒")]
            creation_updates = [u for u in status_updates if u.startswith("☑")]

            if deletion_updates and creation_updates:
                title = "Memory Operations"
                source_path = "module://mrs/memories/operations"
                header = "Memory Operations:"
            elif deletion_updates:
                title = "Memories Deleted"
                source_path = "module://mrs/memories/deleted"
                header = "Memories Deleted:"
            else:
                title = "Memories Saved"
                source_path = "module://mrs/memories/saved"
                header = "Memories Saved [sorted by importance]:"
        elif citation_type == "read":
            title = "Memories Read"
            source_path = "module://mrs/memories/read"
            header = "Memories Read [sorted by relevance]:"
        else:
            # Use the citation_type as the title for custom citations
            title = citation_type
            source_path = (
                f"module://mrs/memories/{citation_type.lower().replace(' ', '_')}"
            )
            header = f"{citation_type}:"

        # Compose citation message
        citation_message = f"{header}\n" + "\n".join(status_updates)

        if self.valves.verbose_logging:
            logger.info(
                f"Sending {citation_type} citation for user {user_id or 'UNKNOWN'}"
            )

        asyncio.create_task(
            self._safe_emit(
                emitter,
                {
                    "type": "citation",
                    "data": {
                        "document": [citation_message],
                        "metadata": [{"source": source_path, "html": False}],
                        "source": {"name": title},
                    },
                },
            )
        )

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
        Process incoming messages (called by Open WebUI).
        Stage 1: Enhances the message context with relevant memories (synchronous).
        Stage 2: Identifies and stores memories from user messages (asynchronous).

        Args:
            body: The message body
            __event_emitter__: Optional event emitter for notifications
            __user__: Optional user information

        Returns:
            The processed message body
        """
        # Basic validation
        if not body or not isinstance(body, dict) or not __user__:
            return body

        # Process only if we have messages
        if not body.get("messages"):
            return body

        try:
            # Always log inlet processing
            logger.info(f"Processing inlet request for user {__user__['id']}")

            # Check if there are any user messages
            user_messages = [m for m in body["messages"] if m.get("role") == "user"]
            if not user_messages:
                logger.info("No user messages found in request")
                return body

            # Reverse message order when multiple messages needed (newest first)
            n = self.valves.recent_messages_count
            if n > 1:
                # Reverse only the user messages
                user_messages = list(reversed(user_messages[-n:]))
            else:
                user_messages = user_messages[-n:]

            # Get the last user message (now first after reversal)
            last_user_message = user_messages[0].get("content", "")
            if not last_user_message:
                logger.info("Last user message is empty")
                return body

            # Essential log with summary
            # Format message with bullet point using helper function
            formatted_message = self.format_bulleted_list([last_user_message])
            if formatted_message:
                # Truncate formatted version instead of raw message
                truncated_message = (
                    formatted_message[:100] + "..."
                    if len(formatted_message) > 100
                    else formatted_message
                )
                # Formatted with bulleted list helper
                logger.info(f"Processing user message: {truncated_message}")
            else:
                logger.info("Processing user message: [empty]")

            # -----------------------------------------------------------------
            # Stage 1: Memory Retrieval (Synchronous)
            # -----------------------------------------------------------------

            # Get user object and memories
            loop = asyncio.get_event_loop()
            user = await loop.run_in_executor(
                None, Users.get_user_by_id, __user__["id"]
            )
            if not user:
                logger.info("User not found in database")
                return body

            db_memories = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, __user__["id"]
            )
            if not db_memories:
                logger.info(
                    f"No memories found for user {__user__['id']} - no memories to process"
                )

            # Always log memory count
            logger.info(f"Found {len(db_memories)} memories for user {__user__['id']}")

            # Emit status update if enabled
            if self.valves.show_status and __event_emitter__:
                if self.valves.verbose_logging:
                    logger.info("Emitting status: Retrieving relevant memories...")
                asyncio.create_task(
                    self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "💭 Retrieving relevant memories...",
                                "done": False,
                            },
                        },
                    )
                )

            # Get relevant memories using LLM-based scoring
            relevant_memories = await self.get_relevant_memories(
                last_user_message, __user__["id"], db_memories
            )

            # Format and update context if relevant memories found
            if relevant_memories:
                formatted_memories, formatted_memories_with_scores = (
                    self._format_memories_for_context(relevant_memories)
                )

                # Always log memory formatting
                logger.info(
                    f"Formatted {len(relevant_memories)} relevant memories for context"
                )

                # Update message context
                self._update_message_context(body, formatted_memories)

                # Log what's being sent to the assistant
                if self.valves.verbose_logging:
                    # In verbose mode, show full memory content with scores
                    truncated_memories = self._truncate_log_lines(
                        formatted_memories_with_scores
                    )
                    # Log relevant memories with scores in the same format as the citation
                    logger.info(
                        "Relevant memories with scores:\n%s",
                        formatted_memories_with_scores,
                    )
                    logger.info(f"Sending to assistant: {truncated_memories}")
                else:
                    # Always show count of memories
                    logger.info(
                        f"Sending to assistant: {len(relevant_memories)} relevant memories with content"
                    )

                # Send citation with memories sent to assistant
                if __event_emitter__:
                    logger.info("Sending citation with memories")

                    # Extract the memory items without the header
                    memory_items = formatted_memories_with_scores.split("\n")[1:]

                    # Send the citation using the shared method
                    asyncio.create_task(
                        self._send_citation(
                            __event_emitter__, memory_items, __user__.get("id"), "read"
                        )
                    )

                # Emit completion status if enabled
                if self.valves.show_status and __event_emitter__:
                    logger.info(
                        f"Emitting status: Retrieved {len(relevant_memories)} relevant memories"
                    )
                    asyncio.create_task(
                        self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": f"☑ Retrieved {len(relevant_memories)} relevant memories (threshold: {self.valves.relevance_threshold:.1f})",
                                    "done": True,
                                },
                            },
                        )
                    )

            else:
                # Log when no relevant memories are found
                logger.info("No relevant memories found for the current message")

                # Emit completion status if enabled
                if self.valves.show_status and __event_emitter__:
                    logger.info("Emitting status: No relevant memories found")
                    asyncio.create_task(
                        self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": "☒ No relevant memories found",
                                    "done": True,
                                },
                            },
                        )
                    )

            # -----------------------------------------------------------------
            # Stage 2: Memory Storage (Asynchronous)
            # -----------------------------------------------------------------

            # Determine the message content to analyze based on settings
            if self.valves.recent_messages_count > 1:
                start_index = max(
                    0, len(user_messages) - self.valves.recent_messages_count
                )
                context_messages = user_messages[start_index:]
                # Format messages as bulleted list
                context_messages_contents = [
                    m.get("content", "") for m in context_messages
                ]
                current_message_content = self.format_bulleted_list(
                    context_messages_contents
                )
                # Always log context selection
                logger.info(
                    f"Using last {len(context_messages)} messages for memory storage context"
                )
            else:
                # Format single message as bulleted list
                current_message_content = self.format_bulleted_list([last_user_message])
                # Always log context selection
                logger.info("Using the last message for memory storage context")

            # Emit status update if enabled
            if self.valves.show_status and __event_emitter__:
                asyncio.create_task(
                    self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "🧠 Processing memories...",
                                "done": False,
                            },
                        },
                    )
                )

            # Create a background task for memory storage, passing relevant memories from Stage 1.
            if self.valves.async_memory_storage:
                task = asyncio.create_task(
                    self._process_memory_storage(
                        current_message_content,
                        __user__,
                        __event_emitter__,
                        relevant_memories=relevant_memories,
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                logger.info("Asynchronous memory storage task created.")

            logger.info("Inlet processing complete.")

        except Exception as e:
            logger.error(f"Error in inlet: {str(e)}")
            # Always log traceback for errors
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """
        Process outgoing messages (called by Open WebUI).
        This stage is a pass-through, as memory storage is now initiated from the inlet.
        """
        # Memory storage is now handled in the inlet to ensure data availability.
        # This function remains for pipeline compatibility.
        if self.valves.verbose_logging:
            logger.info("Outlet is now a pass-through. Skipping processing.")
        return body

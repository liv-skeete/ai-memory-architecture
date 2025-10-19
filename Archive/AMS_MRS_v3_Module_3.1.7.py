"""
title: Adaptive Memory Stream
description: Memory Recall & Storage Module for Open WebUI - Combines synchronous memory retrieval with asynchronous memory storage
author: Cody
version: 3.1.7
date: 2025-07-25
dev_note: "AI Coding Assistant: When making non-trivial code changes, update 'version', 'date', and add an entry under 'changes'."
changes:
  - v3.1.7 (2025-07-25):
      - **Refactor**: Removed the `async_memory_storage` valve and synchronous processing pathway. Memory storage is now always performed asynchronously as a background task, simplifying the logic.
  - v3.1.6 (2025-07-25):
      - **Refactor**: Removed the `recent_messages_count` valve. The Stage 2 analysis now consistently processes only the most recent user message, simplifying the logic.
  - v3.1.5 (2025-07-24):
      - **Fix**: Made memory pruning universal by moving the call to `_delete_excess_memories` into a background task in the `outlet` that runs on every turn. This ensures the `max_memories` limit is enforced even when Stage 2 processing is disabled. Pruning is now decoupled from memory creation workflows.
  - v3.1.4 (2025-07-23):
      - **Refactor**: Removed the `_process_assistant_memories` helper method. The logic is now encapsulated in a nested async function directly within the `outlet`, improving code cohesion and removing a redundant method.
  - v3.1.3 (2025-07-23):
      - **Fix**: Updated the assistant memory parser to detect `💾` and `🗑️` directives anywhere within a line, not just at the beginning. This allows for more natural, inline memory creation and deletion by the assistant.
  - v3.1.2 (2025-07-23):
      - **Refactor**: Consolidated the citation and status update logic into a single, shared helper method (`_finalize_memory_processing`). This eliminates code duplication between the primary and assistant-led memory workflows, improving maintainability.
  - v3.1.1 (2025-07-23):
      - **Feature**: Implemented assistant-led memory deletion. The assistant can now emit a `🗑️` emoji followed by the exact content of a memory to trigger its deletion, adding to the existing `💾` creation command. The underlying execution logic has been unified for both operations.
  - v3.1.0 (2025-07-23):
      - **Feature**: Added a toggle valve (`enable_stage_two_processing`) to bypass Stage 2 memory analysis of user messages. When disabled, the module captures assistant-derived memories marked with a '💾' emoji from the `outlet` response, saving them verbatim. A second toggle (`show_assistant_mem_content`) controls whether the raw memory content is visible in the final response.
  - v3.0.2 (2025-07-22):
      - **Fix**: Implemented a robust "delete and rebuild" backfill process. If an inconsistency is detected between the primary DB and the vector store (e.g., mismatched counts, different embedding dimensions), the user's old vector collection is deleted and a new one is created from scratch. This is a definitive, one-time migration that guarantees perfect data integrity and resolves all legacy data corruption issues.
  - v3.0.1 (2025-07-22):
      - **Retrieval Logic**: Fixed an issue where candidate memories were not correctly sorted by semantic similarity after vector search, ensuring the LLM receives the most relevant candidates first.
  - v3.0.0 (2025-07-21):
      - **Hybrid Search**: Rearchitected retrieval to use vector search for candidate selection followed by LLM re-ranking, boosting capacity and speed.
  - v2.2.x (Legacy):
      - **API & Retention**: Patched Ollama `max_tokens` issues and protected recent memories from deletion.
      - **Logging & Citations**: Improved logging for purged memories and standardized citation formats.
      - **Multimodal**: Added robust handling for image data in messages to prevent crashes.
  - v2.1.x (Legacy):
      - **Core Logic**: Ensured Stage 2 memory storage always runs and hardened resource cleanup.
      - **Logging & Config**: Fixed logger initialization and improved verbose logging.
  - v2.0.x (Legacy):
      - **Architecture**: Refactored core to use `ProcessingMemory` dataclass, separating content from metadata.
      - **API & Pruning**: Consolidated API queries, improved memory pruning logic, and added robust scoring helpers.
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

import chromadb
from sentence_transformers import SentenceTransformer

from open_webui.models.memories import Memories

# Logger configuration
logger = logging.getLogger("ams_mrs")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler: clear any existing handlers and set our own to ensure consistent formatting.
if logger.hasHandlers():
    logger.handlers.clear()

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
        show_assistant_mem_content: bool = Field(
            default=False,
            description="Show assistant-derived memories in the final response (for debugging)",
        )
        # Debug settings
        verbose_logging: bool = Field(
            default=False,
            description="Enable detailed diagnostic logging",
        )
        max_log_lines: int = Field(
            default=100,
            description="Maximum number of lines to show in verbose logs for multi-line content",
        )
        # Memory retention settings
        max_memories: int = Field(
            default=300,
            description="Maximum number of memories to store",
        )
        selection_method: Literal["Oldest", "Hybrid"] = Field(
            default="Hybrid",
            description="Method for selecting which memories to drop",
        )
        max_age: int = Field(
            default=3,
            description="Half-life in days for exponential decay of memory recency",
        )
        importance_weight: float = Field(
            default=0.6,
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
        # Ollama settings
        ollama_api_url: str = Field(
            default="http://ollama:11434",
            description="Ollama API URL",
        )
        ollama_model: str = Field(
            default="qwen2.5:32b",
            description="Ollama model to use for memory processing",
        )
        # Common API settings
        temperature: float = Field(
            default=0.3,
            description="Temperature for API calls",
        )
        max_tokens: int = Field(
            default=4096,
            description="Maximum tokens for API calls",
        )
        request_timeout: int = Field(
            default=180,
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
        # Vector DB settings
        # Stage 1: Memory Retrieval settings
        enable_vector_search: bool = Field(
            default=True,
            description="Enable vector search pre-filtering for scalable memory retrieval.",
        )
        vector_search_top_k: int = Field(
            default=20,
            description="Number of candidate memories to retrieve from vector search for LLM re-ranking.",
        )
        relevance_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory relevance prompt with preserved formatting",
        )
        relevance_threshold: float = Field(
            default=0.3,
            description="Minimum relevance score (0.0-1.0) for memories to be included in context",
        )
        # Stage 2: Memory Storage settings
        enable_stage_two_processing: bool = Field(
            default=True,
            description="Enable Stage 2 memory processing of a user's message to identify new memories.",
        )
        identification_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory identification prompt with preserved formatting",
        )
        memory_importance_threshold: float = Field(
            default=0.3,
            description="Minimum importance threshold for identifying potential memories (0.0-1.0)",
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
        self._backfill_checked_users = set()

        # Initialize vector DB and embedding model
        self.vector_db_client = None
        self.embedding_model = None
        try:
            logger.info("Initializing ChromaDB client...")
            self.vector_db_client = chromadb.PersistentClient(path="data/vector_db")
            logger.info("ChromaDB client initialized.")

            logger.info("Loading embedding model: all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        except Exception as e:
            logger.error(
                f"Failed to initialize vector DB or embedding model: {e}", exc_info=True
            )
            self.vector_db_client = None
            self.embedding_model = None

    async def close(self) -> None:
        """Close the aiohttp session and cancel any background tasks with a shared timeout."""
        SHUTDOWN_TIMEOUT = 30.0

        try:
            async with asyncio.timeout(SHUTDOWN_TIMEOUT):
                # 1. Cancel background tasks
                if hasattr(self, "_background_tasks") and self._background_tasks:
                    tasks_to_cancel = list(self._background_tasks)
                    logger.info(
                        f"Cancelling {len(tasks_to_cancel)} pending background tasks..."
                    )
                    for task in tasks_to_cancel:
                        task.cancel()

                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                    self._background_tasks.clear()
                    logger.info("Background tasks cancelled.")

                # 2. Close the session within the same timeout budget
                if (
                    hasattr(self, "session")
                    and self.session
                    and not self.session.closed
                ):
                    logger.info("Closing aiohttp session...")
                    await self.session.close()
                    self.session = None
                    logger.info("Session closed.")

                # 3. Release embedding model from memory
                if hasattr(self, "embedding_model") and self.embedding_model:
                    logger.info("Releasing embedding model from memory...")
                    del self.embedding_model
                    self.embedding_model = None
                    logger.info("Embedding model released.")

        except TimeoutError:
            logger.warning(
                f"Shutdown timed out after {SHUTDOWN_TIMEOUT} seconds. Some resources may not be cleanly released."
            )

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
        """
        Calculate the retention score based on importance and age.
        Protects memories less than 1 day old from score-based deletion.
        """
        # If a memory is less than 1 day old (86400s), return a score higher
        # than the maximum possible (1.0) to protect it from score-based pruning.
        if (int(time.time()) - m.updated_at) < 86400:
            return 2.0

        inv_age = Filter._inv_age_exponential(m.updated_at, half_life_days)
        importance = Filter._get_memory_importance(m)
        return (importance_weight * importance) + ((1 - importance_weight) * inv_age)

    async def _delete_excess_memories(self, user_id: str, **kwargs) -> List[str]:
        """Remove excess memories, emit status, and return purged citations."""
        loop = asyncio.get_event_loop()
        mems = (
            await loop.run_in_executor(None, Memories.get_memories_by_user_id, user_id)
            or []
        )

        num_to_prune = len(mems) - self.valves.max_memories
        if num_to_prune <= 0:
            if self.valves.verbose_logging:
                logger.info(f"No excess memories to delete for user {user_id}")
            return []

        logger.info(
            f"User {user_id} has {len(mems)} memories; pruning to {self.valves.max_memories} using {self.valves.selection_method} method."
        )

        # Emit the new status message *before* purging.
        await self._safe_emit(
            "status", f"🔥️ Purging {num_to_prune} memories...", **kwargs
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

        purged_citations = []
        # Corrected loop logic to avoid deleting one too many.
        while len(mems) > self.valves.max_memories:
            m = mems.pop(0)

            # Essential logging for each purged memory
            importance_score = self._get_memory_importance(m)
            content_no_scores = self._strip_trailing_scores(m.content)
            content_clean, tag = self._strip_leading_tags(content_no_scores)
            citation_str = self._format_citation_line(
                symbol="🔥️",
                content=content_clean.strip(),
                tag=tag,
                score=importance_score,
                score_label="importance",
            )
            logger.info(f"Purged memory: {citation_str}")
            purged_citations.append(citation_str)

            if self.valves.verbose_logging:
                truncated_content = self._truncate_log_lines(m.content, max_lines=1)
                importance_score = self._get_memory_importance(m)
                logger.info(
                    f"Purging memory ID {m.id} (Importance: {importance_score:.2f}): '{truncated_content}'"
                )
            await loop.run_in_executor(
                None, Memories.delete_memory_by_id_and_user_id, m.id, user_id
            )

            # Also delete from the vector database
            if self.vector_db_client:
                try:
                    collection_name = f"user-memory-{user_id}"
                    collection = self.vector_db_client.get_or_create_collection(
                        name=collection_name
                    )

                    # Graceful delete: Check if the vector exists before trying to delete it.
                    # This prevents the benign, but noisy, "Delete of nonexisting embedding ID" warnings.
                    if collection.get(ids=[m.id]).get("ids"):
                        collection.delete(ids=[m.id])
                        logger.info(
                            f"Deleted memory {m.id} from vector collection '{collection_name}'"
                        )
                    elif self.valves.verbose_logging:
                        logger.info(
                            f"Skipped deleting vector for memory {m.id}: ID not found in collection."
                        )

                except Exception as e:
                    # Log error but don't block deletion from primary DB
                    logger.error(
                        f"Failed to delete memory {m.id} from vector DB: {e}",
                        exc_info=True,
                    )

        if purged_citations:
            logger.info(
                f"Finished purging {len(purged_citations)} excess memories for user {user_id}"
            )
            # Final status/citation is now handled by the calling function.

        return purged_citations

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

    @staticmethod
    def _sanitize_message_for_logging(content: any) -> str:
        """
        Sanitizes complex message content (e.g., list with image data) for clean logging.
        """
        if isinstance(content, list):
            # Reconstruct the message, replacing image URLs with a placeholder
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        parts.append("[Image]")  # Placeholder for image data
            return " ".join(filter(None, parts))
        return str(content)

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
        if self.valves.verbose_logging:
            logger.info(
                f"Processing {len(db_memories)} memories for relevance scoring."
            )

        memory_contents = [
            mem.content
            for mem in db_memories
            if hasattr(mem, "content") and mem.content
        ]

        if not memory_contents:
            logger.info("No valid memory contents found in database")
            return []

        formatted_memories = self.format_bulleted_list(memory_contents)

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
            logger.info(f"Relevance: Raw API response: {truncated_response}")

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

    async def _get_candidate_memories_vector_search(
        self, user_id: str, query_text: str, top_k: int
    ) -> List[str]:
        """
        Perform a vector search to get a list of candidate memory IDs.
        """
        if not all([self.vector_db_client, self.embedding_model]):
            logger.warning(
                "Vector DB client or embedding model not initialized. Skipping vector search."
            )
            return []

        try:
            collection_name = f"user-memory-{user_id}"
            collection = self.vector_db_client.get_or_create_collection(
                name=collection_name
            )
            logger.info(f"Performing vector search for top {top_k} candidate IDs.")

            query_embedding = self.embedding_model.encode(query_text).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents"],
            )

            # ChromaDB returns a dict with 'ids', 'documents', 'metadatas', etc.
            # We only need the IDs from the first (and only) query result.
            candidate_ids = results.get("ids", [[]])[0]

            if not candidate_ids:
                logger.info("Vector search returned no candidate IDs.")
                return []

            logger.info(f"Vector search found {len(candidate_ids)} candidate IDs.")
            return candidate_ids

        except Exception as e:
            logger.error(f"An error occurred during vector search: {e}", exc_info=True)
            return []

    async def get_relevant_memories(
        self,
        current_message: str,
        user_id: str,
        db_memories: Optional[List[Any]] = None,
    ) -> List[ProcessingMemory]:
        """
        Get memories relevant to the current context using either hybrid search or LLM-based scoring.

        Args:
            current_message: The current user message
            user_id: The user ID
            db_memories: List of memories from the database
            vector_db_client: The vector database client instance.
            embedding_function: The function to generate query embeddings.

        Returns:
            List of ProcessingMemory objects containing relevant memories.
        """
        if not db_memories:
            return []

        try:
            candidate_memories = db_memories

            # Stage 1a: Vector Search (Pre-filtering)
            if self.valves.enable_vector_search:
                logger.info("Using vector search to identify candidate memories.")
                candidate_ids = await self._get_candidate_memories_vector_search(
                    user_id=user_id,
                    query_text=current_message,
                    top_k=self.valves.vector_search_top_k,
                )

                if not candidate_ids:
                    logger.info(
                        "No candidate memories found from vector search. Aborting."
                    )
                    return []

                # Build a lookup map for efficient access to full memory objects.
                memory_map = {mem.id: mem for mem in db_memories}

                # Reconstruct the candidate list, preserving the semantic order from the vector search.
                # This ensures the LLM re-ranks based on similarity, not the original DB order.
                candidate_memories = [
                    memory_map[mem_id]
                    for mem_id in candidate_ids
                    if mem_id in memory_map
                ]

                logger.info(
                    f"Found {len(candidate_memories)} matching memories in database from {len(candidate_ids)} candidate IDs."
                )

            # Stage 1b: LLM Re-ranking
            return await self._get_relevant_memories_llm(
                current_message, candidate_memories
            )
        except Exception as e:
            logger.error(f"Error in get_relevant_memories: {str(e)}", exc_info=True)
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
            symbol: The leading symbol (e.g., '🔍', '💾', '🗑️').
            content: The memory content.
            tag: An optional classification tag.
            score: An optional score (relevance or importance).
            score_label: The label for the score (e.g., 'relevance').

        Returns:
            A formatted citation string for a single memory.
        """
        tag_str = f"{tag} " if tag else ""
        score_str = (
            f" [{score_label}: {score:.1f}]"
            if score is not None and score_label
            else ""
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
                    symbol="🔍",
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

        if self.valves.verbose_logging:
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(f"Identification prompt (truncated):\n{truncated_prompt}")

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

        count = len(important_memories)
        noun = "memory" if count == 1 else "memories"
        logger.info(
            f"Identified {count} important {noun} meeting threshold ({self.valves.memory_importance_threshold:.2f})."
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
                        symbol="💾",
                        content=mem_op.content,
                        tag=mem_op.tag,
                        score=mem_op.importance,
                        score_label="importance",
                    )
                    created_memories_for_citation.append(citation_str)
                    logger.info(f"Successfully created memory: {citation_str}")

            elif mem_op.operation == "DELETE" and mem_op.content:
                deleted_content, deleted_tag, importance_score = (
                    await self._delete_memory(mem_op, user_id, **kwargs)
                )
                if deleted_content:
                    citation_str = self._format_citation_line(
                        symbol="🗑️",
                        content=deleted_content,
                        tag=deleted_tag,
                        score=importance_score,
                        score_label="importance",
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

                # Also add the new memory to the vector database
                if self.vector_db_client and self.embedding_model:
                    try:
                        collection_name = f"user-memory-{user_id}"
                        collection = self.vector_db_client.get_or_create_collection(
                            name=collection_name
                        )
                        # Embed the full formatted content to include tags and score context
                        embedding = self.embedding_model.encode(full_content).tolist()
                        collection.add(
                            ids=[memory.id],
                            embeddings=[embedding],
                            documents=[full_content],
                        )
                        logger.info(
                            f"Added memory {memory.id} to vector collection '{collection_name}'"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to add memory {memory.id} to vector DB: {e}",
                            exc_info=True,
                        )

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
    ) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """
        Deletes a memory based on content, returning its content, tag, and importance.
        """
        try:
            if not user_id:
                logger.warning("Cannot delete memory: User ID is missing.")
                return None, None, None

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
                    # Extract importance *before* deleting
                    importance_score = self._get_memory_importance(mem)
                    await loop.run_in_executor(
                        None, Memories.delete_memory_by_id_and_user_id, mem.id, user_id
                    )

                    # Also delete from the vector database
                    if self.vector_db_client:
                        try:
                            collection_name = f"user-memory-{user_id}"
                            collection = self.vector_db_client.get_or_create_collection(
                                name=collection_name
                            )

                            # Graceful delete: Check if the vector exists before trying to delete it.
                            if collection.get(ids=[mem.id]).get("ids"):
                                collection.delete(ids=[mem.id])
                                logger.info(
                                    f"Deleted memory {mem.id} from vector collection '{collection_name}'"
                                )
                            elif self.valves.verbose_logging:
                                logger.info(
                                    f"Skipped deleting vector for memory {mem.id}: ID not found in collection."
                                )

                        except Exception as e:
                            logger.error(
                                f"Failed to delete memory {mem.id} from vector DB: {e}",
                                exc_info=True,
                            )

                    # Return the matched content, tag, and importance score
                    return content_to_delete, db_tag, importance_score

            logger.warning(
                f"Could not find a matching memory to delete for content: '{content_to_delete}'"
            )
            return None, None, None
        except Exception as e:
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            return None, None, None

    async def _finalize_memory_processing(
        self,
        created_citations: List[str],
        deleted_citations: List[str],
        purged_citations: List[str],
        user_id: str,
        **kwargs,
    ) -> None:
        """
        A centralized helper to sort, combine, and send all memory citations,
        and then emit the final completion status message. This ensures consistent
        feedback for all memory processing workflows.
        """
        all_citations = purged_citations + created_citations + deleted_citations

        if all_citations:

            def get_importance_from_citation(citation: str) -> float:
                """Helper to extract importance score from a citation string for sorting."""
                match = re.search(r"\[importance: (\d\.\d+)]$", citation)
                return float(match.group(1)) if match else 0.0

            # Sort each list individually by importance
            purged_citations.sort(key=get_importance_from_citation, reverse=True)
            created_citations.sort(key=get_importance_from_citation, reverse=True)
            deleted_citations.sort(key=get_importance_from_citation, reverse=True)

            # Combine the sorted lists, preserving grouping
            final_citations = created_citations + deleted_citations + purged_citations
            await self._send_citation(final_citations, user_id, "processed", **kwargs)
        else:
            logger.info(f"No new memory operations to report for user {user_id}.")
            await self._safe_emit(
                "status", "✅ Memories are up-to-date", **kwargs, final=True
            )

        # This block guarantees the final status message is always emitted after all operations.
        logger.info("Memory processing complete.")
        await self._safe_emit("status", "🏁 Memory processing complete", **kwargs)

    async def _process_memory_storage(
        self,
        user_id: str,
        body: dict,
        relevant_memories: List[ProcessingMemory],
        **kwargs,
    ) -> None:
        """
        Asynchronously process and store memories based on the conversation context.
        This is the primary, user-message-driven workflow.
        """
        try:
            logger.info(f"Starting async memory processing for user: {user_id}")

            # Pruning is now handled universally in the outlet.
            purged_citations = []

            # Determine message content to analyze
            history = body.get("messages", [])
            user_messages = [
                msg["content"] for msg in history if msg.get("role") == "user"
            ]
            # The content for analysis is now just the most recent user message.
            recent_messages = user_messages[-1:] if user_messages else []
            current_message_content = self.format_bulleted_list(recent_messages)

            # Identify and prepare memories
            memories_to_process = await self._identify_and_prepare_memories(
                current_message_content, relevant_memories, **kwargs
            )

            created_citations, deleted_citations = [], []
            if memories_to_process:
                created_citations, deleted_citations = (
                    await self._execute_memory_operations(
                        memories_to_process, user_id, **kwargs
                    )
                )

            # Use the centralized helper for final reporting.
            await self._finalize_memory_processing(
                created_citations,
                deleted_citations,
                purged_citations,
                user_id,
                **kwargs,
            )

        except Exception as e:
            logger.error(f"Error during async memory processing: {e}", exc_info=True)
            await self._safe_emit(
                "status", f"⚠️ Memory processing error", **kwargs, final=True
            )

    @staticmethod
    def _parse_assistant_response_for_memories(
        response_content: str,
    ) -> Tuple[List[ProcessingMemory], str]:
        """
        Parses the assistant's response to find and extract memory operations
        marked with '💾' (create) or '🗑️' (delete) emojis. It can detect these
        directives anywhere within a line.

        Args:
            response_content: The full text of the assistant's response.

        Returns:
            A tuple containing:
            - A list of `ProcessingMemory` objects for all found operations.
            - The cleaned response content with memory directives removed.
        """
        if "💾" not in response_content and "🗑️" not in response_content:
            return [], response_content

        lines = response_content.split("\n")
        operations_found = []
        cleaned_lines = []
        # This pattern is no longer anchored to the start of the line (`^`).
        op_pattern = re.compile(r"(💾|🗑️)\s*(.*)")

        for line in lines:
            match = op_pattern.search(line)
            if match:
                # The line is split at the emoji. Text before it is preserved.
                pre_text = line[: match.start()].strip()
                if pre_text:
                    cleaned_lines.append(pre_text)

                op_symbol, raw_content = match.groups()
                raw_content = raw_content.strip()

                if op_symbol == "💾":
                    content_no_scores = Filter._strip_trailing_scores(raw_content)
                    content_clean, tag = Filter._strip_leading_tags(content_no_scores)
                    importance = Filter._get_memory_importance(
                        type("obj", (object,), {"content": raw_content})()
                    )
                    if content_clean:
                        operations_found.append(
                            ProcessingMemory(
                                content=content_clean,
                                tag=tag,
                                importance=importance,
                                operation="NEW",
                            )
                        )
                elif op_symbol == "🗑️":
                    content_no_scores = Filter._strip_trailing_scores(raw_content)
                    content_clean, _ = Filter._strip_leading_tags(content_no_scores)
                    if content_clean:
                        operations_found.append(
                            ProcessingMemory(content=content_clean, operation="DELETE")
                        )
            else:
                cleaned_lines.append(line)

        return operations_found, "\n".join(cleaned_lines).strip()

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
                        if self.valves.verbose_logging:
                            logger.info(f"Extracted list from dictionary key '{key}'")
                        return parsed_data[key]
                if self.valves.verbose_logging:
                    logger.info(
                        "API returned a single dictionary; wrapping it in a list."
                    )
                return [parsed_data]

            # Handle responses that are already a list
            elif isinstance(parsed_data, list):
                if all(isinstance(item, dict) for item in parsed_data):
                    if self.valves.verbose_logging:
                        logger.info("API returned a valid list of dictionaries.")
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
                    "num_predict": self.valves.max_tokens,
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
            if not self.valves.show_status:
                return
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
            header = "Memories Processed:"

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
    # Background Tasks & Utilities
    # --------------------------------------------------------------------------

    async def _backfill_user_vectors(self, user_id: str, **kwargs) -> None:
        """
        Ensures all memories in the primary database have corresponding vectors.
        If any inconsistency is found, it performs a full, one-time rebuild
        of the user's vector collection to guarantee data integrity.
        """
        try:
            logger.info(f"Starting vector backfill check for user: {user_id}")
            if not self.vector_db_client or not self.embedding_model:
                logger.warning("Vector DB client/model not available for backfill.")
                return

            loop = asyncio.get_event_loop()
            all_db_mems = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, user_id
            )

            collection_name = f"user-memory-{user_id}"
            collection = self.vector_db_client.get_or_create_collection(
                name=collection_name
            )
            vector_count = collection.count()

            # If counts mismatch, it's the most reliable sign of an inconsistent state.
            if len(all_db_mems) != vector_count:
                logger.warning(
                    f"Inconsistency found for user {user_id}: "
                    f"Primary DB has {len(all_db_mems)} memories, but Vector DB has {vector_count}. "
                    "Performing a full rebuild of the vector collection."
                )

                # Nuke: Delete the old, inconsistent collection.
                self.vector_db_client.delete_collection(name=collection_name)
                # Pave: Re-create it fresh before backfilling.
                collection = self.vector_db_client.get_or_create_collection(
                    name=collection_name
                )

            # If there are no memories, there's nothing to do.
            if not all_db_mems:
                logger.info("No memories found in primary DB. Backfill not needed.")
                self._backfill_checked_users.add(user_id)
                return

            # Proceed with the standard backfill logic. If we just nuked the collection,
            # this will re-populate it with everything. If not, it will just add any deltas.
            db_mem_map = {mem.id: mem for mem in all_db_mems}
            db_ids = set(db_mem_map.keys())
            vector_ids = set(collection.get(include=[]).get("ids", []))

            missing_in_vector_db = db_ids - vector_ids
            if missing_in_vector_db:
                count = len(missing_in_vector_db)
                logger.info(f"Found {count} memories missing vectors. Creating now...")
                await self._safe_emit(
                    "status", f"⚙️ Backfilling {count} vector(s)...", **kwargs
                )

                mems_to_create = [db_mem_map[id] for id in missing_in_vector_db]
                contents = [mem.content for mem in mems_to_create]

                embeddings = await asyncio.to_thread(
                    self.embedding_model.encode, contents
                )

                collection.add(
                    ids=[mem.id for mem in mems_to_create],
                    embeddings=embeddings.tolist(),
                    documents=contents,
                )
                logger.info(f"Successfully created {count} vectors.")
                await self._safe_emit(
                    "status", "✅ Vector backfill complete.", **kwargs
                )
            else:
                logger.info(f"Vector DB is already up-to-date for user {user_id}.")

        except Exception as e:
            logger.error(f"Error during vector backfill: {e}", exc_info=True)
            await self._safe_emit(
                "status", "⚠️ Vector backfill error.", **kwargs, final=True
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

            # --- One-Time Vector DB Backfill Check (per session) ---
            if user_id not in self._backfill_checked_users:
                # This check runs once per user per server session. It is awaited to ensure
                # data integrity before any other operations proceed, preventing race conditions.
                await self._backfill_user_vectors(
                    user_id, __event_emitter__=__event_emitter__
                )
                self._backfill_checked_users.add(user_id)

            user_message_entries = [
                m for m in body["messages"] if m.get("role") == "user"
            ]
            if not user_message_entries:
                return body

            # Get the last user message and sanitize its content IN-PLACE.
            last_message_entry = user_message_entries[-1]
            original_content = last_message_entry.get("content", "")

            # Log a sanitized version of the message before processing
            log_safe_message = self._sanitize_message_for_logging(original_content)
            logger.info(f"Processing user message: {log_safe_message[:200]}...")

            # If the content is a list (multimodal), extract text and overwrite it.
            if isinstance(original_content, list):
                text_content = " ".join(
                    [
                        part["text"]
                        for part in original_content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                ).strip()
                # Modify the body in-place
                last_message_entry["content"] = text_content
                last_user_message_content = text_content
            else:
                last_user_message_content = str(original_content)

            # Early exit if there's no text content to process
            if not last_user_message_content.strip():
                logger.info(
                    "No text content found in message; skipping memory processing."
                )
                return body

            # --- Stage 1: Memory Retrieval (Synchronous) ---
            loop = asyncio.get_event_loop()
            db_memories = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, user_id
            )

            # Prepare kwargs for event emitting, ensuring they are available for the async task.
            kwargs = {"__event_emitter__": __event_emitter__}

            relevant_memories = []
            if db_memories:
                logger.info(f"Found {len(db_memories)} memories for user {user_id}")
                # Stage 1 runs only if there are existing memories to check against.
                await self._safe_emit(
                    "status", "💭 Retrieving relevant memories...", **kwargs
                )
                relevant_memories = await self.get_relevant_memories(
                    current_message=last_user_message_content,
                    user_id=user_id,
                    db_memories=db_memories,
                )

                if relevant_memories:
                    formatted_memories, citation_memories = (
                        self._format_memories_for_context(relevant_memories)
                    )
                    self._update_message_context(body, formatted_memories)
                    count = len(relevant_memories)
                    noun = "memory" if count == 1 else "memories"
                    logger.info(f"Added {count} relevant {noun} to context.")
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
            else:
                # This is a new user or a user with no memories.
                # Skip Stage 1 and directly proceed to Stage 2 for potential memory creation.
                logger.info("No prior memories found. Skipping relevance check.")

            # --- Stage 2: Memory Storage (Asynchronous) ---
            if self.valves.enable_stage_two_processing:
                logger.info("Creating background task for memory storage")
                task = asyncio.create_task(
                    self._process_memory_storage(
                        user_id, body, relevant_memories, **kwargs
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                logger.info(
                    "Stage 2 processing is disabled; outlet will handle assistant-derived memories."
                )

            logger.info("Inlet processing complete.")

        except Exception as e:
            logger.error(f"Error in inlet: {str(e)}", exc_info=True)

        return body

    async def outlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handles all post-response memory operations. This includes universal
        memory pruning and processing of assistant-derived memories when Stage 2
        is disabled.
        """
        # Basic validation for required data
        if not all([body, __user__, body.get("messages")]):
            return body

        user_id = __user__.get("id")
        if not user_id:
            logger.error("User ID not found in __user__ object for outlet processing.")
            return body

        try:
            logger.info(f"Processing outlet request for user {user_id}")
            kwargs = {"__event_emitter__": __event_emitter__}

            memory_operations = []
            cleaned_content = assistant_message = None

            # Only parse assistant messages if Stage 2 is disabled
            if not self.valves.enable_stage_two_processing:
                assistant_message = next(
                    (
                        m
                        for m in reversed(body["messages"])
                        if m.get("role") == "assistant"
                    ),
                    None,
                )

                if assistant_message and isinstance(assistant_message.get("content"), str):
                    (
                        memory_operations,
                        cleaned_content,
                    ) = self._parse_assistant_response_for_memories(
                        assistant_message["content"]
                    )

            # Universal post-processing task
            async def run_universal_post_processing():
                """
                A centralized task that runs after every response. It handles
                memory pruning and, if Stage 2 is disabled, processes memories
                from the assistant's response.
                """
                try:
                    created_citations, deleted_citations = [], []

                    # If stage 2 is disabled, handle assistant-derived memories
                    if not self.valves.enable_stage_two_processing and memory_operations:
                        op_count = len(memory_operations)
                        noun = "operation" if op_count == 1 else "operations"
                        logger.info(f"Found {op_count} assistant-derived memory {noun}.")
                        (
                            created_citations,
                            deleted_citations,
                        ) = await self._execute_memory_operations(
                            memory_operations, user_id, **kwargs
                        )
                    
                    # Universal pruning runs every time, after any other operations.
                    purged_citations = await self._delete_excess_memories(
                        user_id, **kwargs
                    )

                    # If Stage 2 is disabled, we must finalize here. If Stage 2 was
                    # enabled, its own background task handles its finalization.
                    if not self.valves.enable_stage_two_processing:
                        await self._finalize_memory_processing(
                            created_citations,
                            deleted_citations,
                            purged_citations,
                            user_id,
                            **kwargs,
                        )
                    # If Stage 2 was enabled, we only report if pruning occurred.
                    # The main inlet task is responsible for the final "complete" status.
                    elif purged_citations:
                        await self._send_citation(
                            purged_citations, user_id, "processed", **kwargs
                        )

                except Exception as e:
                    logger.error(
                        f"Error in universal outlet processing task: {e}", exc_info=True
                    )
                    await self._safe_emit(
                        "status", "⚠️ Memory processing error", **kwargs, final=True
                    )

            task = asyncio.create_task(run_universal_post_processing())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            # Overwrite the response content if assistant ops were found and hiding is enabled
            if (
                assistant_message
                and memory_operations
                and not self.valves.show_assistant_mem_content
            ):
                assistant_message["content"] = cleaned_content
                logger.info("Hiding assistant memory content from final response.")

        except Exception as e:
            logger.error(f"Error in outlet: {str(e)}", exc_info=True)

        return body

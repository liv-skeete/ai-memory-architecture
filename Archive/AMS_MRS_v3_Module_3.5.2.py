"""
title: Adaptive Memory Stream (MRS)
description: A memory module for Open WebUI where the `inlet` handles memory retrieval and the `outlet` handles memory creation, deletion, and pruning based on assistant commands.
author: Cody
version: 3.5.2
date: 2025-08-01
dev_note: "AI Coding Assistant: When making non-trivial code changes, update 'version', 'date', and add an entry under 'changes'."
changes:
  - v3.5.2 (2025-08-01):
      - **Refactor**: Removed the deprecated `include_assistant_message_in_vector_search` valve and hardcoded the query to always include the assistant's message for improved context.
  - v3.5.1 (2025-08-01):
      - **Refactor**: Removed the unused `_parse_json_response` method, made obsolete by the new index-based re-ranking logic that simplified JSON handling.
  - v3.5.0 (2025-08-01):
      - **Refactor**: Overhauled the LLM re-ranking validation to use a robust, index-based mapping strategy (`Index-Validate-Preserve`). Instead of comparing memory content, the system now sends indexed memories to the LLM and validates only the returned index. This eliminates a significant point of fragility by making the process completely immune to benign LLM content modifications (e.g., added punctuation), providing a more elegant and resilient solution.
  - v3.4.0 (2025-07-31):
      - **Feature**: Implemented chunked re-ranking to handle large numbers of candidate memories without overwhelming the LLM. This fixes an issue where retrieval results would drop when `vector_search_top_k` was set to a high value. Added `rerank_chunk_size` valve to control the chunk size.
  - v3.3.x (2025-07-30):
      - **Feature & Refactor**: Added valves to control LLM re-ranking (`enable_llm_search`, `llm_search_top_k`) and removed the `relevance_threshold` in favor of a simpler top-k selection. Fixed an issue with result limits when LLM search is disabled.
  - v3.2.x (2025-07-25...26):
      - **Refactor**: Removed Stage 2 (LLM-based memory identification) processing entirely, simplifying the architecture and clarifying the roles of the `inlet` (retrieval) and `outlet` (writing/pruning).
  - v3.1.x (2025-07-23...25):
      - **Refactor & Fixes**: Implemented numerous simplifications, including making memory storage always asynchronous, improving the memory pruning cycle, unifying assistant command parsing (`💾`, `🗑️`), and consolidating helper methods.
  - v3.0.x (2025-07-21...22):
      - **Core Architecture**: Rearchitected the module for hybrid search (vector search + LLM re-ranking) and implemented a robust "delete and rebuild" data backfill process to ensure vector store integrity.
  - v2.x (Legacy):
      - **Foundation**: Established core module architecture, including the `ProcessingMemory` dataclass, API query logic, memory pruning, citations, and robust handling for multimodal content.
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

    # This regex is used to find and strip trailing importance scores from memories.
    # It matches a pattern like '[...]' or '(...)' that contains a float (e.g., '[1.0]', '(0.8)').
    _score_pattern = re.compile(r"\s*[\(\[](\d+(?:\.\d+)?)[\)\]]\s*$")

    class Valves(BaseModel):
        # Set processing priority
        priority: int = Field(
            default=2,
            description="Priority level for the filter operations",
        )
        # UI settings
        show_status: bool = Field(
            default=True, description="Show memory operation status in chat"
        )
        show_assistant_mem_content: bool = Field(
            default=True,
            description="Show assistant-derived memories operations in response",
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
            default=1000,
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
            description="Weight for importance when pruning (between 0.0 and 1.0)",
        )
        # API configuration
        api_provider: Literal["OpenAI API", "Ollama API"] = Field(
            default="OpenAI API",
            description="LLM API provider for memory processing",
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
            default="qwen2.5:14b",
            description="Ollama model to use for memory processing",
        )
        ollama_num_ctx: int = Field(
            default=8192,
            description="Ollama context window size",
        )
        # Common API settings
        max_tokens: int = Field(
            default=2048,
            description="Maximum tokens for API calls",
        )
        temperature: float = Field(
            default=0.3,
            description="Temperature for API calls",
        )
        request_timeout: int = Field(
            default=20,
            description="Timeout for API requests (seconds)",
        )
        max_retries: int = Field(
            default=2,
            description="Maximum number of retries for API calls",
        )
        retry_delay: float = Field(
            default=3.0,
            description="Delay between retries (seconds)",
        )
        # Vector DB settings
        # Stage 1: Memory Retrieval settings
        enable_vector_search: bool = Field(
            default=True,
            description="Enable vector search pre-filtering",
        )
        vector_search_top_k: int = Field(
            default=100,
            description="Number of memories to retrieve from vector search for LLM re-ranking",
        )
        enable_llm_search: bool = Field(
            default=True,
            description="Enable LLM-based re-ranking of memories from vector search",
        )
        relevance_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory relevance prompt (required)",
        )
        llm_search_top_k: int = Field(
            default=10,
            description="Number of memories to send to assisatnt for context",
        )
        rerank_chunk_size: int = Field(
            default=25,
            description="The size of each chunk for LLM re-ranking",
        )

    def __init__(self) -> None:
        """
        Initialize the Memory Recall & Storage module.
        """
        # Initialize with empty prompts - must be set via update_valves
        try:
            self.valves = self.Valves(
                relevance_prompt=""  # Empty string to start - must be set via update_valves
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
            # Use the centralized score pattern for consistency.
            match = Filter._score_pattern.search(m.content)
            if match:
                try:
                    # The score is in the first capturing group.
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
            citation_str = self._format_citation_line(
                symbol="🔥️",
                content=m.content.strip(),
                tag=None,  # Tags are part of the content string
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
        self, user_message: str, assistant_message: str, db_memories: List[Any]
    ) -> List[ProcessingMemory]:
        """
        Get memories relevant to the current context using a chunked, parallel, and
        validated 'Index-Validate-Preserve' strategy.

        Args:
            user_message: The current user message.
            assistant_message: The preceding assistant message.
            db_memories: List of candidate memories from the database.

        Returns:
            A list of validated ProcessingMemory objects with original, verbatim 'content' and 'relevance' populated.
        """
        if not db_memories:
            return []

        if self.valves.verbose_logging:
            logger.info(
                f"Processing {len(db_memories)} memories for relevance scoring."
            )

        # 1. Split candidate memories into chunks
        chunk_size = self.valves.rerank_chunk_size
        memory_chunks = [
            db_memories[i : i + chunk_size]
            for i in range(0, len(db_memories), chunk_size)
        ]
        logger.info(
            f"Splitting {len(db_memories)} memories into {len(memory_chunks)} chunks of size {chunk_size}."
        )

        # 2. Create parallel API call tasks for each chunk
        tasks_with_chunks = []
        for i, chunk in enumerate(memory_chunks):
            if not chunk:
                continue

            # Format memories with a 1-based index for the LLM prompt.
            # This index is what the LLM will return.
            formatted_memories = "\n".join(
                [f"{idx}. {mem.content}" for idx, mem in enumerate(chunk, 1)]
            )

            system_prompt = self.valves.relevance_prompt.format(
                user_message=user_message,
                assistant_message=assistant_message,
                memories=formatted_memories,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            if self.valves.verbose_logging:
                model = (
                    self.valves.openai_model
                    if self.valves.api_provider == "OpenAI API"
                    else self.valves.ollama_model
                )
                logger.info(
                    f"Creating API task for chunk {i+1}/{len(memory_chunks)} with model {model}"
                )

            api_task = self._query_api(self.valves.api_provider, messages)
            # Pair the task with its original chunk data to prevent misalignment after asyncio.gather
            tasks_with_chunks.append((api_task, chunk))

        # 3. Execute all tasks concurrently and process results
        api_results = await asyncio.gather(
            *[task for task, _ in tasks_with_chunks], return_exceptions=True
        )

        all_reranked_memories = []
        for i, (response, original_chunk) in enumerate(zip(api_results, [chunk for _, chunk in tasks_with_chunks])):
            if isinstance(response, Exception):
                logger.error(
                    f"API call for chunk {i+1} failed: {response}", exc_info=response
                )
                continue
            
            if not original_chunk:
                continue

            if self.valves.verbose_logging:
                truncated_response = self._truncate_log_lines(response)
                logger.info(
                    f"Relevance (Chunk {i+1}): Raw API response: {truncated_response}"
                )

            prepared_response = self._prepare_json_response(response)
            try:
                memory_data = json.loads(prepared_response)
                if not isinstance(memory_data, list):
                    continue

                for mem_info in memory_data:
                    if (
                        isinstance(mem_info, dict)
                        and "index" in mem_info
                        and "relevance" in mem_info
                    ):
                        try:
                            # --- VALIDATE & PRESERVE ---
                            # The LLM returns a 1-based index; convert to 0-based.
                            mem_index = int(mem_info["index"]) - 1
                            relevance = float(mem_info["relevance"])

                            # Validate if the index is within the bounds of this chunk.
                            if 0 <= mem_index < len(original_chunk):
                                # The index is valid. Retrieve the pristine, original memory.
                                original_memory = original_chunk[mem_index]
                                all_reranked_memories.append(
                                    ProcessingMemory(
                                        content=original_memory.content,  # Preserve original
                                        relevance=relevance,
                                    )
                                )
                            else:
                                if self.valves.verbose_logging:
                                        logger.warning(
                                            f"LLM returned an out-of-bounds index: {mem_info['index']}. Chunk size is {len(original_chunk)}. Discarding."
                                        )

                        except (ValueError, TypeError) as e:
                             logger.warning(
                                f"Invalid index or relevance score from LLM. Raw data: {mem_info}. Error: {e}"
                            )

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding LLM JSON response for chunk {i+1}: {e}")
                if self.valves.verbose_logging:
                    logger.error(f"Problematic JSON content: {prepared_response}")

        # 5. Sort the final aggregated list by relevance
        all_reranked_memories.sort(key=lambda x: x.relevance, reverse=True)
        total_memories = len(all_reranked_memories)

        # 6. Return the top N memories as defined by llm_search_top_k
        top_k_memories = all_reranked_memories[: self.valves.llm_search_top_k]

        logger.info(
            f"LLM re-ranking complete. Selected top {len(top_k_memories)} memories from {total_memories} validated candidates across {len(memory_chunks)} chunks."
        )

        if self.valves.verbose_logging:
            for i, mem in enumerate(top_k_memories):
                logger.info(
                    f"  Top {i+1}: {mem.content} [relevance: {mem.relevance:.2f}]"
                )

        return top_k_memories

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

            if self.valves.verbose_logging:
                # In verbose mode, log the full query text to aid debugging.
                logger.info(f"Vector search query text:\n---\n{query_text}\n---")
            else:
                # In standard mode, log a truncated version of the query text.
                truncated_query = self._truncate_log_lines(query_text, max_lines=5)
                logger.info(f"Vector search using query: {truncated_query}")

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
        user_message: str,
        assistant_message: str,
        user_id: str,
        db_memories: Optional[List[Any]] = None,
    ) -> List[ProcessingMemory]:
        """
        Get memories relevant to the current context using either hybrid search or LLM-based scoring.

        Args:
            user_message: The current user message.
            assistant_message: The preceding assistant message.
            user_id: The user ID.
            db_memories: List of all memories from the database.

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

                # Always combine assistant and user messages for the vector search query for better context.
                query_text = f"{assistant_message}\n{user_message}"

                candidate_ids = await self._get_candidate_memories_vector_search(
                    user_id=user_id,
                    query_text=query_text,
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

            # Stage 1b: LLM Re-ranking or Direct Selection
            if self.valves.enable_llm_search:
                logger.info("Using LLM to re-rank candidate memories.")
                return await self._get_relevant_memories_llm(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    db_memories=candidate_memories,
                )
            else:
                logger.info(
                    "LLM search is disabled. Returning top vector search results directly."
                )
                if not self.valves.enable_vector_search:
                    logger.warning(
                        "Both vector search and LLM search are disabled. No memories will be retrieved."
                    )
                    return []

                # When LLM re-ranking is disabled, we return all candidates retrieved by the vector search.
                # The number of these candidates is controlled by `vector_search_top_k`.
                final_memories = [
                    ProcessingMemory(content=mem.content, relevance=None)
                    for mem in candidate_memories
                ]
                logger.info(
                    f"Selected {len(final_memories)} memories directly from vector search (controlled by vector_search_top_k)."
                )
                return final_memories
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
            # Context gets the original, verbatim memory content
            context_memories += "\n" + "\n".join(
                [mem.content for mem in relevant_memories]
            )
            # Citations also use the original, verbatim memory content
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

    async def _delete_memory(
        self, memory_to_delete: ProcessingMemory, user_id: str, **kwargs
    ) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """
        Deletes a memory based on an exact content match.
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
                # Perform an exact match on the full, raw content.
                if mem.content.strip() == content_to_delete:
                    logger.info(f"Found exact match for deletion: DB ID {mem.id}")
                    importance_score = self._get_memory_importance(mem)
                    await loop.run_in_executor(
                        None, Memories.delete_memory_by_id_and_user_id, mem.id, user_id
                    )

                    if self.vector_db_client:
                        try:
                            collection_name = f"user-memory-{user_id}"
                            collection = self.vector_db_client.get_or_create_collection(
                                name=collection_name
                            )
                            if collection.get(ids=[mem.id]).get("ids"):
                                collection.delete(ids=[mem.id])
                                logger.info(
                                    f"Deleted memory {mem.id} from vector collection '{collection_name}'"
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to delete memory {mem.id} from vector DB: {e}",
                                exc_info=True,
                            )
                    # For citation, we return the full original content and its score.
                    # Tags are no longer handled as separate entities.
                    return mem.content, None, importance_score

            logger.warning(
                f"Could not find an exact match to delete for content: '{content_to_delete}'"
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

    @staticmethod
    def _parse_assistant_response_for_memories(
        response_content: str,
    ) -> Tuple[List[ProcessingMemory], str]:
        """
        Parses the assistant's response to find and extract memory operations
        marked with '💾' (create) or '🗑️' (delete) emojis.

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
        # This pattern finds the emoji and captures the rest of the line.
        op_pattern = re.compile(r"(💾|🗑️)\s*(.*)")

        for line in lines:
            match = op_pattern.search(line)
            if match:
                # Preserve any text on the line that appeared before the emoji directive.
                pre_text = line[: match.start()].strip()
                if pre_text:
                    cleaned_lines.append(pre_text)

                op_symbol, raw_content = match.groups()
                content = raw_content.strip()

                if not content:
                    continue

                if op_symbol == "💾":
                    # For new memories, we extract the importance score but keep the content as is.
                    # Tags are now considered part of the content itself.
                    importance = Filter._get_memory_importance(
                        type("obj", (object,), {"content": content})()
                    )
                    operations_found.append(
                        ProcessingMemory(
                            content=content,
                            tag=None,  # Tags are no longer parsed separately
                            importance=importance,
                            operation="NEW",
                        )
                    )
                elif op_symbol == "🗑️":
                    # For deletions, we use the exact, full content for matching.
                    operations_found.append(
                        ProcessingMemory(content=content, operation="DELETE")
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
                    "num_ctx": self.valves.ollama_num_ctx,
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
        Handles memory retrieval. It fetches memories relevant to the user's
        message and injects them into the context for the assistant.
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
            logger.info(f"Processing inlet for user {user_id}: Memory Retrieval")

            # --- One-Time Vector DB Backfill Check (per session) ---
            if user_id not in self._backfill_checked_users:
                await self._backfill_user_vectors(
                    user_id, __event_emitter__=__event_emitter__
                )
                self._backfill_checked_users.add(user_id)

            # --- Text Extraction from Message ---
            all_messages = body.get("messages", [])
            user_message_entries = [m for m in all_messages if m.get("role") == "user"]
            if not user_message_entries:
                return body

            last_user_message_entry = user_message_entries[-1]
            user_message_index = all_messages.index(last_user_message_entry)

            # Find the last assistant message *before* the last user message.
            last_assistant_message_content = ""
            if user_message_index > 0:
                # Search backwards from the user message for the first assistant message.
                for i in range(user_message_index - 1, -1, -1):
                    if all_messages[i].get("role") == "assistant":
                        content = all_messages[i].get("content", "")
                        # Ensure content is a string, then assign and break.
                        if isinstance(content, str):
                            last_assistant_message_content = content
                        break  # Found the most recent assistant message.

            original_content = last_user_message_entry.get("content", "")
            log_safe_message = self._sanitize_message_for_logging(original_content)
            logger.info(f"Processing user message: {log_safe_message[:200]}...")

            if isinstance(original_content, list):
                text_content = " ".join(
                    [
                        part["text"]
                        for part in original_content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                ).strip()
                last_user_message_entry["content"] = text_content
                last_user_message_content = text_content
            else:
                last_user_message_content = str(original_content)

            if not last_user_message_content.strip():
                logger.info("No text content in message; skipping memory retrieval.")
                return body

            # --- Stage 1: Memory Retrieval (Synchronous) ---
            kwargs = {"__event_emitter__": __event_emitter__}
            loop = asyncio.get_event_loop()
            db_memories = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, user_id
            )

            if not db_memories:
                logger.info("No memories found for user. Skipping retrieval.")
                return body

            logger.info(f"Found {len(db_memories)} memories. Starting retrieval...")
            await self._safe_emit(
                "status", "💭 Retrieving relevant memories...", **kwargs
            )

            relevant_memories = await self.get_relevant_memories(
                user_message=last_user_message_content,
                assistant_message=last_assistant_message_content,
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
                    citation_memories.split("\n")[1:], user_id, "read", **kwargs
                )
            else:
                logger.info("No relevant memories found.")
                await self._safe_emit(
                    "status", "🚫 No relevant memories found", **kwargs, final=True
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
        Handles all memory writing operations: creation, deletion, and pruning.
        This method parses the assistant's final response for memory commands
        (`💾`, `🗑️`) and executes them as a background task.
        """
        if not all([body, __user__, body.get("messages")]):
            return body

        user_id = __user__.get("id")
        if not user_id:
            logger.error("User ID not found, cannot process outlet.")
            return body

        try:
            logger.info(f"Processing outlet for user {user_id}: Memory Storage")
            kwargs = {"__event_emitter__": __event_emitter__}

            assistant_message = next(
                (m for m in reversed(body["messages"]) if m.get("role") == "assistant"),
                None,
            )

            if not assistant_message or not isinstance(
                assistant_message.get("content"), str
            ):
                return body

            (
                memory_operations,
                cleaned_content,
            ) = self._parse_assistant_response_for_memories(
                assistant_message["content"]
            )

            # Create a background task to handle memory operations without blocking.
            async def run_memory_writer_task():
                """
                A centralized task to handle all memory writing:
                1. Executes `💾` and `🗑️` operations from the assistant's response.
                2. Performs universal memory pruning to enforce `max_memories`.
                3. Sends a single, consolidated citation for all operations.
                """
                try:
                    created_citations, deleted_citations = [], []
                    if memory_operations:
                        op_count = len(memory_operations)
                        noun = "operation" if op_count == 1 else "operations"
                        logger.info(
                            f"Found {op_count} assistant-derived memory {noun}."
                        )
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

                    # Finalize and report all changes (created, deleted, purged).
                    await self._finalize_memory_processing(
                        created_citations,
                        deleted_citations,
                        purged_citations,
                        user_id,
                        **kwargs,
                    )
                except Exception as e:
                    logger.error(f"Error in memory writer task: {e}", exc_info=True)
                    await self._safe_emit(
                        "status", "⚠️ Memory processing error", **kwargs, final=True
                    )

            task = asyncio.create_task(run_memory_writer_task())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            # Overwrite the response content if memory commands were found and hiding is enabled.
            if memory_operations and not self.valves.show_assistant_mem_content:
                assistant_message["content"] = cleaned_content
                logger.info("Hiding assistant memory commands from final response.")

        except Exception as e:
            logger.error(f"Error in outlet: {str(e)}", exc_info=True)

        return body

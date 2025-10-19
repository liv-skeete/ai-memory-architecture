"""
title: Adaptive Memory Stream (MRS)
description: A memory module for Open WebUI where the `inlet` handles memory retrieval and the `outlet` handles memory creation, deletion, and pruning based on assistant commands.
author: Cody
version: 3.10.4
date: 2025-08-13
dev_note: "Cody: Use `_scratchpad.md` as needed. Review `_changelog.md` for context, then add a new entry."
"""

import aiohttp
import asyncio
import json
import logging
import os
import re
import time
import math
import numpy as np

from pydantic import BaseModel, Field
from typing import Any, Awaitable, Callable, Literal, Optional

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
            default=0,
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
        # Memory retention settings
        max_memories: int = Field(
            default=1000,
            description="Maximum number of memories to store",
            ge=1,
        )
        selection_method: Literal["Oldest", "Hybrid"] = Field(
            default="Hybrid",
            description="Method for selecting which memories to drop",
        )
        max_age: int = Field(
            default=10,
            description="Half-life in days for exponential decay of memory recency",
        )
        importance_weight: float = Field(
            default=0.5,
            description="Weight for importance when pruning (between 0.0 and 1.0)",
            ge=0.0,
            le=1.0,
        )
        pruning_threshold: int = Field(
            default=100,
            description="Minimum number of memories required to trigger active pruning",
            ge=1,
        )
        pruning_percentage: int = Field(
            default=1,
            description="Percentage of excess memories to prune. Set to 0 to disable",
            ge=0,
            le=100,
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
            ge=0,
        )
        max_retries: int = Field(
            default=2,
            description="Maximum number of retries for API calls",
            ge=0,
        )
        retry_delay: float = Field(
            default=3.0,
            description="Delay between retries (seconds)",
            ge=0.0,
        )
        # Stage 1: Memory Retrieval settings
        enable_vector_search: bool = Field(
            default=True,
            description="Enable vector search pre-filtering",
        )
        # Vector DB settings
        vector_db: str = Field(
            default="memory_vector_db",
            description="The directory to store the vector database, relative to the 'data/' path.",
        )
        embedding_model: str = Field(
            default="all-mpnet-base-v2",
            description="The name of the SentenceTransformer model to use for embeddings.",
        )
        vector_search_top_k: int = Field(
            default=50,
            description="Number of memories to retrieve from vector search for LLM re-ranking",
            ge=1,
        )
        # Deduplication settings
        enable_deduplication: bool = Field(
            default=True,
            description="Enable near-duplicate deduplication during session init",
        )
        dedupe_distance_threshold: float = Field(
            default=0.08,
            description="Distance threshold for considering memories near-duplicates",
        )
        delete_distance_threshold: float = Field(
            default=0.008,
            description="Distance threshold for vector-match deletion safety check",
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
            default=5,
            description="Number of memories to send to assisatnt for context",
            ge=1,
        )
        rerank_chunk_size: int = Field(
            default=25,
            description="The size of each chunk for LLM re-ranking",
            ge=1,
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
        self._session_init_completed_users = set()
        self._backfill_checked_users = set()

        # Initialize vector DB and embedding model
        self.vector_db_client = None
        self.embedding_model = None
        try:
            self._log_message("Initializing ChromaDB client...")
            db_path = os.path.join("data", self.valves.vector_db)
            self.vector_db_client = chromadb.PersistentClient(path=db_path)
            self._log_message(f"ChromaDB client initialized at '{db_path}'.")

            # Defer embedding model loading until it's needed to ensure valves are set.
            self.embedding_model = None
        except Exception as e:
            logger.error(f"Failed to initialize vector DB: {e}", exc_info=True)
            self.vector_db_client = None

    async def close(self) -> None:
        """Close the aiohttp session and cancel any background tasks with a shared timeout."""
        SHUTDOWN_TIMEOUT = 30.0

        try:
            async with asyncio.timeout(SHUTDOWN_TIMEOUT):
                # 1. Cancel background tasks
                if hasattr(self, "_background_tasks") and self._background_tasks:
                    tasks_to_cancel = list(self._background_tasks)
                    self._log_message(
                        f"Cancelling {len(tasks_to_cancel)} pending background tasks..."
                    )
                    for task in tasks_to_cancel:
                        task.cancel()

                    await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                    self._background_tasks.clear()
                    self._log_message("Background tasks cancelled.")

                # 2. Close the session within the same timeout budget
                if (
                    hasattr(self, "session")
                    and self.session
                    and not self.session.closed
                ):
                    self._log_message("Closing aiohttp session...")
                    await self.session.close()
                    self.session = None
                    self._log_message("Session closed.")

                # 3. Release embedding model from memory
                if hasattr(self, "embedding_model") and self.embedding_model:
                    self._log_message("Releasing embedding model from memory...")
                    del self.embedding_model
                    self.embedding_model = None
                    self._log_message("Embedding model released.")

        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(
                f"Shutdown timed out after {SHUTDOWN_TIMEOUT} seconds. Some resources may not be cleanly released."
            )

    async def __aenter__(self):
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager with resource cleanup."""
        await self.close()

    def update_valves(self, new_valves: dict[str, Any]) -> None:
        """
        Update valve settings.

        Args:
            new_valves: Dictionary of valve settings to update
        """
        # Only log configuration updates in verbose mode
        self._log_message("Updating module configuration")
        for key, value in new_valves.items():
            if hasattr(self.valves, key):
                setattr(self.valves, key, value)

                lower_key = key.lower()

                # Determine a safe display value for logs
                if isinstance(value, str):
                    if any(
                        s in lower_key for s in ("key", "token", "secret", "password")
                    ):
                        display_val = (
                            (value[:2] + "***" + value[-2:])
                            if len(value) > 4
                            else "***"
                        )
                    elif key.endswith("_prompt"):
                        display_val = value[:50] + "..." if len(value) > 50 else value
                    else:
                        display_val = value
                else:
                    display_val = value

                self._log_message(
                    f"Updating {key}", verbose_msg=f"  └─ New value: {display_val}"
                )

    @staticmethod
    def _inv_age_exponential(updated_at: int, half_life_days: float) -> float:
        now = int(time.time())
        age_days = (now - updated_at) / 86400.0
        lam = math.log(2) / half_life_days
        return math.exp(-lam * age_days)

    @staticmethod
    def _strip_score(content: str) -> str:
        """A canonical helper to remove a trailing score from a memory string."""
        if not content:
            return ""
        return Filter._score_pattern.sub("", content).strip()

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

    async def _delete_memory_by_id(self, memory_id: str, user_id: str) -> bool:
        """
        A centralized, atomic-safe method to delete a memory from both the
        primary and vector databases.
        """
        if not user_id or not memory_id:
            return False

        loop = asyncio.get_event_loop()

        try:
            # 1. Delete from Vector DB first. This is the less critical of the two.
            collection = self._get_collection(user_id)
            if collection:
                # Graceful delete: Check before deleting to avoid noisy warnings.
                if collection.get(ids=[memory_id]).get("ids"):
                    collection.delete(ids=[memory_id])
                    self._log_message(
                        None,
                        verbose_msg=f"Deleted vector {memory_id} from '{self._collection_name(user_id)}'",
                    )

            # 2. Upon success, delete from the primary database.
            await loop.run_in_executor(
                None, Memories.delete_memory_by_id_and_user_id, memory_id, user_id
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to delete memory {memory_id} for user {user_id}: {e}",
                exc_info=True,
            )
            # A failure here might mean the primary DB record still exists.
            # The backfill process will detect the missing vector and repair it.
            return False

    async def _delete_excess_memories(
        self, user_id: str, perform_percentage_prune: bool = False, **kwargs
    ) -> list[str]:
        """Remove excess memories based on the selected strategy."""
        if perform_percentage_prune:
            self._log_message(f"Starting deep pruning check for user {user_id}...")

        loop = asyncio.get_event_loop()
        mems = (
            await loop.run_in_executor(None, Memories.get_memories_by_user_id, user_id)
            or []
        )

        # Pruning Calculation
        num_mems = len(mems)
        num_to_prune_by_max = max(0, num_mems - self.valves.max_memories)
        num_to_prune_by_percent = 0

        # Percentage-based pruning is now optional and only runs when explicitly triggered (e.g., once per session).
        if (
            perform_percentage_prune
            and num_mems >= self.valves.pruning_threshold
            and self.valves.pruning_percentage > 0
        ):
            # The calculation is based on the surplus over the threshold, not the total.
            surplus = num_mems - self.valves.pruning_threshold
            num_to_prune_by_percent = int(
                surplus * (self.valves.pruning_percentage / 100.0)
            )

        num_to_prune = max(num_to_prune_by_max, num_to_prune_by_percent)

        if num_to_prune <= 0:
            if perform_percentage_prune:
                self._log_message(
                    f"Deep prune for {user_id}: No memories meet criteria (Total: {num_mems}, Threshold: {self.valves.pruning_threshold})"
                )
            else:
                self._log_message(
                    f"Shallow prune for {user_id}: No excess memories to delete."
                )
            return []

        self._log_message(
            f"Pruning {num_to_prune} memories for user {user_id} ({self.valves.selection_method} method).",
            verbose_msg=f"  └─ Total memories: {num_mems}, Pruning threshold: {self.valves.pruning_threshold}",
        )

        # Let the user know what's happening before the purge.
        await self._emit_status(f"🔥️ Purging {num_to_prune} memories...", **kwargs)

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
        mems_to_prune = mems[:num_to_prune]

        for m in mems_to_prune:
            success = await self._delete_memory_by_id(m.id, user_id)

            if success:
                # If deletion was successful, prepare citation and logging.
                importance_score = self._get_memory_importance(m)
                citation_str = self._format_citation_line(
                    symbol="🔥️",
                    content=self._strip_score(m.content).strip(),
                    score=importance_score,
                    score_label="importance",
                )
                purged_citations.append(citation_str)
                self._log_message(
                    f"Purged memory:🔥️{self._format_log_content(m.content)}",
                    verbose_msg=f"  └─ Citation: {citation_str}",
                )

        if purged_citations:
            self._log_message(
                f"Finished purging {len(purged_citations)} memories for user {user_id}"
            )

        return purged_citations

    @staticmethod
    def _truncate_log_lines(text: str, max_lines: int = 1000) -> str:
        """
        Truncate a multi-line string to a maximum number of lines.
        Used for verbose logging to prevent flooding the console.

        Args:
            text: The text to truncate.
            max_lines: The maximum number of lines to keep.

        Returns:
            Truncated text with an indicator of how many lines were omitted.
        """
        lines = text.split("\n")
        if len(lines) <= max_lines:
            return text

        truncated = lines[:max_lines]
        omitted = len(lines) - max_lines
        truncated.append(f"... [truncated, {omitted} more lines omitted]")
        return "\n".join(truncated)

    def _log_message(
        self, standard_msg: str, verbose_msg: Optional[str] = None
    ) -> None:
        """
        Logs messages, handling standard and verbose output.
        The standard message is always logged. The verbose message is logged only
        if verbose_logging is enabled.
        """
        if standard_msg:
            logger.info(standard_msg)
        if self.valves.verbose_logging and verbose_msg:
            logger.info(verbose_msg)

    def _format_log_content(self, text: str, max_len: int = 200) -> str:
        """
        Sanitizes and truncates text for clean, single-line logging.
        Replaces newlines and truncates to a specified maximum length.
        """
        if not text:
            return ""
        # Replace newlines with spaces to ensure single-line output
        single_line_text = text.replace("\n", " ").strip()
        if len(single_line_text) <= max_len:
            return single_line_text
        return single_line_text[: max_len - 3] + "..."

    @staticmethod
    def _sanitize_message_for_logging(content: Any) -> str:
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

    # Unified helpers for collection access and embeddings
    def _collection_name(self, user_id: str) -> str:
        """Return the vector collection name for a user."""
        return f"user-memory-{user_id}"

    def _get_collection(self, user_id: str):
        """Return ChromaDB collection for the given user, or None if client unavailable."""
        if not self.vector_db_client:
            return None
        return self.vector_db_client.get_or_create_collection(
            name=self._collection_name(user_id)
        )

    async def _embed(self, texts):
        """
        Thread-safe embedding helper.
        Accepts a string or list of strings and returns a list (or list of lists) of floats.
        """
        if isinstance(texts, str):
            vec = await asyncio.to_thread(self.embedding_model.encode, texts)
            return vec.tolist()
        else:
            vecs = await asyncio.to_thread(self.embedding_model.encode, texts)
            return vecs.tolist()

    # --------------------------------------------------------------------------
    # Stage 1: Memory Retrieval (Synchronous)
    # --------------------------------------------------------------------------

    async def _get_relevant_memories_llm(
        self, user_message: str, assistant_message: str, db_memories: list[Any]
    ) -> list[ProcessingMemory]:
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

        self._log_message(
            f"Starting LLM re-ranking for {len(db_memories)} candidate memories."
        )

        # 1. Split candidate memories into chunks
        chunk_size = self.valves.rerank_chunk_size
        memory_chunks = [
            db_memories[i : i + chunk_size]
            for i in range(0, len(db_memories), chunk_size)
        ]
        self._log_message(
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
                f"{idx}. {mem.content}" for idx, mem in enumerate(chunk, 1)
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

            model = (
                self.valves.openai_model
                if self.valves.api_provider == "OpenAI API"
                else self.valves.ollama_model
            )
            self._log_message(
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
        for i, (response, original_chunk) in enumerate(
            zip(api_results, [chunk for _, chunk in tasks_with_chunks])
        ):
            if isinstance(response, Exception):
                logger.error(
                    f"API call for chunk {i+1} failed: {response}", exc_info=response
                )
                continue

            if not original_chunk:
                continue

            self._log_message(
                f"Received response for re-ranking chunk {i+1}",
                verbose_msg=f"  └─ Raw response: {Filter._truncate_log_lines(response)}",
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
                                logger.warning(
                                    f"LLM returned an out-of-bounds index: {mem_info['index']}. Chunk size is {len(original_chunk)}. Discarding."
                                )

                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Invalid index or relevance score from LLM. Raw data: {mem_info}. Error: {e}"
                            )

            except json.JSONDecodeError as e:
                logger.error(
                    f"Error decoding LLM JSON response for chunk {i+1}: {e}",
                    exc_info=True,
                )
                self._log_message(
                    None,
                    verbose_msg=f"Problematic JSON content for chunk {i+1}: {prepared_response}",
                )

        # 5. Sort the final aggregated list by relevance
        all_reranked_memories.sort(key=lambda x: x.relevance, reverse=True)

        # 6. Return the top N memories as defined by llm_search_top_k
        top_k_memories = all_reranked_memories[: self.valves.llm_search_top_k]

        self._log_message(
            f"LLM re-ranking complete. Selected top {len(top_k_memories)} memories.",
            verbose_msg="\n".join(
                f"  - Top {i+1}: {self._format_log_content(mem.content)} [relevance: {mem.relevance:.2f}]"
                for i, mem in enumerate(top_k_memories)
            ),
        )

        return top_k_memories

    async def _get_candidate_memories_vector_search(
        self, user_id: str, query_text: str, top_k: int
    ) -> list[str]:
        """
        Perform a vector search to get a list of candidate memory IDs.
        """
        if not all([self.vector_db_client, self.embedding_model]):
            logger.warning(
                "Vector DB client or embedding model not initialized. Skipping vector search."
            )
            return []

        try:
            collection = self._get_collection(user_id)
            # Create a concise, single-line version for the standard log
            standard_query_log = f'Vector search for top {top_k} candidates with query: "{self._format_log_content(query_text)}"'
            # Create a detailed, multi-line version for the verbose log
            verbose_query_log = f"Full vector search query:\n---\n{Filter._truncate_log_lines(query_text)}\n---"
            self._log_message(standard_query_log, verbose_msg=verbose_query_log)

            query_embedding = await self._embed(query_text)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents"],
            )

            # ChromaDB returns a dict with 'ids', 'documents', 'metadatas', etc.
            # We only need the IDs from the first (and only) query result.
            candidate_ids = results.get("ids", [[]])[0]

            if not candidate_ids:
                self._log_message("Vector search returned no candidate IDs.")
                return []

            self._log_message(
                f"Vector search found {len(candidate_ids)} candidate IDs."
            )
            return candidate_ids

        except Exception as e:
            logger.error(f"An error occurred during vector search: {e}", exc_info=True)
            return []

    async def get_relevant_memories(
        self,
        user_message: str,
        assistant_message: str,
        user_id: str,
        db_memories: Optional[list[Any]] = None,
    ) -> list[ProcessingMemory]:
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
                self._log_message("Using vector search to identify candidate memories.")

                query_text = user_message

                candidate_ids = await self._get_candidate_memories_vector_search(
                    user_id=user_id,
                    query_text=query_text,
                    top_k=self.valves.vector_search_top_k,
                )

                if not candidate_ids:
                    self._log_message(
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

                self._log_message(
                    f"Found {len(candidate_memories)} matching memories in database from {len(candidate_ids)} candidate IDs."
                )

            # Stage 1b: LLM Re-ranking or Direct Selection
            if self.valves.enable_llm_search:
                self._log_message("Using LLM to re-rank candidate memories.")
                return await self._get_relevant_memories_llm(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    db_memories=candidate_memories,
                )
            else:
                self._log_message(
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
                self._log_message(
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
        score: Optional[float] = None,
        score_label: Optional[str] = None,
    ) -> str:
        """
        Format a single memory item for citation with a symbol.

        Args:
            symbol: The leading symbol (e.g., '🔍', '💾', '🗑️').
            content: The memory content.
            score: An optional score (relevance or importance).
            score_label: The label for the score (e.g., 'relevance').

        Returns:
            A formatted citation string for a single memory.
        """
        score_str = (
            f" [{score_label}: {score:.2f}]"
            if score is not None and score_label
            else ""
        )
        return f"{symbol} {content}{score_str}"

    def _format_memories_for_context(
        self, relevant_memories: list[ProcessingMemory]
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
                mem.content for mem in relevant_memories
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
        memory_operations: list[ProcessingMemory],
        user_id: str,
        **kwargs,
    ) -> tuple[list[str], list[str]]:
        """
        Executes memory operations (creation and deletion) and returns lists of
        strings for citation.
        """
        await self._emit_status("💾 Storing and updating memories...", **kwargs)
        if not memory_operations:
            return [], []

        created_memories_for_citation = []
        deleted_memories_for_citation = []

        # Process DELETEs first to ensure we cite the old memory's importance and
        # avoid matching freshly-created vectors; then order NEWs by importance.
        sorted_ops = sorted(
            memory_operations,
            key=lambda x: (0 if x.operation == "DELETE" else 1, -(x.importance or 0)),
        )

        for mem_op in sorted_ops:
            if mem_op.operation == "NEW" and mem_op.content:
                created_id = await self._create_memory(mem_op, user_id, **kwargs)
                if created_id:
                    citation_str = self._format_citation_line(
                        symbol="💾",
                        content=mem_op.content,
                        score=mem_op.importance,
                        score_label="importance",
                    )
                    created_memories_for_citation.append(citation_str)
                    self._log_message(
                        f"Created memory:💾{self._format_log_content(mem_op.content)}",
                        verbose_msg=f"  └─ Citation: {citation_str}",
                    )

            elif mem_op.operation == "DELETE" and mem_op.content:
                deleted_content, importance_score = await self._delete_memory(
                    mem_op, user_id, **kwargs
                )
                if deleted_content:
                    citation_str = self._format_citation_line(
                        symbol="🗑️",
                        content=self._strip_score(deleted_content),
                        score=importance_score,
                        score_label="importance",
                    )
                    deleted_memories_for_citation.append(citation_str)
                    self._log_message(
                        f"Deleted memory:🗑️{self._format_log_content(deleted_content)}",
                        verbose_msg=f"  └─ Citation: {citation_str}",
                    )

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
            if memory_to_create.importance is not None:
                full_content = f"{full_content} [{memory_to_create.importance:.2f}]"

            memory = await loop.run_in_executor(
                None,
                Memories.insert_new_memory,
                user_id,
                full_content,
            )
            if memory and hasattr(memory, "id"):
                self._log_message(
                    None, verbose_msg=f"Successfully created memory {memory.id}"
                )

                # Also add the new memory to the vector database
                if self.vector_db_client and self.embedding_model:
                    try:
                        collection = self._get_collection(user_id)

                        # New in 3.8.0: Create the embedding from the *canonical* content (score-stripped)
                        # This ensures that lookups for deletion are based on the same vector representation.
                        canonical_content = self._strip_score(full_content)
                        embedding = await self._embed(canonical_content)

                        collection.add(
                            ids=[memory.id],
                            embeddings=[embedding],
                            # IMPORTANT: Store the *original* full_content in the document for citation.
                            documents=[full_content],
                        )
                        self._log_message(
                            f"Added vector for memory {memory.id} to '{self._collection_name(user_id)}'"
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
    ) -> tuple[Optional[str], Optional[float]]:
        """
        Finds a memory via vector search and deletes it using the centralized
        _delete_memory_by_id method.
        """
        try:
            if not all([user_id, self.vector_db_client, self.embedding_model]):
                logger.warning(
                    "Cannot delete memory: User ID or vector services are missing."
                )
                return None, None

            # The incoming content is already canonical (score-stripped).
            content_to_delete = memory_to_delete.content.strip()
            if not content_to_delete:
                return None, None

            self._log_message(
                f"Searching vector DB for memory to delete: '{self._format_log_content(content_to_delete)}'"
            )

            # 1. Generate an embedding for the canonical content.
            query_embedding = await self._embed(content_to_delete)

            # 2. Query for the single most similar vector.
            collection = self._get_collection(user_id)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=[
                    "distances",
                    "documents",
                ],  # Get distance and original document
            )

            # 3. Validate the result before deleting.
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            documents = results.get("documents", [[]])[0]

            # We must have a result, and it must be a near-perfect match.
            if (
                ids
                and distances
                and distances[0] < self.valves.delete_distance_threshold
            ):
                memory_id_to_delete = ids[0]
                original_content = documents[0]
                self._log_message(
                    f"Found vector match for deletion: {self._format_log_content(original_content)}",
                    verbose_msg=f"  └─ DB ID: {memory_id_to_delete}, Distance: {distances[0]:.6f}",
                )

                # 4. Delete using the new centralized method.
                success = await self._delete_memory_by_id(memory_id_to_delete, user_id)

                if success:
                    self._log_message(
                        f"Deleted memory {memory_id_to_delete} from primary DB and vector store."
                    )
                    # For citation, get the score from the original content and return it.
                    importance_score = self._get_memory_importance(
                        type("obj", (object,), {"content": original_content})()
                    )
                    return original_content, importance_score
            else:
                # If no sufficiently close match is found, log details for diagnostics and do not delete.
                distance_log = (
                    f"distance: {distances[0]:.6f}" if distances else "no results"
                )

                # In verbose mode, show the content retrieved vs. the content requested for deletion.
                verbose_comparison_log = ""
                if documents:
                    retrieved_canon = self._strip_score(documents[0])
                    verbose_comparison_log = (
                        f"\n  - Requested Delete (canonical): '{content_to_delete}'"
                        f"\n  - Retrieved (Top 1 canonical):  '{retrieved_canon}'"
                        f"\n  - Retrieved (Top 1 original):   '{self._format_log_content(documents[0])}'"
                    )

                logger.warning(
                    f"Could not find a sufficiently close vector match for deletion ({distance_log}). No memory deleted."
                )
                self._log_message(None, verbose_msg=verbose_comparison_log)

            return None, None

        except Exception as e:
            logger.error(
                f"Error during vector-based memory deletion search: {e}", exc_info=True
            )
            return None, None

    def _parse_assistant_response_for_memories(
        self,
        response_content: str,
    ) -> tuple[list[ProcessingMemory], str]:
        """
        Parses the assistant's response to find and extract memory operations
        using the robust `❗️💾...🔚❗️` and `❗️🗑️...🔚❗️` format.

        Args:
            response_content: The full text of the assistant's response.

        Returns:
            A tuple containing:
            - A list of `ProcessingMemory` objects for all found operations.
            - The cleaned response content with memory directives removed.
        """
        # This single regex finds all memory operations and captures the symbol and content.
        # It's robust against multi-line content because of re.DOTALL.
        op_pattern = re.compile(r"❗️(💾|🗑️)(.*?)🔚❗️", re.DOTALL)
        operations_found = []
        last_end = 0
        cleaned_parts = []

        for match in op_pattern.finditer(response_content):
            # Append the text between the last match and this one
            cleaned_parts.append(response_content[last_end : match.start()])
            last_end = match.end()

            op_symbol, raw_content = match.groups()
            content = raw_content.strip()

            if not content:
                continue

            if op_symbol == "💾":
                importance = self._get_memory_importance(
                    type("obj", (object,), {"content": content})()
                )
                clean_content = self._strip_score(content)
                operations_found.append(
                    ProcessingMemory(
                        content=clean_content,
                        importance=importance,
                        operation="NEW",
                    )
                )
            elif op_symbol == "🗑️":
                clean_content = self._strip_score(content)
                operations_found.append(
                    ProcessingMemory(content=clean_content, operation="DELETE")
                )

        # Append any remaining text after the last match
        cleaned_parts.append(response_content[last_end:])
        cleaned_response = "".join(cleaned_parts).strip()

        return operations_found, cleaned_response

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

    async def _query_api(self, provider: str, messages: list[dict[str, Any]]) -> str:
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

    async def _emit_event(self, payload: dict, **kwargs) -> None:
        """A low-level wrapper to safely emit events to the WebSocket."""
        emitter = kwargs.get("__event_emitter__")
        if emitter:
            try:
                await emitter(payload)
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}", exc_info=True)

    async def _emit_status(
        self, description: str, done: bool = False, **kwargs
    ) -> None:
        """
        Handles sending all status messages.

        Args:
            description: The text to display.
            done: If True, the message will be cleared from the UI.
            **kwargs: Must contain the event emitter.
        """
        if not self.valves.show_status:
            return

        payload = {
            "type": "status",
            "data": {"description": description, "done": done},
        }
        await self._emit_event(payload, **kwargs)

    async def _send_citation(
        self,
        status_updates: list[str],
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
            "type": "citation",
            "data": {
                "document": [citation_message],
                "metadata": [{"source": source_path, "html": False}],
                "source": {"name": title},
            },
        }

        self._log_message(f"Sending '{title}' citation for user {user_id}")
        await self._emit_event(payload, **kwargs)

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
            self._log_message(f"Starting vector backfill check for user: {user_id}")
            if not self.vector_db_client or not self.embedding_model:
                logger.warning("Vector DB client/model not available for backfill.")
                return

            loop = asyncio.get_event_loop()
            all_db_mems = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, user_id
            )

            collection_name = self._collection_name(user_id)
            collection = self._get_collection(user_id)
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
                collection = self._get_collection(user_id)

            # If there are no memories, there's nothing to do.
            if not all_db_mems:
                self._log_message(
                    "No memories found in primary DB. Backfill not needed."
                )
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
                self._log_message(
                    f"Found {count} memories missing vectors. Creating now..."
                )
                await self._emit_status(f"⚙️ Backfilling {count} vector(s)...", **kwargs)

                mems_to_create = [db_mem_map[id] for id in missing_in_vector_db]
                documents = [mem.content for mem in mems_to_create]
                # Bug Fix: Ensure backfilled vectors are created from canonical, score-stripped content.
                canonical_contents = [self._strip_score(doc) for doc in documents]

                embeddings = await self._embed(canonical_contents)

                collection.add(
                    ids=[mem.id for mem in mems_to_create],
                    embeddings=embeddings,
                    documents=documents,  # Store the original document with score
                )
                self._log_message(f"Successfully created {count} vectors.")
                # As per user feedback, this message indicates the user's memories are current.
                await self._emit_status(
                    "✅ Memories are up-to-date", done=True, **kwargs
                )
            else:
                self._log_message(
                    f"Vector DB is already up-to-date for user {user_id}."
                )

        except Exception as e:
            logger.error(f"Error during vector backfill: {e}", exc_info=True)
            await self._emit_status("⚠️ Vector backfill error.", done=True, **kwargs)

    async def _deduplicate_user_memories(self, user_id: str, **kwargs) -> list[str]:
        """
        Finds and consolidates near-duplicate memories using efficient, local vector
        distance calculations. This avoids redundant network calls and simplifies the logic.
        """
        citations: list[str] = []
        if not self.vector_db_client or not self.valves.enable_deduplication:
            return citations

        loop = asyncio.get_event_loop()
        db_mems = (
            await loop.run_in_executor(None, Memories.get_memories_by_user_id, user_id)
            or []
        )
        if len(db_mems) < 2:
            return citations

        await self._emit_status("🧹 Deduplicating near-duplicate memories...", **kwargs)
        self._log_message(f"Deduplication started for user {user_id}")

        try:
            collection = self._get_collection(user_id)
            data = collection.get(include=["embeddings", "documents"])

            ids = data.get("ids", [])
            embs = data.get("embeddings", [])

            if (
                not ids
                or not hasattr(embs, "__len__")
                or len(embs) == 0
                or len(ids) < 2
            ):
                self._log_message(
                    "Deduplication skipped: Not enough memories with embeddings to compare.",
                    verbose_msg=f"  └─ IDs found: {len(ids)}, Embeddings found: {len(embs) if hasattr(embs, '__len__') else 0}",
                )
                await self._emit_status(
                    "🧹 Deduplicating near-duplicate memories...", done=True, **kwargs
                )
                return citations

            # --- Local Cosine Distance Calculation ---
            E = np.array(embs).astype(np.float32)
            # L2 normalization for cosine similarity
            E_norm = np.linalg.norm(E, axis=1, keepdims=True)
            # Add a small epsilon to avoid division by zero
            E_normalized = E / (E_norm + 1e-8)
            # Cosine distance matrix: D = 1 - (E @ E.T)
            distance_matrix = 1 - np.dot(E_normalized, E_normalized.T)
            # --- End Local Calculation ---

            # Build adjacency graph from the local distance matrix
            adjacency: dict[str, set] = {mid: set() for mid in ids}
            threshold = self.valves.dedupe_distance_threshold

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if distance_matrix[i, j] <= threshold:
                        id_i, id_j = ids[i], ids[j]
                        adjacency[id_i].add(id_j)
                        adjacency[id_j].add(id_i)

            # Find connected components (clusters of duplicates)
            visited: set = set()
            clusters: list[set] = []
            for node in ids:
                if node not in visited:
                    comp = set()
                    q = [node]
                    visited.add(node)
                    while q:
                        curr = q.pop(0)
                        comp.add(curr)
                        for neighbor in adjacency.get(curr, []):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append(neighbor)
                    if len(comp) > 1:
                        clusters.append(comp)

            if not clusters:
                await self._emit_status(
                    "🧹 Deduplicating near-duplicate memories...", done=True, **kwargs
                )
                return citations

            # --- Process Clusters ---
            self._log_message(f"Found {len(clusters)} duplicate cluster(s) to process.")
            db_map: dict[str, Any] = {str(m.id): m for m in db_mems}
            id_to_doc: dict[str, str] = dict(zip(ids, data.get("documents", [])))

            for comp in clusters:
                members = list(comp)

                # Select keeper with the highest retention score
                def retention(mid: str) -> float:
                    m = db_map.get(mid)
                    if m is None:
                        stub = type(
                            "obj",
                            (object,),
                            {
                                "updated_at": 0,
                                "importance": None,
                                "content": id_to_doc.get(mid, ""),
                            },
                        )()
                        return self._calculate_retention_score(
                            stub, self.valves.importance_weight, self.valves.max_age
                        )
                    return self._calculate_retention_score(
                        m, self.valves.importance_weight, self.valves.max_age
                    )

                keeper_id = max(members, key=retention)

                # Compute minimum cosine distance from keeper to other members in this cluster
                try:
                    keeper_idx = ids.index(keeper_id)
                    other_indices = [
                        ids.index(mid) for mid in members if mid != keeper_id
                    ]
                    min_distance = (
                        float(np.min(distance_matrix[keeper_idx, other_indices]))
                        if other_indices
                        else float("nan")
                    )
                except Exception:
                    min_distance = float("nan")

                # Average importance across the cluster
                importances = []
                for mid in members:
                    mem_obj = db_map.get(mid)
                    if not mem_obj:
                        mem_obj = type(
                            "obj", (object,), {"content": id_to_doc.get(mid, "")}
                        )()
                    importances.append(self._get_memory_importance(mem_obj))

                avg_importance = (
                    sum(importances) / len(importances) if importances else 0.5
                )

                # Get canonical content from the keeper
                keeper_obj = db_map.get(keeper_id)
                keeper_content = (
                    keeper_obj.content if keeper_obj else id_to_doc.get(keeper_id, "")
                )
                canonical_content = self._strip_score(keeper_content)

                # Delete all members, then re-create the canonical one
                for mid in members:
                    await self._delete_memory_by_id(mid, user_id)

                await self._create_memory(
                    ProcessingMemory(
                        content=canonical_content,
                        importance=avg_importance,
                        operation="NEW",
                    ),
                    user_id,
                    **kwargs,
                )

                citation_line = self._format_citation_line(
                    symbol="♻️",
                    content=canonical_content,
                    score=avg_importance,
                    score_label="importance",
                )
                citations.append(citation_line)
                self._log_message(
                    f"Deduplicated cluster of {len(members)} (min_dist: {min_distance:.4f}) -> '{self._format_log_content(canonical_content)}'"
                )

            await self._emit_status(
                "🧹 Deduplicating near-duplicate memories...", done=True, **kwargs
            )
            return citations

        except Exception as e:
            logger.error(f"Error during deduplication: {e}", exc_info=True)
            await self._emit_status("⚠️ Deduplication error.", done=True, **kwargs)
            return []

    async def _run_session_init_tasks(self, user_id: str, **kwargs) -> None:
        """
        Runs one-time session initialization tasks sequentially to prevent race conditions.
        """
        self._log_message(
            f"Running sequential session init tasks for {user_id}: Pruning then Backfill then Dedupe"
        )

        # 1. First, run the deep-pruning process and send a citation if needed.
        # This establishes the correct state of the primary DB.
        try:
            purged_citations = await self._delete_excess_memories(
                user_id, perform_percentage_prune=True, **kwargs
            )
            if purged_citations:
                await self._send_citation(
                    purged_citations, user_id, "processed", **kwargs
                )
        except Exception as e:
            logger.error(
                f"Error during sequential init (pruning step): {e}", exc_info=True
            )
            # Emit an error but continue to backfill, which can fix some states.
            await self._emit_status("⚠️ Pruning task error.", done=True, **kwargs)

        # 2. Second, run the vector backfill process.
        # This will now correctly sync the vector DB to the pruned primary DB.
        try:
            await self._backfill_user_vectors(user_id, **kwargs)
        except Exception as e:
            logger.error(
                f"Error during sequential init (backfill step): {e}", exc_info=True
            )
            await self._emit_status("⚠️ Vector backfill error.", done=True, **kwargs)

        # 3. Third, run deduplication after pruning and backfill complete.
        try:
            if self.valves.enable_deduplication:
                dedupe_citations = await self._deduplicate_user_memories(
                    user_id, **kwargs
                )
                if dedupe_citations:
                    await self._send_citation(
                        dedupe_citations, user_id, "processed", **kwargs
                    )
        except Exception as e:
            logger.error(
                f"Error during sequential init (dedupe step): {e}", exc_info=True
            )
            await self._emit_status("⚠️ Deduplication error.", done=True, **kwargs)

        self._log_message(f"Session init tasks complete for {user_id}.")

    # --------------------------------------------------------------------------
    # Main Entry Points (Inlet/Outlet)
    # --------------------------------------------------------------------------

    def _ensure_embedding_model_is_loaded(self) -> None:
        """
        A helper to lazily load the embedding model on its first use, ensuring
        that it's configured with the correct valve settings.
        """
        if self.embedding_model is None:
            try:
                self._log_message(
                    f"Loading embedding model for the first time: {self.valves.embedding_model}"
                )
                self.embedding_model = SentenceTransformer(self.valves.embedding_model)
                self._log_message("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to load embedding model '{self.valves.embedding_model}': {e}",
                    exc_info=True,
                )
                # Set to a known-bad state to prevent repeated load attempts.
                self.embedding_model = False

    async def inlet(
        self,
        body: dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
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

        # Ensure the embedding model is loaded before any operations that need it.
        self._ensure_embedding_model_is_loaded()

        user_id = __user__.get("id")
        if not user_id:
            logger.error("User ID not found in __user__ object.")
            return body

        self._log_message(f"Processing inlet for user {user_id}: Memory Retrieval")
        kwargs = {"__event_emitter__": __event_emitter__}
        await self._emit_status("💭 Retrieving relevant memories...", **kwargs)

        try:
            # --- Text Extraction from Message ---
            all_messages = body.get("messages", [])
            user_message_entries = [m for m in all_messages if m.get("role") == "user"]
            if not user_message_entries:
                return body

            last_user_message_entry = user_message_entries[-1]
            user_message_index = all_messages.index(last_user_message_entry)

            last_assistant_message_content = ""
            if user_message_index > 0:
                for i in range(user_message_index - 1, -1, -1):
                    if all_messages[i].get("role") == "assistant":
                        content = all_messages[i].get("content", "")
                        if isinstance(content, str):
                            last_assistant_message_content = content
                        break

            original_content = last_user_message_entry.get("content", "")
            log_safe_message = self._sanitize_message_for_logging(original_content)
            self._log_message(
                f'Processing user message: "{self._format_log_content(log_safe_message)}"'
            )

            if isinstance(original_content, list):
                text_content = " ".join(
                    part["text"]
                    for part in original_content
                    if isinstance(part, dict) and part.get("type") == "text"
                ).strip()
                last_user_message_entry["content"] = text_content
                last_user_message_content = text_content
            else:
                last_user_message_content = str(original_content)

            if not last_user_message_content.strip():
                self._log_message("No text content; skipping retrieval.")
                return body

            # --- Stage 1: Memory Retrieval (Synchronous) ---
            loop = asyncio.get_event_loop()
            db_memories = await loop.run_in_executor(
                None, Memories.get_memories_by_user_id, user_id
            )

            if not db_memories:
                self._log_message("No memories found. Skipping retrieval.")
                return body

            self._log_message(
                f"Found {len(db_memories)} memories. Starting retrieval..."
            )

            relevant_memories = await self.get_relevant_memories(
                user_message=last_user_message_content,
                assistant_message=last_assistant_message_content,
                user_id=user_id,
                db_memories=db_memories,
            )

            if relevant_memories:
                formatted, citation = self._format_memories_for_context(
                    relevant_memories
                )
                self._update_message_context(body, formatted)
                count = len(relevant_memories)
                noun = "memory" if count == 1 else "memories"
                self._log_message(f"Added {count} relevant {noun} to context.")
                await self._send_citation(
                    citation.split("\n")[1:], user_id, "read", **kwargs
                )
            else:
                self._log_message("No relevant memories found.")

            # --- One-Time Session Initialization Tasks ---
            if user_id not in self._session_init_completed_users:
                self._session_init_completed_users.add(user_id)
                self._log_message(
                    f"First turn for user {user_id}. Kicking off session maintenance."
                )
                init_task = asyncio.create_task(
                    self._run_session_init_tasks(user_id, **kwargs)
                )
                self._background_tasks.add(init_task)
                init_task.add_done_callback(self._background_tasks.discard)

        except Exception as e:
            logger.error(f"Error in inlet: {str(e)}", exc_info=True)
            await self._emit_status("⚠️ Memory retrieval error.", done=True, **kwargs)

        finally:
            # This is crucial: it guarantees the 'Retrieving' message is always cleared.
            await self._emit_status(
                "💭 Retrieving relevant memories...", done=True, **kwargs
            )
            self._log_message("Inlet processing complete.")

        return body

    async def _memory_writer_task(self, memory_operations, user_id, **kwargs) -> None:
        """
        A centralized background task for all memory writing operations. It ensures
        that a final, conclusive status message is always emitted.
        """
        created_citations, deleted_citations, purged_citations = [], [], []
        try:
            if memory_operations:
                op_count = len(memory_operations)
                noun = "operation" if op_count == 1 else "operations"
                self._log_message(f"Found {op_count} assistant-derived memory {noun}.")
                (
                    created_citations,
                    deleted_citations,
                ) = await self._execute_memory_operations(
                    memory_operations, user_id, **kwargs
                )

            # Universal shallow pruning (hard limit only) runs every time as a safety net.
            purged_citations = await self._delete_excess_memories(
                user_id, perform_percentage_prune=False, **kwargs
            )
        except Exception as e:
            logger.error(f"Error in memory writer task: {e}", exc_info=True)
            await self._emit_status("⚠️ Memory processing error.", done=True, **kwargs)
        finally:
            # This block guarantees a final status is always sent and then cleared.
            all_citations = purged_citations + created_citations + deleted_citations

            # Determine which final message to show.
            if memory_operations or all_citations:
                # Send citation only if there's content for it.
                if all_citations:
                    final_citations = (
                        created_citations + deleted_citations + purged_citations
                    )
                    await self._send_citation(
                        final_citations, user_id, "processed", **kwargs
                    )
                final_message = "🏁 Memory processing complete"
            else:
                self._log_message(f"No new memory operations for user {user_id}.")
                final_message = "🚫 No memory operations required"

            # Centralized final status handling: show the message, wait, then clear it.
            # This is a simple, direct, and reliable way to manage the final UI state.
            await self._emit_status(final_message, **kwargs)
            await asyncio.sleep(4.0)
            await self._emit_status("", done=True, **kwargs)

    async def outlet(
        self,
        body: dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
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
            self._log_message(f"Processing outlet for user {user_id}: Memory Storage")
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

            # Spawn a background task to handle all memory writing operations.
            task = asyncio.create_task(
                self._memory_writer_task(memory_operations, user_id, **kwargs)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

            # Overwrite the response content if memory commands were found and hiding is enabled.
            if memory_operations and not self.valves.show_assistant_mem_content:
                assistant_message["content"] = cleaned_content
                self._log_message(
                    "Hiding assistant memory commands from final response."
                )

        except Exception as e:
            logger.error(f"Error in outlet: {str(e)}", exc_info=True)

        return body

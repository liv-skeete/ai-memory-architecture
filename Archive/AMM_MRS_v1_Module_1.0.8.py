"""
title: AMM_Memory_Recall_Storage
description: Memory Recall & Storage Module for Open WebUI - Combines synchronous memory retrieval with asynchronous memory storage
author: Cody
version: 1.0.8
date: 2025-05-31
changes:
- v1.0.8 (2025-05-31):
  - Fixed workflow to ensure identification stage runs even when relevance stage returns no memories
- v1.0.7 (2025-05-31):
  - Passed stage 1 relevance output to stage 2 identification
  - Added relevant_memories parameter to identification methods
  - Updated identification prompt to include existing memories context
  - Added helper method to format relevant memories
  - Modified memory identification to consider existing relevant memories
- v1.0.6 (2025-05-28):
  - Fixed logging system to ensure visibility of essential operational events
  - Removed repetitive module-level logging that was flooding logs
  - Made critical operational logs always visible regardless of verbose_logging setting
  - Fixed logical error in memory processing flow that suppressed important logs
  - Silently handled cleanup warnings to prevent log flooding
  - Improved logging consistency across all memory operations
- v1.0.4 (2025-05-20):
  - Simplified memory formatting logic to fix double dash issue
  - Standardized citation formatting between 'memories read' and 'memories saved'
  - Removed redundant string manipulations and complex parsing logic
  - Created a single consistent memory formatting function
- v1.0.2 (2025-05-20):
  - Simplified memory formatting logic in _format_memories_for_context method
  - Fixed issue with unreliable display of memory tags in 'memories read' citations
  - Removed unnecessary complex parsing for bulleted compound memories
- v1.0.1 (2025-05-20):
  - Improved status emission consistency between Stage 1 and Stage 2
  - Moved "🧠 Saving important memories..." emission from _process_memory_storage to inlet method
  - Standardized status emission pattern across both processing stages
- v1.0.0 (2025-05-18):
  - Initial release combining MRE v5 and MIS v11 modules
  - Implemented two-stage processing:
    - Stage 1 (Synchronous): Memory retrieval and context enhancement
    - Stage 2 (Asynchronous): Memory identification and storage
  - Maintained core functionality from both source modules
  - Combined valve configurations while preserving module-specific parameters
  - Ensured proper error handling and logging throughout
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

import aiohttp
from open_webui.models.memories import Memories
from open_webui.models.users import Users
from pydantic import BaseModel, Field

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


class Filter:
    """Memory Recall & Storage module combining synchronous memory retrieval with asynchronous memory storage."""

    class Valves(BaseModel):
        # Enable/Disable Function
        enabled: bool = Field(
            default=True,
            description="Enable memory recall & storage",
        )
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
            default="gpt-4o-mini",
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
        """Close the aiohttp session and cancel any background tasks."""
        # Cancel any running background tasks
        if hasattr(self, "_background_tasks"):
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self._background_tasks.clear()

        # Close the session
        if hasattr(self, "session") and self.session and not self.session.closed:
            await self.session.close()

    def __del__(self):
        """Ensure session is closed when Filter instance is garbage collected."""
        if hasattr(self, "session") and self.session and not self.session.closed:
            # Silently close the session without logging
            try:
                asyncio.create_task(self.session.close())
            except RuntimeError:
                # If no event loop is running, we can't close gracefully
                pass

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
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the current context using LLM-based scoring.

        Args:
            current_message: The current user message
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing relevant memories with their scores
        """
        # Only log in verbose mode or when processing a significant number of memories
        if self.valves.verbose_logging or len(db_memories) > 10:
            logger.info("Getting relevant memories using LLM-based scoring")

        # Extract memory contents
        memory_contents = [
            mem.content
            for mem in db_memories
            if hasattr(mem, "content") and mem.content
        ]

        # If no valid memory contents, return empty list
        if not memory_contents:
            logger.info("No valid memory contents found in database")
            return []

        # Format memories for the prompt using helper
        formatted_memories = self.format_bulleted_list(memory_contents)

        # Only log in verbose mode or when processing a significant number of memories
        if self.valves.verbose_logging or len(memory_contents) > 10:
            logger.info(f"Processing {len(memory_contents)} memories from database")

        # Check if prompt is empty and fail fast
        if not self.valves.relevance_prompt:
            logger.error("Relevance prompt is empty - cannot process memories")
            raise ValueError("Relevance prompt is empty - module cannot function")

        # Format the system prompt using the valve-stored prompt
        system_prompt = self.valves.relevance_prompt.format(
            current_message=self.format_bulleted_list([current_message]),
            memories=formatted_memories,
        )

        if self.valves.verbose_logging:
            truncated_prompt = self._truncate_log_lines(system_prompt)
            # Formatted with bulleted list helper
            logger.info(f"System prompt (truncated):\n{truncated_prompt}")

        # Query the appropriate API based on the provider setting
        if self.valves.api_provider == "OpenAI API":
            if self.valves.verbose_logging:
                logger.info(f"Querying OpenAI API with model: {self.valves.openai_model}")
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            if self.valves.verbose_logging:
                logger.info(f"Querying Ollama API with model: {self.valves.ollama_model}")
            response = await self.query_ollama_api(system_prompt, current_message)

        # Log the raw response only in verbose mode
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info(f"Raw API response: {truncated_response}")

        # Prepare and parse the response
        prepared_response = self._prepare_json_response(response)
        try:
            memory_data = json.loads(prepared_response)

            # Basic validation and processing
            valid_memories = []
            if isinstance(memory_data, list):
                for mem in memory_data:
                    if isinstance(mem, dict) and "content" in mem and "score" in mem:
                        try:
                            score = float(mem["score"])
                            valid_memories.append(
                                {"content": str(mem["content"]), "score": score}
                            )
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid memory format: {mem}")
                            pass

            # Sort by score in descending order
            valid_memories.sort(key=lambda x: x["score"], reverse=True)

            # Count total memories before threshold filtering
            total_memories = len(valid_memories)

            # Filter memories based on the relevance threshold
            threshold_filtered_memories = [
                mem
                for mem in valid_memories
                if mem["score"] >= self.valves.relevance_threshold
            ]

            # Essential log summarizing memory filtering results
            filtered_count = total_memories - len(threshold_filtered_memories)
            if filtered_count > 0:
                logger.info(f"Found {total_memories} memories, filtered {filtered_count} below threshold {self.valves.relevance_threshold:.2f}")
            else:
                logger.info(f"Found {total_memories} relevant memories")

            # Log details of memories only in verbose mode
            if self.valves.verbose_logging and threshold_filtered_memories:
                for i, mem in enumerate(threshold_filtered_memories):
                    logger.info(f"Memory {i+1}: {mem['content']} (score: {mem['score']:.2f})")

                # Log filtered out memories only in verbose mode
                if filtered_count > 0:
                    logger.info(f"Memories filtered out by threshold {self.valves.relevance_threshold:.2f}:")
                    for i, mem in enumerate([m for m in valid_memories if m["score"] < self.valves.relevance_threshold]):
                        logger.info(f"Filtered {i+1}: {mem['content']} (score: {mem['score']:.2f})")

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
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the current context using LLM-based scoring.

        Args:
            current_message: The current user message
            user_id: The user ID
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing relevant memories with their scores
        """
        if not self.valves.enabled or not db_memories:
            return []

        try:
            # Use LLM for memory retrieval
            return await self._get_relevant_memories_llm(current_message, db_memories)
        except Exception as e:
            logger.error(f"Error in get_relevant_memories: {str(e)}")
            return []

    def _format_memory_item(
        self, content: str, score: float, score_label: str = "score"
    ) -> str:
        """
        Format a single memory item with consistent formatting.

        Args:
            content: The memory content
            score: The relevance or importance score
            score_label: Label for the score (e.g., "score" or "importance")

        Returns:
            Formatted memory string (without additional bullets)
        """
        # Return raw content without adding bullet formatting
        return f"☑ {content} ({score:.2f})"

    def _format_memories_for_context(
        self, relevant_memories: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        """
        Format relevant memories for inclusion in the context.

        Args:
            relevant_memories: List of dictionaries containing relevant memories with their scores

        Returns:
            Tuple containing:
            - Formatted string of memories for inclusion in the context (without scores)
            - Formatted string of memories with scores for logging
        """
        if not relevant_memories:
            return "", ""

        # Format memories using helper functions
        context_memories = "User Information (sorted by relevance):"
        citation_memories = "Memories Read (sorted by relevance):"
        
        if relevant_memories:
            context_memories += "\n" + self.format_bulleted_list(
                [mem["content"] for mem in relevant_memories]
            )
            citation_memories += "\n" + self.format_bulleted_list(
                [self._format_memory_item(mem["content"], mem["score"]) for mem in relevant_memories]
            )
            
        return context_memories, citation_memories

    def _format_relevant_memories(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format relevant memories for inclusion in the identification prompt.
        
        Args:
            memories: List of relevant memories from Stage 1 (sorted by relevance)
            
        Returns:
            Formatted string of memories in chronological order (newest first)
        """
        # Reverse order to show newest memories first
        return self.format_bulleted_list([mem["content"] for mem in reversed(memories)])

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
            # Insert the memories as a system message at the beginning of the conversation
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
        relevant_memories: List[Dict[str, Any]]  # NEW PARAMETER
    ) -> List[Dict[str, Any]]:
        """
        Identify potential memories from the current message and recent context,
        considering existing relevant memories to avoid duplication.

        Args:
            current_message: The current user message
            recent_messages: List of recent messages (user and assistant)
            relevant_memories: List of relevant memories from Stage 1

        Returns:
            List of potential memories with content and importance scores
        """
        # Always log essential operations
        logger.info("Identifying potential memories from message")

        # Check if prompt is empty and fail fast
        if not self.valves.identification_prompt:
            logger.error("Identification prompt is empty - cannot process memories")
            raise ValueError("Identification prompt is empty - module cannot function")

        # Format the system prompt with current_message AND relevant_memories
        formatted_relevant = self._format_relevant_memories(relevant_memories)
        
        # Log the relevant memories without additional formatting
        if self.valves.verbose_logging:
            # Log raw memory contents to avoid double formatting
            logger.info(f"Relevant memories for identification: {[mem['content'] for mem in relevant_memories]}")
        system_prompt = self.valves.identification_prompt.format(
            current_message=current_message,  # Pass raw string
            relevant_memories=formatted_relevant
        )

        # Log system prompt only in verbose mode
        if self.valves.verbose_logging:
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(f"System prompt for Identification (truncated):\n{truncated_prompt}")

        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, current_message)

        # Log raw response only in verbose mode
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info(f"Identification: Raw API response: {truncated_response}")

        # Parse the response
        potential_memories = self._parse_json_response(response)

        # Log a summary of potential memories
        if potential_memories:
            above_threshold = sum(
                1
                for mem in potential_memories
                if mem.get("importance", 0) >= self.valves.memory_importance_threshold
            )
            below_threshold = len(potential_memories) - above_threshold

            # Essential logging (always shown)
            # Only log memory details for important memories
            for mem in potential_memories:
                importance = mem.get("importance", 0)
                if importance >= self.valves.memory_importance_threshold:
                    logger.info(f"Memory content: {mem.get('content', '')} (score: {importance:.2f})")
            
            # Log full breakdown only in verbose mode
            if self.valves.verbose_logging:
                logger.info(f"Found {len(potential_memories)} potential memories ({above_threshold} above threshold, {below_threshold} below)")
        else:
            # Essential logging (always shown)
            logger.info("No potential memories identified")

        # Return all potential memories, regardless of threshold
        return potential_memories

    async def _identify_and_prepare_memories(
        self, current_message: str
    ) -> List[Dict[str, Any]]:  # Returns list of NEW memory operations
        """
        Identifies potential memories and prepares NEW memory operations.

        Args:
            current_message: The current user message

        Returns:
            List of NEW memory operations for memories meeting the importance threshold.
            Returns an empty list if no important memories are identified.
        """
        # Initialize memory_operations to prevent UnboundLocalError
        memory_operations = []

        # Identify potential memories (passing required parameters)
        # Using empty lists for recent_messages and relevant_memories as placeholders
        potential_memories = await self._identify_potential_memories(
            current_message,
            [],  # recent_messages (empty list placeholder)
            []   # relevant_memories (empty list placeholder)
        )

        # Filter potential memories by importance threshold
        important_memories = [
            mem
            for mem in potential_memories
            if mem.get("importance", 0) >= self.valves.memory_importance_threshold
        ]

        if not important_memories:
            if self.valves.verbose_logging:
                logger.info(
                    "No potential memories met the importance threshold (%.2f). No operations needed.",
                    self.valves.memory_importance_threshold,
                )
        else:
            # Always log important memories identified
            logger.info(f"Identified {len(important_memories)} important potential memories meeting threshold ({self.valves.memory_importance_threshold:.2f})")

            # Create NEW operations for the important memories
            memory_operations = [
                {
                    "operation": "NEW",
                    "content": mem["content"],
                    "importance": mem["importance"],
                }
                for mem in important_memories
            ]

            # Always log the final set of operations
            op_summary = ", ".join(f"NEW({op.get('importance', 0):.2f})" for op in memory_operations)
            logger.info(f"Prepared NEW memory operations: {op_summary}")

        return memory_operations

    async def _execute_memory_creation(
        self,
        memory_operations: List[Dict[str, Any]],
        __user__: dict,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> None:
        """
        Executes the NEW memory creation operations.

        Args:
            memory_operations: List of NEW operations from _identify_and_prepare_memories
            __user__: The user dictionary object
            __event_emitter__: Optional event emitter for sending status updates
        """
        # Always log essential operations
        logger.info(f"Executing {len(memory_operations)} NEW memory creation operations")
        if not memory_operations:
            return

        # Store status messages with importance for sorting (only NEW operations)
        status_updates_with_importance = []

        for op in memory_operations:
            operation_type = op.get("operation")

            # Only NEW operations are expected now
            if operation_type == "NEW":
                content = op.get("content")
                importance = op.get("importance", 0)  # Get importance for logging
                if content:
                    # Log with summary only in verbose mode
                    if self.valves.verbose_logging:
                        truncated_content = content[:100] + "..." if len(content) > 100 else content
                        # Always log memory creation (not just in verbose mode)
                        truncated_content = content[:100] + "..." if len(content) > 100 else content
                        logger.info(f"Creating memory for user {__user__.get('id', 'UNKNOWN')} (Importance: {importance:.2f}): {truncated_content}")
                    memory_id = await self._create_memory(content, __user__, importance)
                    if memory_id:
                        # Use the same formatting method as for memory retrieval
                        status = self._format_memory_item(
                            content, importance, "importance"
                        )
                        # Always log successful memory creation
                        logger.info(status)
                        # Store status with importance for sorting
                        status_updates_with_importance.append((status, importance))
                    else:
                        status = "- ⚠️ Failed to create new memory"
                        logger.warning(status)
                        # Failed operations get importance 0 for sorting
                        status_updates_with_importance.append((status, 0))
                else:
                    logger.warning("Skipping NEW operation: Content is missing")
                    # Skipped operations get importance 0 for sorting
                    status_updates_with_importance.append(
                        ("- ⚠️ Skipped creating memory: missing content", 0)
                    )
            else:
                # This case should not happen after refactoring
                logger.warning(f"Skipping unexpected operation type '{operation_type}': {op}")
                # Unknown operations get importance 0 for sorting
                status_updates_with_importance.append(
                    (f"- ⚠️ Skipped unknown operation: {operation_type}", 0)
                )

        # Sort status updates by importance (highest first)
        sorted_status_updates = [
            status
            for status, _ in sorted(
                status_updates_with_importance, key=lambda x: x[1], reverse=True
            )
        ]

        # --- Send Citation ---
        if self.valves.show_status and __event_emitter__ and sorted_status_updates:
            # Send citation only if there were operations attempted
            await self._send_citation(
                __event_emitter__, sorted_status_updates, __user__.get("id"), "saved"
            )

    async def _create_memory(
        self, content: str, __user__: dict, importance: float = 0.0
    ) -> Optional[str]:
        """
        Create a new memory in the database.

        Args:
            content: The content of the memory
            __user__: The user dictionary object
            importance: The importance score to append to the memory

        Returns:
            The ID of the created memory, or None if creation failed
        """
        try:
            # Append importance score to content in the desired format
            formatted_content = f"{content} ({importance:.1f})"

            memory = Memories.insert_new_memory(
                user_id=__user__.get("id"), content=formatted_content
            )
            if memory and hasattr(memory, "id"):
                if self.valves.verbose_logging:
                    logger.info("Successfully created memory %s", memory.id)
                return memory.id
            else:
                logger.warning("Memory creation returned None or object without ID")
                return None
        except Exception as e:
            logger.error(f"Error creating memory: {e}", exc_info=True)
            return None

    async def _process_memory_storage(
        self,
        current_message_content: str,
        __user__: dict,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
        relevant_memories: List[Dict[str, Any]] = None,
    ) -> None:
        """
        Process memory storage asynchronously.

        Args:
            current_message_content: The current user message content
            __user__: The user dictionary object
            __event_emitter__: Optional event emitter for sending status updates
            relevant_memories: Optional list of relevant memories from Stage 1
        """
        # Initialize relevant_memories if None
        if relevant_memories is None:
            relevant_memories = []

        try:
            # Always log essential operations
            logger.info("Processing memory storage")
            
            # Log the number of relevant memories being passed from Stage 1
            if relevant_memories:
                logger.info(f"Received {len(relevant_memories)} relevant memories from Stage 1")
            else:
                logger.info("No relevant memories received from Stage 1")

            # First: Identify potential memories
            # Pass all required arguments: current_message, recent_messages, relevant_memories
            potential_memories = await self._identify_potential_memories(
                current_message_content,
                [],  # Placeholder for recent_messages (to be implemented)
                relevant_memories  # Pass relevant memories from Stage 1
            )
            
            # Filter potential memories by importance threshold
            important_memories = [
                mem
                for mem in potential_memories
                if mem.get("importance", 0) >= self.valves.memory_importance_threshold
            ]
            
            # Memory count logging removed to avoid duplication (already handled in _identify_potential_memories)
            # Create NEW operations for the important memories
            memory_operations = [
                {
                    "operation": "NEW",
                    "content": mem["content"],
                    "importance": mem["importance"],
                }
                for mem in important_memories
            ]
            
            # Log operations
            if memory_operations:
                op_summary = ", ".join(f"NEW({op.get('importance', 0):.2f})" for op in memory_operations)
                logger.info(f"Prepared NEW memory operations: {op_summary}")

            # Then: Execute the creation operations
            if memory_operations:
                await self._execute_memory_creation(
                    memory_operations, __user__, __event_emitter__
                )
            else:
                logger.info("No memory operations to execute")

        except Exception as e:
            logger.error(
                f"Error during asynchronous memory storage: {str(e)}", exc_info=True
            )

        finally:
            # Always log completion
            logger.info("Memory storage processing completed")
            if self.valves.show_status and __event_emitter__:
                # Determine appropriate message based on whether memory operations were found
                if memory_operations:
                    description = f"☑ Saved {len(memory_operations)} important memories (threshold: {self.valves.memory_importance_threshold:.1f})"
                else:
                    description = "☒ No important memories identified for storage"

                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "done": True,
                        },
                    },
                )

                # Create a background task to clear the status after delay
                task = asyncio.create_task(
                    self._delayed_clear_status(__event_emitter__, 3.0)
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
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
                # Remove potential leading/trailing non-JSON characters if necessary
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

            # Ensure the extracted string is not empty
            if not json_str:
                logger.warning("Could not extract valid JSON content from response.")
                return []

            # Parse the JSON string
            parsed_data = json.loads(json_str)

            # Standardize the output format to always be a list of dictionaries
            if isinstance(parsed_data, dict):
                # If the API returns a single dictionary, check if it contains a list
                # under a common key like 'memories' or 'operations'
                for key in ["memories", "operations", "memory_operations"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        logger.debug(f"Found list under key '{key}'")
                        return parsed_data[key]
                # If it's a single dictionary without a list, wrap it in a list
                logger.debug("API returned a single dictionary, wrapping in list")
                return [parsed_data]
            elif isinstance(parsed_data, list):
                # Ensure all items in the list are dictionaries
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
        except Exception as e:
            logger.error(f"Error in event emitter: {e}")

    async def _delayed_clear_status(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        delay_seconds: float = 5.0,
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
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": "",
                        "done": True,
                    },
                },
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
            citation_type: Type of citation ("saved" or "read")
        """
        if not status_updates or not emitter:
            return

        # Determine citation title and source based on type
        if citation_type == "saved":
            title = "Memories Saved"
            source_path = "module://mrs/memories/saved"
            header = "Memories Saved (sorted by importance):"
        else:
            title = "Memories Read"
            source_path = "module://mrs/memories/read"
            header = "Memories Read (sorted by relevance):"

        # Format the citation message - dashes are added in the status formatting
        citation_message = f"{header}\n" + "\n".join(status_updates)

        if self.valves.verbose_logging:
            logger.info(f"Sending {citation_type} citation for user {user_id or 'UNKNOWN'}")

        await self._safe_emit(
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
        if (
            not self.valves.enabled
            or not body
            or not isinstance(body, dict)
            or not __user__
        ):
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
                truncated_message = formatted_message[:100] + "..." if len(formatted_message) > 100 else formatted_message
                # Formatted with bulleted list helper
                logger.info(f"Processing user message: {truncated_message}")
            else:
                logger.info("Processing user message: [empty]")

            # -----------------------------------------------------------------
            # Stage 1: Memory Retrieval (Synchronous)
            # -----------------------------------------------------------------

            # Get user object and memories
            user = Users.get_user_by_id(__user__["id"])
            if not user:
                logger.info("User not found in database")
                return body

            db_memories = Memories.get_memories_by_user_id(__user__["id"])
            if not db_memories:
                logger.info(f"No memories found for user {__user__['id']} - no memories to process")

            # Always log memory count
            logger.info(f"Found {len(db_memories)} memories for user {__user__['id']}")

            # Emit status update if enabled
            if self.valves.show_status and __event_emitter__:
                if self.valves.verbose_logging:
                    logger.info("Emitting status: Retrieving relevant memories...")
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "💭 Retrieving relevant memories...",
                            "done": False,
                        },
                    },
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
                logger.info(f"Formatted {len(relevant_memories)} relevant memories for context")

                # Update message context
                self._update_message_context(body, formatted_memories)

                # Log what's being sent to the assistant
                if self.valves.verbose_logging:
                    # In verbose mode, show full memory content with scores
                    truncated_memories = self._truncate_log_lines(formatted_memories_with_scores)
                    logger.info(f"Sending to assistant: {truncated_memories}")
                else:
                    # Always show count of memories
                    logger.info(f"Sending to assistant: {len(relevant_memories)} relevant memories with content")

                # Send citation with memories sent to assistant
                if __event_emitter__:
                    logger.info("Sending citation with memories")

                    # Extract the memory items without the header
                    memory_items = formatted_memories_with_scores.split("\n")[1:]

                    # Send the citation using the shared method
                    await self._send_citation(
                        __event_emitter__, memory_items, __user__.get("id"), "read"
                    )

                # Emit completion status if enabled
                if self.valves.show_status and __event_emitter__:
                    logger.info(f"Emitting status: Retrieved {len(relevant_memories)} relevant memories")
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": f"☑ Retrieved {len(relevant_memories)} relevant memories (threshold: {self.valves.relevance_threshold:.1f})",
                                "done": True,
                            },
                        },
                    )

                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(__event_emitter__, 3.0)
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
            else:
                # Log when no relevant memories are found
                logger.info("No relevant memories found for the current message")

                # Emit completion status if enabled
                if self.valves.show_status and __event_emitter__:
                    logger.info("Emitting status: No relevant memories found")
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "☒ No relevant memories found",
                                "done": True,
                            },
                        },
                    )

                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(__event_emitter__, 3.0)
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
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
                context_messages_contents = [m.get("content", "") for m in context_messages]
                current_message_content = self.format_bulleted_list(context_messages_contents)
                # Always log context selection
                logger.info(f"Using last {len(context_messages)} messages for memory storage context")
            else:
                # Format single message as bulleted list
                current_message_content = self.format_bulleted_list([last_user_message])
                # Always log context selection
                logger.info("Using the last message for memory storage context")

            # Emit status update if enabled
            if self.valves.show_status and __event_emitter__:
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "🧠 Saving important memories...",
                            "done": False,
                        },
                    },
                )

            # Process memory storage asynchronously if enabled
            if self.valves.async_memory_storage:
                # Always log essential operations
                logger.info("Starting memory storage task (asynchronous)")
                # Create a background task for memory storage
                task = asyncio.create_task(
                    self._process_memory_storage(
                        current_message_content, __user__, __event_emitter__, relevant_memories
                    )
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            else:
                # Process memory storage synchronously
                # Always log essential operations
                logger.info("Processing memory storage (synchronous)")
                await self._process_memory_storage(
                    current_message_content, __user__, __event_emitter__, relevant_memories
                )

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
        """Pass-through method for outgoing messages."""
        return body

"""
title: AMM_Memory_Recall_Storage
description: Memory Recall & Storage Module for Open WebUI - Combines synchronous memory retrieval with asynchronous memory storage
author: Cody
version: 1.0.5
date: 2025-05-22
changes:
- v1.0.5 (2025-05-22):
  - Made memory storage fully non-blocking by removing task tracking
  - Removed unnecessary background task tracking mechanism
  - Added clarifying comments for short-lived tasks
  - Simplified close method by removing task tracking code
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
    # Explicitly set handler level to match logger
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


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
        logger.info("Initializing Memory Recall & Storage module")

        # Initialize with empty prompts - must be set via update_valves
        try:
            self.valves = self.Valves(
                relevance_prompt="",  # Empty string to start - must be set via update_valves
                identification_prompt="",  # Empty string to start - must be set via update_valves
            )
            logger.warning(
                "Prompts are empty - module will not function until prompts are set"
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

        logger.info(
            "MRS module initialized with API provider: %s", self.valves.api_provider
        )

    async def close(self) -> None:
        """Close the aiohttp session."""
        logger.info("Closing MRS module session")

        # Close the session
        await self.session.close()

    def update_valves(self, new_valves: Dict[str, Any]) -> None:
        """
        Update valve settings.

        Args:
            new_valves: Dictionary of valve settings to update
        """
        logger.info("Updating valves")

        for key, value in new_valves.items():
            if hasattr(self.valves, key):
                # For prompt fields, log a truncated version
                if key.endswith("_prompt") and isinstance(value, str):
                    preview = value[:50] + "..." if len(value) > 50 else value
                    logger.info(f"Updating {key} with: {preview}")
                    setattr(self.valves, key, value)
                else:
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
        if self.valves.verbose_logging:
            logger.info("Getting relevant memories using LLM-based scoring")

        # Extract memory contents
        memory_contents = [
            mem.content
            for mem in db_memories
            if hasattr(mem, "content") and mem.content
        ]

        # If no valid memory contents, return empty list
        if not memory_contents:
            if self.valves.verbose_logging:
                logger.info("No valid memory contents found in database")
            return []

        # Format memories for the prompt
        formatted_memories = "\n".join([f"- {mem}" for mem in memory_contents])

        if self.valves.verbose_logging:
            logger.info(f"Processing {len(memory_contents)} memories from database")

        # Check if prompt is empty and fail fast
        if not self.valves.relevance_prompt:
            logger.error("Relevance prompt is empty - cannot process memories")
            raise ValueError("Relevance prompt is empty - module cannot function")

        # Format the system prompt using the valve-stored prompt
        system_prompt = self.valves.relevance_prompt.format(
            current_message=current_message,
            memories=formatted_memories,
        )

        if self.valves.verbose_logging:
            # Log a truncated version of the system prompt
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(f"System prompt (truncated):\n{truncated_prompt}")

        # Query the appropriate API based on the provider setting
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

        # Log the raw response for debugging
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
                            if self.valves.verbose_logging:
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

            # Log the filtering results - ALWAYS LOG THIS (not conditional)
            filtered_count = total_memories - len(threshold_filtered_memories)
            if filtered_count > 0:
                logger.info(
                    f"Found {total_memories} memories, filtered {filtered_count} below threshold {self.valves.relevance_threshold:.2f}"
                )
            else:
                logger.info(f"Found {total_memories} relevant memories")

            # Log details of memories in debug mode
            if self.valves.verbose_logging and threshold_filtered_memories:
                for i, mem in enumerate(threshold_filtered_memories):
                    logger.info(
                        f"Memory {i+1}: {mem['content']} (score: {mem['score']:.2f})"
                    )

                # Log filtered out memories in debug mode
                if filtered_count > 0:
                    logger.info(
                        f"Memories filtered out by threshold {self.valves.relevance_threshold:.2f}:"
                    )
                    for i, mem in enumerate(
                        [
                            m
                            for m in valid_memories
                            if m["score"] < self.valves.relevance_threshold
                        ]
                    ):
                        logger.info(
                            f"Filtered {i+1}: {mem['content']} (score: {mem['score']:.2f})"
                        )

            return threshold_filtered_memories

        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            if self.valves.verbose_logging:
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

    def _format_memory_item(self, content: str, score: float, score_label: str = "score") -> str:
        """
        Format a single memory item with consistent formatting.
        
        Args:
            content: The memory content
            score: The relevance or importance score
            score_label: Label for the score (e.g., "score" or "importance")
            
        Returns:
            Formatted memory string with consistent formatting
        """
        # Simply format the memory item without modifying the content
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

        # Create headers
        formatted_memories = ["User Information (sorted by relevance):"]
        formatted_memories_with_scores = ["Memories Read (sorted by relevance):"]

        for memory in relevant_memories:
            content = memory["content"]
            score = memory["score"]
            
            # Format the memory item consistently
            formatted_item = self._format_memory_item(content, score)
            
            # For context, we don't include scores
            simple_item = formatted_item.split(" (score:", 1)[0]
            
            formatted_memories.append(simple_item)
            formatted_memories_with_scores.append(formatted_item)

        return "\n".join(formatted_memories), "\n".join(formatted_memories_with_scores)

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
        self, current_message: str
    ) -> List[Dict[str, Any]]:
        """
        Identify potential memories from the current message based on importance.

        Args:
            current_message: The current user message

        Returns:
            List of potential memories with content and importance scores
        """
        # Essential logging (always shown)
        logger.info("Identifying potential memories from message")

        # Check if prompt is empty and fail fast
        if not self.valves.identification_prompt:
            logger.error("Identification prompt is empty - cannot process memories")
            raise ValueError("Identification prompt is empty - module cannot function")

        # Format the system prompt using the valve-stored prompt
        system_prompt = self.valves.identification_prompt.format(
            current_message=current_message
        )

        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            # Log a truncated version of the system prompt
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(
                "System prompt for Identification (truncated):\n%s", truncated_prompt
            )

        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, current_message)

        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info("Identification: Raw API response: %s", truncated_response)

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
            logger.info(
                "Found %d potential memories (%d above threshold, %d below)",
                len(potential_memories),
                above_threshold,
                below_threshold,
            )

            # Verbose logging (only when verbose logging is enabled)
            if self.valves.verbose_logging:
                # Log details of memories that meet the threshold
                for mem in potential_memories:
                    importance = mem.get("importance", 0)
                    if importance >= self.valves.memory_importance_threshold:
                        logger.info(
                            "Memory content: %s (score: %.2f)",
                            mem.get("content", ""),
                            importance,
                        )
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
        logger.info("Starting memory identification for message")

        # Identify potential memories
        potential_memories = await self._identify_potential_memories(current_message)

        # Filter potential memories by importance threshold
        important_memories = [
            mem
            for mem in potential_memories
            if mem.get("importance", 0) >= self.valves.memory_importance_threshold
        ]

        if not important_memories:
            logger.info(
                "No potential memories met the importance threshold (%.2f). No operations needed.",
                self.valves.memory_importance_threshold,
            )
            return []

        # Log the important memories identified
        logger.info(
            "Identified %d important potential memories meeting threshold (%.2f)",
            len(important_memories),
            self.valves.memory_importance_threshold,
        )

        # Directly create NEW operations for the important memories
        memory_operations = [
            {
                "operation": "NEW",
                "content": mem["content"],
                "importance": mem["importance"],
            }
            for mem in important_memories
        ]

        # Log the final set of operations
        op_summary = ", ".join(
            f"NEW({op.get('importance', 0):.2f})" for op in memory_operations
        )
        logger.info("Prepared NEW memory operations: %s", op_summary)

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
        logger.info(
            "Executing %d NEW memory creation operations", len(memory_operations)
        )
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
                    logger.info(
                        "Attempting to CREATE memory for user %s (Importance: %.2f): %s",
                        __user__.get("id", "UNKNOWN"),
                        importance,
                        content[:100] + "..." if len(content) > 100 else content,
                    )
                    memory_id = await self._create_memory(content, __user__, importance)
                    if memory_id:
                        # Use the same formatting method as for memory retrieval
                        status = self._format_memory_item(content, importance, "importance")
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
                logger.warning(
                    "Skipping unexpected operation type '%s': %s", operation_type, op
                )
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

    async def _create_memory(self, content: str, __user__: dict, importance: float = 0.0) -> Optional[str]:
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
    ) -> None:
        """
        Process memory storage asynchronously.

        Args:
            current_message_content: The current user message content
            __user__: The user dictionary object
            __event_emitter__: Optional event emitter for sending status updates
        """
        try:
            logger.info("Starting asynchronous memory storage processing")

            # First: Identify potential memories and prepare NEW operations
            memory_operations = await self._identify_and_prepare_memories(
                current_message_content
            )

            # Then: Execute the creation operations
            if memory_operations:
                await self._execute_memory_creation(
                    memory_operations, __user__, __event_emitter__
                )
            else:
                logger.info("No memory operations to execute.")

        except Exception as e:
            logger.error(
                f"Error during asynchronous memory storage: {str(e)}", exc_info=True
            )

        finally:
            logger.info("Asynchronous memory storage processing completed")
            
            # Add completion status emission for consistency with Stage 1
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
                # No need to track this task as it's short-lived
                asyncio.create_task(self._delayed_clear_status(__event_emitter__))

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
        
        logger.info(
            "Sending %s citation for user %s", citation_type, user_id or "UNKNOWN"
        )

        await self._safe_emit(
            emitter,
            {
                "type": "citation",
                "data": {
                    "document": [citation_message],
                    "metadata": [
                        {"source": source_path, "html": False}
                    ],
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
            logger.info("Processing inlet request for user %s", __user__["id"])

            # Check if there are any user messages
            user_messages = [m for m in body["messages"] if m.get("role") == "user"]
            if not user_messages:
                if self.valves.verbose_logging:
                    logger.info("No user messages found in request")
                return body

            # Get the last user message
            last_user_message = user_messages[-1].get("content", "")
            if not last_user_message:
                if self.valves.verbose_logging:
                    logger.info("Last user message is empty")
                return body

            if self.valves.verbose_logging:
                truncated_message = (
                    last_user_message[:50] + "..."
                    if len(last_user_message) > 50
                    else last_user_message
                )
                logger.info("Processing user message: %s", truncated_message)

            # -----------------------------------------------------------------
            # Stage 1: Memory Retrieval (Synchronous)
            # -----------------------------------------------------------------

            # Get user object and memories
            user = Users.get_user_by_id(__user__["id"])
            if not user:
                if self.valves.verbose_logging:
                    logger.info("User not found in database")
                return body

            db_memories = Memories.get_memories_by_user_id(__user__["id"])
            if not db_memories:
                logger.info("No memories found for user %s - no memories to process", __user__["id"])

            logger.info(
                "Found %d memories for user %s", len(db_memories), __user__["id"]
            )

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

            # If no relevant memories, emit status and return body unchanged
            if not relevant_memories:
                logger.info("No relevant memories found for the current message")

                # Emit completion status if enabled
                if self.valves.show_status and __event_emitter__:
                    if self.valves.verbose_logging:
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
                    # No need to track this task as it's short-lived
                    asyncio.create_task(self._delayed_clear_status(__event_emitter__))

            else:
                # Format memories and update context
                formatted_memories, formatted_memories_with_scores = (
                    self._format_memories_for_context(relevant_memories)
                )

                logger.info(
                    "Formatted %d relevant memories for context", len(relevant_memories)
                )

                # Update message context
                self._update_message_context(body, formatted_memories)

                # Log what's being sent to the assistant
                if self.valves.verbose_logging:
                    # In verbose mode, show full memory content with scores
                    truncated_memories = self._truncate_log_lines(
                        formatted_memories_with_scores
                    )
                    logger.info(f"Sending to assistant: {truncated_memories}")
                else:
                    # In normal mode, just show count of memories
                    logger.info(
                        f"Sending to assistant: {len(relevant_memories)} relevant memories"
                    )

                # Send citation with memories sent to assistant
                if __event_emitter__:
                    if self.valves.verbose_logging:
                        logger.info("Sending citation with memories")
                    
                    # Extract the memory items without the header
                    memory_items = formatted_memories_with_scores.split("\n")[1:]
                    
                    # Send the citation using the shared method
                    await self._send_citation(
                        __event_emitter__,
                        memory_items,
                        __user__.get("id"),
                        "read"
                    )

                # Emit completion status if enabled
                if self.valves.show_status and __event_emitter__:
                    if self.valves.verbose_logging:
                        logger.info(
                            "Emitting status: Retrieved %d relevant memories",
                            len(relevant_memories),
                        )
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
                    # No need to track this task as it's short-lived
                    asyncio.create_task(self._delayed_clear_status(__event_emitter__))

            # -----------------------------------------------------------------
            # Stage 2: Memory Storage (Asynchronous)
            # -----------------------------------------------------------------

            # Determine the message content to analyze based on settings
            if self.valves.recent_messages_count > 1:
                start_index = max(
                    0, len(user_messages) - self.valves.recent_messages_count
                )
                context_messages = user_messages[start_index:]
                current_message_content = "\n".join(
                    [m.get("content", "") for m in context_messages]
                )
                logger.info(
                    "Using last %d messages for memory storage context.",
                    len(context_messages),
                )
            else:
                current_message_content = last_user_message
                logger.info("Using the last message for memory storage context.")

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
                logger.info("Starting asynchronous memory storage task")
                # Create a background task for memory storage
                # No need to track this task as it's short-lived and will be garbage collected
                # when completed, allowing the assistant to respond immediately
                asyncio.create_task(
                    self._process_memory_storage(
                        current_message_content, __user__, __event_emitter__
                    )
                )
            else:
                # Process memory storage synchronously
                logger.info("Processing memory storage synchronously")
                await self._process_memory_storage(
                    current_message_content, __user__, __event_emitter__
                )

        except Exception as e:
            logger.error(f"Error in inlet: {str(e)}")
            if self.valves.verbose_logging:
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

# Plan: Semantic Pre-filtering for Memory Retrieval

## 1. Executive Summary

The current memory retrieval system (`AMS_MRS_v1_Module.py`) fetches all of a user's memories from the database and uses a single, large LLM call to determine their relevance. This approach is limited by the LLM's context window size, is slow, and does not scale well as the number of memories grows.

This plan outlines a new multi-stage architecture that introduces a semantic pre-filtering step using a vector database (ChromaDB). This will dramatically increase the system's memory capacity, improve response latency, and create a more scalable and robust memory system.

## 2. Analysis of Existing System

- **Data Store:** SQLite for the canonical memory text (`models/memories.py`).
- **Vector Store:** ChromaDB is used by the upstream Open WebUI application to store vector embeddings of memories, but the AMS module does not currently leverage it (`routers/memories.py`).
- **Retrieval Logic:** The AMS module's `inlet` function calls `Memories.get_memories_by_user_id` to fetch all memories. These are then formatted into a single prompt for an LLM to score relevance (`_get_relevant_memories_llm`). This is the primary performance bottleneck.

### Current Data Flow

```mermaid
graph TD
    A[User Message] --> B{inlet()};
    B --> C[Memories.get_memories_by_user_id];
    C --> D[Get ALL memories from SQLite];
    D --> E{_get_relevant_memories_llm};
    E --> F[Format ALL memories into prompt];
    F --> G[LLM API Call for Relevance Scoring];
    G --> H[Relevant Memories];
    H --> I[Update Chat Context];
```

## 3. Proposed Architecture: Hybrid Search

We will implement a hybrid search model that combines the speed of vector search with the semantic understanding of an LLM.

1.  **Stage 1: Candidate Retrieval (Vector Search):** Instead of fetching all memories, we will first perform a fast vector similarity search in ChromaDB to retrieve a small, highly relevant set of candidate memories (e.g., the top 20-30).
2.  **Stage 2: Re-ranking (LLM):** This smaller candidate set will then be passed to the existing LLM-based relevance scoring function for more nuanced re-ranking.

### Proposed Data Flow

```mermaid
graph TD
    subgraph Proposed Flow
        A_new[User Message] --> B_new{inlet()};
        B_new --> C_new[Vector Search for Candidates];
        C_new --> D_new{_get_relevant_memories_llm};
        D_new --> E_new[Format CANDIDATE memories into prompt];
        E_new --> F_new[LLM API Call for Re-ranking];
        F_new --> G_new[Relevant Memories];
        G_new --> H_new[Update Chat Context];
    end
```

## 4. Latency Impact Analysis

This change will significantly **reduce** synchronous latency.

| Stage | Current System (High Latency) | Proposed System (Low Latency) |
| :--- | :--- | :--- |
| **Initial Steps**| SQLite Query (~30ms) | Embedding + Vector Search (~150ms) |
| **LLM Call**| 2,000ms - 10,000ms+ | 500ms - 2,000ms |
| **Total Sync Time**| **~2.0 - 10.0+ seconds** | **~0.7 - 2.2 seconds** |

The proposed system trades a slow, large LLM call for a few fast, local operations, resulting in a faster and more scalable user experience.

## 5. Plan of Action

The implementation will focus on updating `AMS_MRS_v1_Module.py`:

1.  **Integrate with Upstream Services:**
    *   Modify the `inlet` function signature to accept the `VECTOR_DB_CLIENT` and `EMBEDDING_FUNCTION` objects from the Open WebUI application state. This allows the module to use the existing, configured services.

2.  **Add New Configuration Valves:**
    *   Introduce new settings in the `Valves` class to control the pre-filtering stage:
        *   `enable_vector_search`: A boolean to toggle the feature on or off.
        *   `vector_search_top_k`: An integer to define how many candidate memories to retrieve from the vector search (e.g., `25`).

3.  **Implement Vector Search Logic:**
    *   Create a new private method, `_get_candidate_memories_vector_search`, that will:
        *   Take the user's message as input.
        *   Use the `EMBEDDING_FUNCTION` to generate a query vector.
        *   Use the `VECTOR_DB_CLIENT` to perform a search on the appropriate user-specific collection (`user-memory-{user.id}`).
        *   Return the list of memory contents retrieved from the search results.

4.  **Update Core Retrieval Function:**
    *   Modify the `get_relevant_memories` function:
        *   Check if `enable_vector_search` is true.
        *   If true, call `_get_candidate_memories_vector_search` to get the pre-filtered candidate set.
        *   Pass this smaller candidate set to `_get_relevant_memories_llm` for the final re-ranking step.
        *   If false, maintain the existing behavior of fetching all memories from SQLite.
# AI Memory Architecture

Sophisticated memory system for AI assistants featuring hybrid retrieval, intelligent pruning, and vector-based deduplication. Built as a production-ready module for Open WebUI.

## Overview

This system solves a critical problem in conversational AI: **how to maintain coherent, long-term memory across sessions while staying within context limits**. Traditional approaches either keep everything (context overflow) or forget everything (no continuity). This architecture implements a hybrid solution that combines vector search pre-filtering with LLM-based re-ranking to surface only the most relevant memories for each interaction.

## Key Features

### 🔍 Hybrid Retrieval System
- **Vector search pre-filtering**: Uses ChromaDB with sentence transformers to narrow candidates from thousands to ~50 memories in milliseconds
- **LLM re-ranking**: Parallel chunked classification to select the top 5 most contextually relevant memories
- **Index-Validate-Preserve strategy**: Ensures LLM responses map correctly to original memory content without hallucination

### 🧠 Intelligent Memory Management
- **Importance-weighted retention scoring**: Combines memory importance with exponential age decay (configurable half-life)
- **Deduplication via cosine distance clustering**: Identifies and consolidates near-duplicates using local vector calculations
- **Automatic pruning**: Percentage-based and hard-limit strategies to maintain optimal collection size

### ⚡ Production-Grade Architecture
- **Async background tasks**: Memory operations don't block user interactions
- **Sequential session initialization**: Prevents race conditions during pruning/backfill/dedupe
- **Graceful shutdown handling**: Proper cleanup of connections and background tasks
- **Comprehensive error recovery**: Backfill detection and vector consistency checks

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Inlet (Retrieval)                  │
├─────────────────────────────────────────────────┤
│  1. Vector Search (ChromaDB + embeddings)       │
│     └─> Top 50 candidates by cosine similarity  │
│                                                  │
│  2. LLM Re-ranking (parallel chunks)            │
│     └─> Top 5 contextually relevant memories    │
│                                                  │
│  3. Inject into conversation context            │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Outlet (Storage)                   │
├─────────────────────────────────────────────────┤
│  1. Parse assistant response for memory ops     │
│     • 💾 CREATE new memories                    │
│     • 🗑️ DELETE outdated memories              │
│                                                  │
│  2. Execute operations (background task)        │
│                                                  │
│  3. Prune excess memories                       │
│     └─> Hybrid: importance × recency score      │
│                                                  │
│  4. Backfill missing vectors (session init)     │
└─────────────────────────────────────────────────┘
```

## Technical Highlights

### Vector Search Optimization
- Uses `all-mpnet-base-v2` embeddings (768 dimensions)
- Persistent ChromaDB storage with per-user collections
- Configurable similarity thresholds for deletion safety

### Memory Retention Algorithm
```python
retention_score = (importance_weight × importance) + ((1 - importance_weight) × e^(-λt))
# where λ = ln(2) / half_life_days, t = age in days
```

Memories less than 24 hours old receive protection score of 2.0 to prevent premature pruning.

### Deduplication Strategy
1. Generate local cosine distance matrix for all memory embeddings
2. Build adjacency graph with configurable distance threshold (default: 0.08)
3. Find connected components via BFS
4. For each cluster: keep highest retention score, average importance across duplicates
5. Delete all members and recreate canonical version

## Configuration

Key parameters (via Valves):
- `vector_search_top_k`: Candidate pool size (default: 50)
- `llm_search_top_k`: Final memory count (default: 5)
- `max_memories`: Hard collection limit (default: 1000)
- `pruning_percentage`: Proactive cleanup rate (default: 1%)
- `importance_weight`: Balance between importance/recency (default: 0.5)

## Evolution

This repo contains 99 files spanning v1.0 through v3.10, documenting the architectural evolution:
- **v1.x**: Tag-based taxonomy system
- **v2.x**: Introduction of importance scoring and pruning
- **v3.x**: Hybrid vector-LLM retrieval, deduplication, async processing

See `/Archive` for version history and `/Prompts` for prompt engineering iterations.

## Built With

- **ChromaDB**: Vector database for semantic search
- **sentence-transformers**: Embedding generation
- **Open WebUI**: Integration framework
- **asyncio**: Non-blocking background operations
- **numpy**: Efficient vector calculations

## Author

**Liv Skeete** | [liv@di.st](mailto:liv@di.st)

Built as part of an AI toolset for enhanced conversational experiences. See also: [smart-model-router](https://github.com/liv-skeete/smart-model-router) and [ai-toolkit](https://github.com/liv-skeete/ai-toolkit).

## License

MIT License - See LICENSE file for details


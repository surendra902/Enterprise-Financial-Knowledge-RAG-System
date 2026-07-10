# Enterprise Financial Knowledge RAG System
 
Most financial research still means manually reading through hundreds of pages of SEC filings to find a single number or understand a risk. This project automates that — you point it at any public company's 10-K, and you can ask questions about it in plain English and get answers grounded directly in the document.
 
It's built as a production-ready RAG (Retrieval-Augmented Generation) system, meaning the LLM never guesses — every claim in the answer is pulled from the actual filing and cited with a section, estimated page, and paragraph reference like `[Item 7. MD&A, ~page 47, ¶3]`.
 
---
 
## How it works
 
When you ingest a filing, the system downloads it directly from EDGAR, splits it into 512-word chunks, detects which SEC section each chunk belongs to (Risk Factors, MD&A, Financial Statements, etc.), and stores it in two indexes — a vector index (Qdrant) for semantic search and a BM25 index for keyword search.
 
When you ask a question, it runs both searches in parallel, merges the results using Reciprocal Rank Fusion, then passes the top candidates through a cross-encoder reranker that reads each (question, chunk) pair together to score true relevance. The best 6 chunks go to a local LLM running via Ollama, which generates the answer with inline citations.
 
Before any of that runs, the query is checked against a Redis semantic cache. If a similar enough question was asked before (cosine similarity ≥ 0.95), the cached answer is returned in under 30ms instead of running the full pipeline.
 
---

## Architecture

```
User Query
    │
    ▼
EmbeddingEngine  ←──────────── BAAI/bge-small-en-v1.5 (384-dim, CPU)
    │
    ├──► SemanticCache (Redis Stack KNN)  ──► Cache HIT → stream cached answer
    │
    ▼  (cache miss)
    ├──► [optional HyDE]  ← Ollama qwen2.5:3b (default OFF)
    │
    ├──► Dense Search  ──────── Qdrant cosine KNN
    │         ↕  RRF Fusion (k=60)
    └──► Sparse Search  ─────── BM25Okapi
              │
              ▼
         Cross-Encoder Reranker  ← ms-marco-MiniLM-L-6-v2 (22M params, 15–40ms)
              │
              ▼
         Generator  ────────────── Ollama qwen2.5:3b (sync + async SSE streaming)
              │
              ▼
         FastAPI SSE Response  →  Client
```

---

## Key Features

**Hybrid Retrieval** — Combines dense vector search (Qdrant) and keyword-based BM25 sparse search, fused with Reciprocal Rank Fusion (RRF). Neither method alone handles all query types; together they complement each other.

**Cross-Encoder Reranking** — After fusion, a `ms-marco-MiniLM-L-6-v2` cross-encoder rescores all candidate chunks by reading the full (query, passage) pair. This model was specifically chosen over the heavier `bge-reranker-base` (278M params) for a 12× CPU speed gain (15–40ms vs 80–200ms) while staying within the latency budget.

**Semantic Cache (Redis Stack)** — Query vectors are stored in Redis with a RediSearch FLAT vector index. Semantically similar queries (cosine similarity ≥ 0.95) are served instantly from cache without running the full pipeline. Cache TTL defaults to 7 days. This is the single biggest latency win for repeated or paraphrased questions.

**Human-Readable Citations** — Every chunk is tagged during ingestion with an approximate citation: section name, estimated page number (based on word count), and paragraph index. Answers reference these directly, e.g. `[Item 1A. Risk Factors, ~page 12, ¶2]`.

**SSE Streaming** — The primary endpoint (`POST /query/stream`) streams individual LLM tokens to the client via Server-Sent Events, enabling a responsive chat-like interface even over slower connections.

**HyDE (Hypothetical Document Embedding)** — Optionally, the system generates a hypothetical answer to the query first, then embeds *that* for retrieval. This can improve recall for abstract questions. It is opt-in per request (`use_hyde: true`) because it adds ~800ms on CPU.

**Financial Highlights Extraction** — Structured financial metrics (revenue, operating income, EPS, etc.) are automatically parsed from table chunks in Items 6, 7, and 8 and returned alongside the LLM answer.

**Append-Safe Persistence** — Both Qdrant (persistent file mode) and BM25 (pickle file) are append-only. Re-running ingestion for a ticker/year that's already indexed is a no-op; it never corrupts or wipes existing data.

---

## Project Structure

```
.
├── pipeline.py          # Core RAG logic (1559 lines)
│   ├── SECIngester      # Downloads & parses 10-K filings from EDGAR
│   ├── EmbeddingEngine  # BAAI/bge-small-en-v1.5 via sentence-transformers
│   ├── VectorStore      # Qdrant persistent collection
│   ├── BM25Index        # BM25Okapi sparse index (pickle-backed)
│   ├── Reranker         # Cross-encoder ms-marco-MiniLM-L-6-v2
│   ├── Retriever        # Hybrid pipeline: HyDE → dense → sparse → RRF → rerank
│   ├── Generator        # Ollama sync + async token streaming
│   ├── FinRAGEvaluator  # ROUGE, BERTScore, BLEU, RAGAS metrics
│   └── FinRAG           # Top-level orchestrator
│
├── api.py               # FastAPI application (1044 lines)
│   ├── SemanticCache    # Redis Stack KNN semantic cache
│   ├── LatencyMiddleware# ASGI middleware for X-Request-Latency-Ms header
│   └── Routes           # /query/stream, /query, /ingest, /cache/*, /health
│
├── utils.py             # Shared utilities (573 lines, zero ML dependencies)
│   ├── LatencyBreakdown # Per-stage timing dataclass
│   ├── profile_stage    # Context manager for timing pipeline stages
│   ├── EntityLinker     # SEC section detector + citation builder
│   ├── extract_financial_highlights  # Financial table parser
│   └── NumericUtils     # Financial number normalisation & comparison
│
├── requirements.txt     # All Python dependencies
├── bm25_index.pkl       # Persisted BM25 index (auto-created on first ingest)
└── qdrant_db/           # Qdrant persistent vector store (auto-created)
```

---

## How It Works

### 1. Ingestion Pipeline

**Trigger:** `POST /ingest` with `{"ticker": "MSFT", "year": 2024}`

**Steps:**

`SECIngester.download_and_parse()` calls the EDGAR API via `edgartools` to find and download the 10-K HTML. The HTML is parsed by `unstructured` into a list of elements (paragraphs and tables).

`_section_aware_chunk()` walks through elements, detecting SEC section headings using regex patterns (Items 1–15). An `EntityLinker` instance tracks the current section, cumulative word count, and paragraph index. Each text block is split into 512-word chunks with a 64-word overlap. Every chunk is assigned a citation string like `[Item 7. MD&A, ~page 47, ¶3]`.

The resulting `Chunk` objects are batch-embedded by `EmbeddingEngine` using `BAAI/bge-small-en-v1.5` (384-dimensional vectors, L2-normalised so cosine similarity equals dot product). Chunks are upserted into Qdrant and added to the BM25 index. Both stores deduplicate by `chunk_id`, so re-ingestion is safe.

**Typical output:** ~800–2000 chunks per 10-K filing.

---

### 2. Retrieval Pipeline

**Entry point:** `Retriever.retrieve(query, top_k, ticker, year, use_hyde, use_reranker, precomputed_vec)`

**Stage 1 — Embedding (~15ms)**
The query is embedded with a BGE-specific prefix: `"Represent this sentence for searching relevant passages: {query}"`. If the API layer already embedded the query for the cache lookup, the precomputed vector is passed in to skip double-embedding.

**Stage 2 — HyDE (optional, ~800ms)**
If `use_hyde=True`, an Ollama call generates a short hypothetical answer. That answer is embedded instead of the raw query. Disabled by default; opt-in per request.

**Stage 3 — Dense Retrieval (~5–20ms)**
Qdrant KNN search over the 384-dim cosine index. Returns 18 candidates (= `TOP_K × 3`) when reranking is enabled, or 12 otherwise. Filtered by ticker and/or year if provided.

**Stage 4 — Sparse Retrieval (~2–5ms)**
BM25Okapi search over the same document set. Returns 18 candidates, also filtered.

**Stage 5 — RRF Fusion (~1ms)**
Reciprocal Rank Fusion merges the two ranked lists: `score = Σ 1/(k + rank)` with `k=60`. This is a parameter-free, robust fusion method that handles scale differences between dense and sparse scores. The merged list has 18 candidates.

**Stage 6 — Cross-Encoder Reranking (~15–40ms)**
The `ms-marco-MiniLM-L-6-v2` cross-encoder reads all 18 `(query, chunk[:512])` pairs in a single batch and assigns a relevance logit to each. The top 6 chunks by logit score are returned as the final context.

---

### 3. Generation

**Entry point:** `Generator.generate()` (sync) or `Generator.astream()` (async streaming)

The generator builds a prompt that includes:
- A system instruction that prohibits hallucination and requires every claim to be followed by a citation tag
- All 6 retrieved chunks, each prefixed with its citation string, ticker, year, section, and a `[TABLE]` flag if applicable
- The user's question

`astream()` runs Ollama's synchronous streaming API in a background thread and bridges tokens to an `asyncio.Queue`, yielding each token as it arrives. The FastAPI SSE endpoint consumes this generator and forwards tokens to the client.

The system prompt enforces: every factual claim must cite its source; if context is insufficient, respond with `INSUFFICIENT_CONTEXT`; never fabricate numbers or dates.

---

### 4. Semantic Cache (Redis)

**Index:** RediSearch FLAT vector index over 384-dim FLOAT32 vectors, COSINE distance metric.

**Lookup flow (cache miss → store):**
1. Embed the query (~15ms)
2. KNN search for the nearest stored query vector (~3–8ms)
3. If cosine distance ≤ 0.05 (similarity ≥ 0.95), return the cached answer immediately
4. On a miss: run the full pipeline, stream tokens to the client, then save the result to Redis asynchronously (background task — does not block the response)

Each cache entry stores: original query text, full LLM answer, serialised chunk list (truncated to 500 chars per chunk), ticker, year, embedding bytes, hit counter, and creation timestamp. TTL defaults to 7 days.

The cache degrades gracefully if Redis is unavailable — every method becomes a no-op.

---

### 5. Evaluation

`FinRAGEvaluator` supports the following metrics when optional dependencies are installed:

| Metric | Library | What it measures |
|---|---|---|
| ROUGE-1/2/L | `rouge-score` | n-gram overlap with ground truth |
| BERTScore | `bert-score` | Semantic similarity using DistilBERT |
| BLEU | `nltk` | Precision-based n-gram overlap |
| Answer Relevancy | `ragas` | LLM-judged relevance to the question |
| Faithfulness | `ragas` | Whether the answer is grounded in context |
| Context Recall | `ragas` | How much of the ground truth is in context |
| Context Precision | `ragas` | Proportion of retrieved context that is useful |
| Numeric Match | `utils.NumericUtils` | Exact financial number comparison (±1%) |

---

## API Reference

### `POST /query/stream` — Primary endpoint (SSE)

Streams the answer token-by-token via Server-Sent Events.

**Request body:**
```json
{
  "query": "What were the total revenues and operating income?",
  "ticker": "MSFT",
  "year": 2024,
  "top_k": 6,
  "use_hyde": false,
  "use_reranker": true
}
```

**Event stream:**
```
data: {"type": "metadata", "chunks": [...], "financial_highlights": {...}}

data: {"type": "token", "content": "Microsoft"}
data: {"type": "token", "content": " reported"}
...

data: {"type": "done", "latency_ms": {...}, "cached": false, "model": "qwen2.5:3b"}
```

On a cache hit:
```
data: {"type": "cache_hit", "answer": "...", "similarity": 0.9823, "latency_ms": {...}}
data: {"type": "done", "latency_ms": {...}, "cached": true}
```

---

### `POST /query` — Synchronous fallback

Returns the full answer in a single JSON response. Checks Redis cache first. Intended for clients that do not support SSE.

**Response:**
```json
{
  "answer": "Microsoft's total revenue was $245.1B [Item 7. MD&A, ~page 12, ¶3]...",
  "chunks": [{"chunk_id": "...", "text": "...", "citation": "[Item 7. MD&A, ~page 12, ¶3]", ...}],
  "model": "qwen2.5:3b",
  "latency_s": 3.42,
  "latency_breakdown": {
    "cache_ms": 4.1, "embed_ms": 14.8, "dense_ms": 8.2,
    "sparse_ms": 3.1, "fusion_ms": 0.4, "rerank_ms": 28.6,
    "generation_ms": 2900.0, "total_ms": 2959.2
  },
  "financial_highlights": {"revenue": {"2024": 245122000000.0}},
  "cached": false
}
```

---

### `POST /ingest`

Downloads and indexes a 10-K filing. Runs in a thread pool (non-blocking). Expect 2–5 minutes per filing.

```json
// Request
{"ticker": "MSFT", "year": 2024}

// Response
{"status": "ok", "ticker": "MSFT", "year": 2024, "chunks_indexed": 1247}
```

---

### `GET /cache/stats`

```json
{
  "available": true,
  "total_requests": 142,
  "total_hits": 89,
  "hit_rate": 0.627,
  "index_num_docs": 73,
  "similarity_threshold": 0.95
}
```

---

### `DELETE /cache`

Flush all entries or entries for a specific ticker:
```json
// Flush all
{}

// Flush MSFT only
{"ticker": "MSFT"}
```

---

### `GET /health`

Liveness probe. Returns Redis status, Qdrant collection count, and model names.

---

## Installation

**Prerequisites:**
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running locally
- Docker (for Redis Stack)

**1. Clone the repository**
```bash
git clone https://github.com/Pragna-echuri/Enterprise-Financial-Knowledge-RAG-System.git
cd Enterprise-Financial-Knowledge-RAG-System
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull the LLM**
```bash
ollama pull qwen2.5:3b
```

**4. Start Redis Stack** (required for semantic cache)
```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

---

## Quick Start

**Start the API server:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

**Ingest a 10-K filing** (one-time per ticker/year):
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker": "MSFT", "year": 2024}'
```

**Query with streaming:**
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the main risk factors?", "ticker": "MSFT", "year": 2024}'
```

**Query synchronously:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was total revenue?", "ticker": "MSFT", "year": 2024}'
```

**Check API docs:**  
Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Configuration

All configuration is via environment variables or constants in `pipeline.py` / `api.py`:

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | `None` | Redis password (optional) |
| `CACHE_TTL_SECONDS` | `604800` (7 days) | Redis entry TTL |
| `CACHE_SIM_THRESHOLD` | `0.95` | Minimum cosine similarity for cache hit |
| `USE_RERANKER` | `true` | Enable/disable cross-encoder reranking |
| `QDRANT_PATH` | `./qdrant_db` | Local Qdrant storage directory |
| `BM25_INDEX_PATH` | `./bm25_index.pkl` | BM25 index pickle file |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Sentence transformer model |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model name |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `6` | Final number of chunks returned |
| `RERANK_CANDIDATES` | `18` | Candidates fed to cross-encoder (TOP_K × 3) |

---

## Latency Budget

Target: **< 5 seconds** end-to-end on CPU for a cache miss.

| Stage | Typical (ms) | Notes |
|---|---|---|
| Redis cache lookup | 3–8 | Near-instant on hit |
| Query embedding | 12–18 | bge-small, CPU |
| HyDE generation | ~800 | Off by default |
| Qdrant KNN | 5–20 | 18 candidates |
| BM25 search | 2–5 | In-memory |
| RRF fusion | < 1 | Pure Python |
| Cross-encoder rerank | 15–40 | 18 pairs, MiniLM-L-6 |
| LLM generation | 2000–4000 | qwen2.5:3b, 6 chunks |
| **Total (no cache)** | **~2100–4100** | |
| **Total (cache hit)** | **~20–30** | |

The reranker model choice (`ms-marco-MiniLM-L-6-v2`, 22M params) is a deliberate v3 decision over `bge-reranker-base` (278M params): 12× faster on CPU with acceptable quality trade-off for the latency budget.

---

## Dependencies

**Core pipeline:**
- `edgartools>=2.0` — EDGAR filing download
- `unstructured[html]>=0.12` — HTML parsing
- `qdrant-client>=1.8` — Vector store
- `sentence-transformers>=2.7` — Embedding model + CrossEncoder
- `rank-bm25>=0.2.2` — BM25 sparse index
- `ollama>=0.2.0` — Local LLM client

**API:**
- `fastapi>=0.110`
- `uvicorn[standard]>=0.29`
- `pydantic>=2.0`
- `redis[hiredis]>=4.6.0` — Semantic cache (Redis Stack)

**Evaluation (optional):**
- `rouge-score>=0.1.2`
- `bert-score>=0.3.13`
- `nltk>=3.8`
- `ragas>=0.1.10`
- `datasets>=2.18`

**Logging:**
- `loguru>=0.7`

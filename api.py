"""
api.py — FinRAG v3 FastAPI Application
========================================
Responsibilities
----------------
  SemanticCache     Redis Stack vector cache; returns cached answers for
                    semantically similar queries (cosine sim ≥ 0.95).
  LatencyMiddleware ASGI middleware that injects X-Request-Latency-Ms header
                    and logs per-request timing.
  Routes
    POST /query/stream   SSE streaming endpoint (primary, low-latency path).
    POST /query          Synchronous endpoint (for simple clients / testing).
    POST /ingest         Ingest a 10-K filing into Qdrant + BM25.
    GET  /cache/stats    Redis cache hit-rate and key-count statistics.
    DELETE /cache        Flush the semantic cache (optionally by ticker).
    GET  /health         Liveness probe.

Semantic cache flow
-------------------
  1. Embed query with bge-small   (~15 ms)
  2. Redis KNN search             (~3-8 ms)
     Threshold: cosine distance ≤ 0.05  (= similarity ≥ 0.95)
  3a. HIT  → stream cached answer instantly, tag response as cached.
  3b. MISS → run full pipeline, stream tokens, save result to Redis
             asynchronously (background task — does not delay response).

SSE event protocol
------------------
  {"type": "cache_hit",  "answer": "...", "latency_ms": {...}}
  {"type": "metadata",   "chunks": [...], "financial_highlights": {...}}
  {"type": "token",      "content": "..."}
  {"type": "done",       "latency_ms": {...}}
  {"type": "error",      "message": "..."}

Requirements
------------
  pip install "redis[hiredis]>=4.6" fastapi uvicorn[standard] pydantic loguru
  docker run -d -p 6379:6379 redis/redis-stack-server:latest
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncGenerator, Dict, List, Optional

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("finrag.api")          # type: ignore[assignment]
    logger.setLevel(_logging.INFO)
    _logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s")

# ── Redis Stack ───────────────────────────────────────────────────────────────
# try:
#     import redis.asyncio as aioredis                                    # type: ignore
#     from redis.commands.search.field import (                           # type: ignore
#         NumericField, TagField, TextField, VectorField,
#     )
#     from redis.commands.search.index_definition import (                 # type: ignore
#         IndexDefinition, IndexType,
#     )
#     from redis.commands.search.query import Query as RediSearchQuery    # type: ignore
#     _REDIS_AVAILABLE = True
# except ImportError:
#     _REDIS_AVAILABLE = False
#     logger.warning(
#         "redis[hiredis] not installed — semantic cache disabled.\n"
#         "Install: pip install 'redis[hiredis]>=4.6'"
#     )
try:
    import redis.asyncio as aioredis
    print("✓ aioredis")

    from redis.commands.search.field import (
        NumericField, TagField, TextField, VectorField,
    )
    print("✓ field")

    from redis.commands.search.index_definition import (
        IndexDefinition, IndexType,
    )
    print("✓ index_definition")

    from redis.commands.search.query import Query as RediSearchQuery
    print("✓ query")

    _REDIS_AVAILABLE = True

except Exception as e:
    _REDIS_AVAILABLE = False
    print("\nREDIS IMPORT ERROR:")
    print(type(e).__name__)
    print(str(e))

# ── Local pipeline ────────────────────────────────────────────────────────────
from pipeline import (
    EMBED_DIM,
    OLLAMA_MODEL,
    RERANKER_MODEL,
    TOP_K,
    FinRAG,
    QueryResult,
    RAGResponse,
)
from utils import LatencyBreakdown, profile_stage


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

REDIS_HOST               = os.getenv("REDIS_HOST",  "localhost")
REDIS_PORT               = int(os.getenv("REDIS_PORT", "6379"))

print(f"DEBUG REDIS_HOST={REDIS_HOST}")
print(f"DEBUG REDIS_PORT={REDIS_PORT}")

REDIS_PASSWORD           = os.getenv("REDIS_PASSWORD", None)

CACHE_PREFIX             = "finrag:cache:"
CACHE_INDEX_NAME         = "idx:finrag_cache"
CACHE_TTL_SECONDS        = int(os.getenv("CACHE_TTL_SECONDS", str(86_400 * 7)))  # 7 days
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.95"))
# Redis COSINE distance = 1 − cosine_sim → threshold in distance space:
_CACHE_DISTANCE_THRESHOLD  = 1.0 - CACHE_SIMILARITY_THRESHOLD   # 0.05

_FINRAG_USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"


# ══════════════════════════════════════════════════════════════════════════════
# Semantic Cache  (Redis Stack / RediSearch)
# ══════════════════════════════════════════════════════════════════════════════

class CacheHit:
    """Returned by SemanticCache.get() on a successful cache hit."""
    __slots__ = ("answer", "chunks_json", "query_text", "distance")

    def __init__(
        self,
        answer:      str,
        chunks_json: str,
        query_text:  str,
        distance:    float,
    ) -> None:
        self.answer      = answer
        self.chunks_json = chunks_json
        self.query_text  = query_text
        self.distance    = distance


class SemanticCache:
    """
    Redis Stack vector cache for semantic query deduplication.

    Storage model
    -------------
    Each entry is a Redis Hash at key ``finrag:cache:{hex_id}``:

        query_text  — original query string
        answer      — full LLM answer (plain text)
        chunks_json — JSON-encoded list of QueryResult dicts
        ticker      — ticker filter used (or empty string)
        year        — year filter used (or "0")
        embedding   — 384 × float32 bytes (little-endian, struct.pack)
        hits        — integer hit counter
        created_at  — Unix timestamp string

    A RediSearch FLAT vector index over ``embedding`` enables KNN search.
    The index is created automatically on first use if it does not exist.

    Graceful degradation
    --------------------
    If Redis is unavailable (ImportError or connection error), every method
    becomes a no-op and get() always returns None — the pipeline runs normally
    without caching.
    """

    def __init__(
        self,
        host:       str   = REDIS_HOST,
        port:       int   = REDIS_PORT,
        password:   Optional[str] = REDIS_PASSWORD,
        embed_dim:  int   = EMBED_DIM,
        ttl:        int   = CACHE_TTL_SECONDS,
    ) -> None:
        self._embed_dim = embed_dim
        self._ttl       = ttl
        self._available = False
        self._redis: Optional[aioredis.Redis] = None   # type: ignore[type-arg]
        self._total_requests = 0
        self._total_hits     = 0

        if not _REDIS_AVAILABLE:
            return

        try:
            self._redis = aioredis.Redis(
                host=host, port=port,
                password=password,
                decode_responses=False,   # we handle bytes ourselves
                socket_connect_timeout=2,
            )
            self._available = True
            logger.info(f"SemanticCache: Redis at {host}:{port}")
        except Exception as exc:
            logger.warning(f"Redis connection failed ({exc}) — cache disabled.")

    # ── index lifecycle ───────────────────────────────────────────────────────

    async def ensure_index(self) -> None:
        """
        Create the RediSearch vector index if it does not already exist.

        Schema
        ------
        @embedding  VECTOR FLAT FLOAT32 DIM=384 COSINE
        @query_text TEXT
        @ticker     TAG
        @year       NUMERIC
        """
        if not self._available or self._redis is None:
            return
        try:
            await self._redis.ft(CACHE_INDEX_NAME).info()
            logger.info(f"SemanticCache: index '{CACHE_INDEX_NAME}' already exists.")
        except Exception:
            # Index does not exist — create it
            schema = [
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE":            "FLOAT32",
                        "DIM":             self._embed_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                TextField("query_text"),
                TagField("ticker"),
                NumericField("year"),
            ]
            definition = IndexDefinition(
                prefix=[CACHE_PREFIX],
                index_type=IndexType.HASH,
            )
            await self._redis.ft(CACHE_INDEX_NAME).create_index(
                schema, definition=definition
            )
            logger.success(f"SemanticCache: created index '{CACHE_INDEX_NAME}'.")

    async def ping(self) -> bool:
        """Return True if Redis responds to PING."""
        if not self._available or self._redis is None:
            return False
        try:
            return await self._redis.ping()
        except Exception:
            return False

    # ── cache lookup ──────────────────────────────────────────────────────────

    async def get(
        self,
        query_vec: List[float],
        ticker:    Optional[str] = None,
        year:      Optional[int] = None,
    ) -> Optional[CacheHit]:
        """
        Perform a KNN search over stored query vectors.

        Returns a ``CacheHit`` when the nearest neighbour has cosine
        distance ≤ ``_CACHE_DISTANCE_THRESHOLD`` (i.e., similarity ≥ 0.95),
        otherwise ``None``.

        When a hit is returned the entry's ``hits`` counter is incremented
        in the background (fire-and-forget, no await).
        """
        if not self._available or self._redis is None:
            return None

        self._total_requests += 1

        vec_bytes = struct.pack(f"{self._embed_dim}f", *query_vec)

        # Build base filter on ticker/year if supplied
        filter_expr = "*"
        if ticker:
            filter_expr = f"(@ticker:{{{ticker}}})"
        if year:
            year_clause = f"(@year:[{year} {year}])"
            filter_expr = f"({filter_expr} {year_clause})" if ticker else year_clause

        q = (
            RediSearchQuery(
                f"{filter_expr}=>[KNN 1 @embedding $vec AS __score]"
            )
            .sort_by("__score")
            .return_fields("query_text", "answer", "chunks_json", "__score")
            .paging(0, 1)
            .dialect(2)
        )

        try:
            results = await self._redis.ft(CACHE_INDEX_NAME).search(
                q, query_params={"vec": vec_bytes}
            )
        except Exception as exc:
            logger.warning(f"Cache search failed: {exc}")
            return None

        if results.total == 0:
            return None

        doc      = results.docs[0]
        distance = float(getattr(doc, "__score", 1.0))

        if distance > _CACHE_DISTANCE_THRESHOLD:
            return None   # Not similar enough

        # Increment hit counter (fire-and-forget)
        key = doc.id
        asyncio.create_task(self._increment_hits(key))

        self._total_hits += 1
        return CacheHit(
            answer=getattr(doc, "answer", ""),
            chunks_json=getattr(doc, "chunks_json", "[]"),
            query_text=getattr(doc, "query_text", ""),
            distance=distance,
        )

    # ── cache store ───────────────────────────────────────────────────────────

    async def put(
        self,
        query_text: str,
        query_vec:  List[float],
        answer:     str,
        chunks:     List[QueryResult],
        ticker:     Optional[str] = None,
        year:       Optional[int] = None,
    ) -> None:
        """
        Store a query-answer pair in Redis with a TTL.

        Called asynchronously after streaming completes so it never delays
        the response to the user.
        """
        if not self._available or self._redis is None:
            return

        import hashlib
        key_suffix = hashlib.md5(
            f"{query_text}{ticker}{year}".encode()
        ).hexdigest()[:16]
        key = f"{CACHE_PREFIX}{key_suffix}"

        vec_bytes   = struct.pack(f"{self._embed_dim}f", *query_vec)
        chunks_json = json.dumps([
            {
                "chunk_id":    c.chunk_id,
                "text":        c.text[:500],   # truncate to save Redis memory
                "ticker":      c.ticker,
                "year":        c.year,
                "sec_section": c.sec_section,
                "score":       c.score,
                "is_table":    c.is_table,
                "citation":    c.citation,
            }
            for c in chunks
        ])

        mapping: Dict[str, Any] = {
            "query_text":  query_text,
            "answer":      answer,
            "chunks_json": chunks_json,
            "ticker":      ticker or "",
            "year":        str(year) if year else "0",
            "embedding":   vec_bytes,
            "hits":        "0",
            "created_at":  str(time.time()),
        }

        try:
            pipe = self._redis.pipeline()
            await pipe.hset(key, mapping=mapping)
            await pipe.expire(key, self._ttl)
            await pipe.execute()
            logger.debug(f"Cache: stored '{key}' (TTL={self._ttl}s)")
        except Exception as exc:
            logger.warning(f"Cache put failed: {exc}")

    # ── stats & management ────────────────────────────────────────────────────

    async def stats(self) -> Dict[str, Any]:
        """Return hit-rate, key-count, and index info."""
        base: Dict[str, Any] = {
            "available":      self._available,
            "total_requests": self._total_requests,
            "total_hits":     self._total_hits,
            "hit_rate":       round(self._total_hits / max(1, self._total_requests), 4),
        }
        if not self._available or self._redis is None:
            return base
        try:
            info       = await self._redis.ft(CACHE_INDEX_NAME).info()
            base["index_num_docs"]   = info.get("num_docs", "?")
            base["index_num_terms"]  = info.get("num_terms", "?")
            base["similarity_threshold"] = CACHE_SIMILARITY_THRESHOLD
        except Exception as exc:
            base["index_error"] = str(exc)
        return base

    async def clear(self, ticker: Optional[str] = None) -> int:
        """
        Delete cache entries.  If ``ticker`` is supplied, only delete entries
        for that ticker; otherwise flush all entries.

        Returns the number of keys deleted.
        """
        if not self._available or self._redis is None:
            return 0
        deleted = 0
        try:
            if ticker:
                # Scan for keys matching the ticker field
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor, match=f"{CACHE_PREFIX}*", count=100
                    )
                    for key in keys:
                        stored_ticker = await self._redis.hget(key, "ticker")
                        if stored_ticker and stored_ticker.decode() == ticker:
                            await self._redis.delete(key)
                            deleted += 1
                    if cursor == 0:
                        break
            else:
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor, match=f"{CACHE_PREFIX}*", count=100
                    )
                    if keys:
                        await self._redis.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
        except Exception as exc:
            logger.error(f"Cache clear failed: {exc}")
        logger.info(f"Cache: cleared {deleted} entries (ticker={ticker})")
        return deleted

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _increment_hits(self, key: str) -> None:
        try:
            if self._redis:
                await self._redis.hincrby(key, "hits", 1)
        except Exception:
            pass

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()


# ══════════════════════════════════════════════════════════════════════════════
# Latency Middleware
# ══════════════════════════════════════════════════════════════════════════════

class LatencyMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware that:
      * Times every request end-to-end.
      * Injects ``X-Request-Latency-Ms`` response header.
      * Logs METHOD path → status (latency ms) at INFO level.

    Note: For streaming responses the timer stops when the *first* byte is
    written, not when the stream closes.  Use the ``latency_ms`` field in the
    final SSE ``done`` event for accurate end-to-end timing.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        t0       = time.perf_counter()
        response = await call_next(request)
        elapsed  = (time.perf_counter() - t0) * 1_000
        response.headers["X-Request-Latency-Ms"] = f"{elapsed:.1f}"
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed:.0f}ms)"
        )
        return response


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query:        str  = Field(..., min_length=3, description="Question to answer")
    ticker:       Optional[str] = Field(None, description="Filter by ticker, e.g. MSFT")
    year:         Optional[int] = Field(None, ge=2000, le=2100)
    top_k:        int  = Field(TOP_K, ge=1, le=20)
    use_hyde:     bool = Field(False,  description="Enable HyDE (adds ~800ms on CPU)")
    use_reranker: bool = Field(True,   description="Enable cross-encoder reranking")

    model_config = {"json_schema_extra": {
        "example": {
            "query":   "What were the total revenues and operating income?",
            "ticker":  "MSFT",
            "year":    2024,
            "top_k":   6,
            "use_hyde":     False,
            "use_reranker": True,
        }
    }}


class IngestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10)
    year:   int = Field(..., ge=2000, le=2100)


class ChunkOut(BaseModel):
    chunk_id:    str
    text:        str
    ticker:      str
    year:        int
    sec_section: str
    score:       float
    is_table:    bool
    citation:    str   # ★ v3: "[Item 7. MD&A, ~page 47, ¶3]"


class LatencyOut(BaseModel):
    cache_ms:      float
    embed_ms:      float
    hyde_ms:       float
    dense_ms:      float
    sparse_ms:     float
    fusion_ms:     float
    rerank_ms:     float
    generation_ms: float
    retrieval_ms:  float
    total_ms:      float


class QueryResponse(BaseModel):
    answer:               str
    chunks:               List[ChunkOut]
    model:                str
    latency_s:            float
    latency_breakdown:    Optional[LatencyOut] = None
    financial_highlights: Optional[Dict[str, Any]] = None
    cached:               bool = False


class ClearCacheRequest(BaseModel):
    ticker: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Application lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise FinRAG and SemanticCache.  Shutdown: close Redis."""
    logger.info("FinRAG v3 API starting …")

    finrag = FinRAG(use_reranker=_FINRAG_USE_RERANKER)
    cache  = SemanticCache()

    await cache.ensure_index()
    redis_ok = await cache.ping()
    logger.info(f"Redis available: {redis_ok}")

    app.state.finrag = finrag
    app.state.cache  = cache

    logger.success("FinRAG v3 API ready.")
    yield

    # ── shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down …")
    await cache.close()


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="FinRAG v3 API",
    description=(
        "Financial Retrieval-Augmented Generation over SEC 10-K filings.\n\n"
        "Key features: semantic cache (Redis), SSE streaming, "
        "cross-encoder reranking, human-readable citations."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(LatencyMiddleware)


# ── Dependency helpers ────────────────────────────────────────────────────────

def get_finrag(request: Request) -> FinRAG:
    return request.app.state.finrag


def get_cache(request: Request) -> SemanticCache:
    return request.app.state.cache


# ══════════════════════════════════════════════════════════════════════════════
# SSE streaming helper
# ══════════════════════════════════════════════════════════════════════════════

async def _sse_pipeline(
    req:    QueryRequest,
    finrag: FinRAG,
    cache:  SemanticCache,
) -> AsyncGenerator[str, None]:
    """
    Full pipeline as an async Server-Sent Events generator.

    Event sequence (success path)
    ------------------------------
    1. [metadata]   chunk list + financial highlights (always emitted)
    2. [cache_hit]  if Redis hit → emit cached answer + done, return early
    3. [token ×N]   individual LLM tokens (streaming, miss path)
    4. [done]       final latency breakdown

    Error path
    ----------
    Any exception emits a single [error] event, then [done].
    """
    lat = LatencyBreakdown()

    try:
        # ── Step 1: Embed query ───────────────────────────────────────────
        with profile_stage(lat, "embed"):
            query_vec: List[float] = await asyncio.to_thread(
                finrag.embed_engine.embed_query, req.query
            )

        # ── Step 2: Check semantic cache ──────────────────────────────────
        with profile_stage(lat, "cache"):
            hit = await cache.get(
                query_vec=query_vec,
                ticker=req.ticker,
                year=req.year,
            )

        if hit is not None:
            similarity = round(1.0 - hit.distance, 4)
            logger.info(
                f"Cache HIT | sim={similarity:.4f} | "
                f"matched='{hit.query_text[:60]}'"
            )
            cached_chunks = _deserialise_chunks(hit.chunks_json)
            yield _sse(
                "cache_hit",
                {
                    "answer":    hit.answer,
                    "chunks":    cached_chunks,
                    "cached":    True,
                    "similarity": similarity,
                    "latency_ms": lat.to_dict(),
                },
            )
            yield _sse("done", {"latency_ms": lat.to_dict(), "cached": True})
            return

        # ── Step 3: Retrieve chunks ───────────────────────────────────────
        chunks, ret_lat = await asyncio.to_thread(
            finrag.retriever.retrieve,
            req.query,
            req.top_k,
            req.ticker,
            req.year,
            req.use_hyde,
            req.use_reranker,
            query_vec,           # precomputed_vec — skip re-embedding
        )
        # Merge per-stage retrieval timing into lat
        lat.hyde_ms   = ret_lat.hyde_ms
        lat.dense_ms  = ret_lat.dense_ms
        lat.sparse_ms = ret_lat.sparse_ms
        lat.fusion_ms = ret_lat.fusion_ms
        lat.rerank_ms = ret_lat.rerank_ms

        if not chunks:
            yield _sse(
                "error",
                {"message": f"INSUFFICIENT_CONTEXT: No passages found for "
                            f"ticker={req.ticker}, year={req.year}."},
            )
            yield _sse("done", {"latency_ms": lat.to_dict(), "cached": False})
            return

        # ── Step 4: Emit chunk metadata ───────────────────────────────────
        from utils import extract_financial_highlights
        highlights = extract_financial_highlights(chunks)
        yield _sse(
            "metadata",
            {
                "chunks": [_chunk_to_dict(c) for c in chunks],
                "financial_highlights": highlights,
            },
        )

        # ── Step 5: Stream generation tokens ─────────────────────────────
        t_gen_start     = time.perf_counter()
        collected_tokens: List[str] = []

        async for token in finrag.generator.astream(req.query, chunks):
            collected_tokens.append(token)
            yield _sse("token", {"content": token})

        lat.generation_ms = (time.perf_counter() - t_gen_start) * 1_000
        full_answer = "".join(collected_tokens)

        # ── Step 6: Save to cache (background, non-blocking) ──────────────
        asyncio.create_task(
            cache.put(
                query_text=req.query,
                query_vec=query_vec,
                answer=full_answer,
                chunks=chunks,
                ticker=req.ticker,
                year=req.year,
            )
        )

        # ── Step 7: Done ──────────────────────────────────────────────────
        lat.log(logger)
        yield _sse(
            "done",
            {
                "latency_ms": lat.to_dict(),
                "cached":     False,
                "model":      finrag.generator._model,
            },
        )

    except Exception as exc:
        logger.exception(f"SSE pipeline error: {exc}")
        yield _sse("error", {"message": str(exc)})
        yield _sse("done",  {"latency_ms": lat.to_dict(), "cached": False})


def _sse(event_type: str, data: Dict[str, Any]) -> str:
    """Format a single Server-Sent Events message."""
    payload = json.dumps({"type": event_type, **data}, default=str)
    return f"data: {payload}\n\n"


def _chunk_to_dict(c: QueryResult) -> Dict[str, Any]:
    return {
        "chunk_id":    c.chunk_id,
        "text":        c.text,
        "ticker":      c.ticker,
        "year":        c.year,
        "sec_section": c.sec_section,
        "score":       round(c.score, 6),
        "is_table":    c.is_table,
        "citation":    c.citation,
    }


def _deserialise_chunks(chunks_json: str) -> List[Dict[str, Any]]:
    try:
        return json.loads(chunks_json)
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

# ── POST /query/stream  (primary endpoint — SSE) ──────────────────────────────

@app.post(
    "/query/stream",
    summary="Stream answer tokens via Server-Sent Events",
    response_description=(
        "text/event-stream — emits metadata, token, cache_hit, done, or error events"
    ),
)
async def query_stream(
    req:    QueryRequest,
    finrag: FinRAG         = Depends(get_finrag),
    cache:  SemanticCache  = Depends(get_cache),
) -> StreamingResponse:
    """
    Primary query endpoint.

    Returns a ``text/event-stream`` response.  Each event is a JSON object
    with a ``"type"`` field:

    * ``cache_hit``  — full answer returned from Redis (no generation needed)
    * ``metadata``   — retrieved chunks + financial highlights
    * ``token``      — one LLM output token
    * ``done``       — final latency breakdown
    * ``error``      — pipeline error message

    The ``X-Accel-Buffering: no`` header disables Nginx proxy buffering so
    tokens reach the browser without batching.
    """
    return StreamingResponse(
        _sse_pipeline(req, finrag, cache),
        media_type="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":      "keep-alive",
        },
    )


# ── POST /query  (synchronous fallback) ──────────────────────────────────────

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Synchronous query (non-streaming)",
)
def query_sync(
    req:    QueryRequest,
    finrag: FinRAG         = Depends(get_finrag),
    cache:  SemanticCache  = Depends(get_cache),
) -> QueryResponse:
    """
    Synchronous query endpoint for clients that do not support SSE.

    Checks the Redis cache first.  On a miss, runs the full pipeline
    and saves the result to cache before returning.

    Note: FastAPI runs plain ``def`` endpoints in a thread pool, so this
    does not block the event loop.
    """
    t0 = time.perf_counter()

    # Embed query
    query_vec = finrag.embed_engine.embed_query(req.query)

    # Synchronous Redis check (run in thread since we're already off the loop)
    cache_hit: Optional[CacheHit] = None
    if cache._available:
        try:
            cache_hit = asyncio.get_event_loop().run_until_complete(
                cache.get(query_vec, ticker=req.ticker, year=req.year)
            )
        except Exception:
            pass  # degrade gracefully

    if cache_hit is not None:
        cached_chunks = _deserialise_chunks(cache_hit.chunks_json)
        return QueryResponse(
            answer=cache_hit.answer,
            chunks=[ChunkOut(**c) for c in cached_chunks if _is_valid_chunk(c)],
            model=finrag.generator._model,
            latency_s=time.perf_counter() - t0,
            cached=True,
        )

    resp: RAGResponse = finrag.query(
        query=req.query,
        ticker=req.ticker,
        year=req.year,
        top_k=req.top_k,
        use_hyde=req.use_hyde,
        use_reranker=req.use_reranker,
        precomputed_vec=query_vec,
    )

    # Save to cache (best-effort; we're in a sync thread)
    if cache._available:
        try:
            asyncio.get_event_loop().run_until_complete(
                cache.put(
                    query_text=req.query,
                    query_vec=query_vec,
                    answer=resp.answer,
                    chunks=resp.chunks,
                    ticker=req.ticker,
                    year=req.year,
                )
            )
        except Exception:
            pass

    lat_out: Optional[LatencyOut] = None
    if resp.latency_breakdown:
        d = resp.latency_breakdown.to_dict()
        lat_out = LatencyOut(**d)

    return QueryResponse(
        answer=resp.answer,
        chunks=[
            ChunkOut(
                chunk_id=c.chunk_id, text=c.text, ticker=c.ticker,
                year=c.year, sec_section=c.sec_section,
                score=c.score, is_table=c.is_table, citation=c.citation,
            )
            for c in resp.chunks
        ],
        model=resp.model,
        latency_s=resp.latency_s,
        latency_breakdown=lat_out,
        financial_highlights=resp.financial_highlights,
        cached=False,
    )


# ── POST /ingest ──────────────────────────────────────────────────────────────

@app.post("/ingest", summary="Ingest a 10-K filing into Qdrant + BM25")
def ingest(
    req:    IngestRequest,
    finrag: FinRAG = Depends(get_finrag),
) -> Dict[str, Any]:
    """
    Download the specified 10-K from EDGAR, parse it, embed all chunks, and
    upsert them into Qdrant (dense) and BM25 (sparse).

    This endpoint runs synchronously in a thread pool.  Expect 2–5 minutes
    for a typical filing depending on document size and CPU speed.
    """
    try:
        count = finrag.ingest(req.ticker.upper(), req.year)
        return {
            "status":          "ok",
            "ticker":          req.ticker.upper(),
            "year":            req.year,
            "chunks_indexed":  count,
        }
    except Exception as exc:
        logger.exception(f"Ingest error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /cache/stats ──────────────────────────────────────────────────────────

@app.get("/cache/stats", summary="Redis semantic cache statistics")
async def cache_stats(
    cache: SemanticCache = Depends(get_cache),
) -> Dict[str, Any]:
    """Return hit-rate, key count, and index metadata for the Redis cache."""
    return await cache.stats()


# ── DELETE /cache ─────────────────────────────────────────────────────────────

@app.delete("/cache", summary="Flush the semantic cache")
async def clear_cache(
    body:  ClearCacheRequest = ClearCacheRequest(),
    cache: SemanticCache     = Depends(get_cache),
) -> Dict[str, Any]:
    """
    Delete cached entries.

    * Supply ``{"ticker": "MSFT"}`` to flush only MSFT entries.
    * Supply an empty body ``{}`` to flush the entire cache.
    """
    deleted = await cache.clear(ticker=body.ticker)
    return {
        "status":  "ok",
        "deleted": deleted,
        "ticker":  body.ticker or "all",
    }


# ── GET /health ───────────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness probe")
async def health(
    finrag: FinRAG        = Depends(get_finrag),
    cache:  SemanticCache = Depends(get_cache),
) -> Dict[str, Any]:
    redis_ok = await cache.ping()
    return {
        "status":          "healthy",
        "ollama_model":    OLLAMA_MODEL,
        "reranker":        RERANKER_MODEL if finrag.reranker else "disabled",
        "redis_available": redis_ok,
        "cache_hit_rate":  round(
            cache._total_hits / max(1, cache._total_requests), 4
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_valid_chunk(d: Dict[str, Any]) -> bool:
    """Guard against malformed cached chunk dicts."""
    return all(k in d for k in ("chunk_id", "text", "ticker", "year",
                                "sec_section", "score", "is_table"))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import uvicorn  # type: ignore

    parser = argparse.ArgumentParser(description="FinRAG v3 API server")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   default=8000, type=int)
    parser.add_argument("--reload", action="store_true", help="Hot-reload (dev only)")
    args = parser.parse_args()

    logger.info(f"Starting FinRAG v3 API on http://{args.host}:{args.port}")
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )

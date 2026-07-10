"""
pipeline.py — FinRAG v3 Core Pipeline
=======================================
All retrieval, fusion, reranking, generation, and evaluation logic.

Architecture
------------
  Query
    → EmbeddingEngine  (BAAI/bge-small-en-v1.5, 384-dim, CPU)
    → [optional HyDE]  (Ollama qwen2.5:3b, default OFF)
    → Dense Search     (Qdrant cosine KNN)
    ↕ RRF Fusion       (Reciprocal Rank Fusion, k=60)
    → Sparse Search    (BM25Okapi)
    → Reranker         (cross-encoder/ms-marco-MiniLM-L-6-v2, 22 M params)
    → Generator        (Ollama qwen2.5:3b — sync + async streaming)

★ v3 changes from v2
--------------------
  Reranker    bge-reranker-base (278 M, 80-200 ms)
           →  ms-marco-MiniLM-L-6-v2 (22 M, 15-40 ms) — 12× faster on CPU
  HyDE        default True  →  default False (opt-in per request)
  Citations   Chunk gains .citation field populated by EntityLinker during
              ingestion, e.g. "[Item 7. MD&A, ~page 47, ¶3]"
  Streaming   Generator.astream() yields tokens via AsyncGenerator using a
              background thread + asyncio.Queue bridge
  Latency     profile_stage() context manager throughout; cache_ms + embed_ms
              tracked externally (in api.py) and merged into LatencyBreakdown
  Precomputed If the API layer already embedded the query for cache lookup,
  vector      pass it as precomputed_vec to avoid double-embedding
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import pickle
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import (
    Any, AsyncGenerator, Dict, List, Optional, Set, Tuple,
)

# ── Logging ───────────────────────────────────────────────────────────────────
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("finrag")          # type: ignore[assignment]
    logger.setLevel(_logging.INFO)
    _logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s")

# ── Ollama ────────────────────────────────────────────────────────────────────
try:
    import ollama as ollama_client                  # type: ignore
except ImportError:
    ollama_client = None                           # type: ignore[assignment]

# ── Qdrant ────────────────────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, FieldCondition, Filter,
        MatchValue, PointStruct, VectorParams,
    )
except ImportError as _exc:
    raise SystemExit(f"qdrant-client not installed: {_exc}")

# ── sentence-transformers ─────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError as _exc:
    raise SystemExit(f"sentence-transformers not installed: {_exc}")

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None                            # type: ignore[assignment,misc]
    _CROSS_ENCODER_AVAILABLE = False

# ── BM25 ──────────────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi               # type: ignore
except ImportError:
    BM25Okapi = None                              # type: ignore[assignment,misc]

# ── Optional evaluation deps ─────────────────────────────────────────────────
try:
    from rouge_score import rouge_scorer as _rouge_lib   # type: ignore
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from bert_score import score as _bert_score_fn       # type: ignore
    _BERT_SCORE_AVAILABLE = True
except ImportError:
    _BERT_SCORE_AVAILABLE = False

try:
    import nltk                                          # type: ignore
    for _tok in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{_tok}")
        except LookupError:
            nltk.download(_tok, quiet=True)
    from nltk.translate.bleu_score import (             # type: ignore
        sentence_bleu, SmoothingFunction as _SmoothingFn,
    )
    from nltk.tokenize import word_tokenize             # type: ignore
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# try:
#     from datasets import Dataset                        # type: ignore
#     from ragas import evaluate as _ragas_evaluate       # type: ignore
#     from ragas.metrics import answer_relevancy, faithfulness  # type: ignore
#     try:
#         from ragas.metrics import context_recall, context_precision  # type: ignore
#         _RAGAS_CONTEXT_AVAILABLE = True
#     except ImportError:
#         _RAGAS_CONTEXT_AVAILABLE = False
#     _RAGAS_AVAILABLE = True
# except ImportError:
#     _RAGAS_AVAILABLE = _RAGAS_CONTEXT_AVAILABLE = False
try:
    from datasets import Dataset

    print("✓ datasets")

    from ragas import evaluate as _ragas_evaluate
    print("✓ ragas.evaluate")

    from ragas.metrics import answer_relevancy
    print("✓ answer_relevancy")

    from ragas.metrics import faithfulness
    print("✓ faithfulness")

    try:
        from ragas.metrics import context_recall
        from ragas.metrics import context_precision

        print("✓ context metrics")
        _RAGAS_CONTEXT_AVAILABLE = True

    except Exception as e:
        print(f"CONTEXT METRICS ERROR: {e}")
        _RAGAS_CONTEXT_AVAILABLE = False

    _RAGAS_AVAILABLE = True

except Exception as e:
    import traceback

    print("\n========== RAGAS IMPORT FAILURE ==========")
    traceback.print_exc()
    print("==========================================\n")

    _RAGAS_AVAILABLE = False
    _RAGAS_CONTEXT_AVAILABLE = False

# ── Local utilities ───────────────────────────────────────────────────────────
from utils import (
    EntityLinker,
    LatencyBreakdown,
    NumericUtils,
    extract_financial_highlights,
    profile_stage,
)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

QDRANT_PATH      = "./qdrant_db"
BM25_INDEX_PATH  = "./bm25_index.pkl"
COLLECTION_NAME  = "finrag_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM        = 384
OLLAMA_MODEL     = "qwen2.5:3b"
CHUNK_SIZE       = 512
CHUNK_OVERLAP    = 64
TOP_K            = 6
RERANK_CANDIDATES = TOP_K * 3    # 18 candidates → reranker → 6 final

# ★ v3 Reranker model
#
# ┌────────────────────────────────────┬────────┬─────────────┬──────────────────────┐
# │ Model                              │ Params │ Latency @18 │ v3 status            │
# ├────────────────────────────────────┼────────┼─────────────┼──────────────────────┤
# │ BAAI/bge-reranker-base       (v2)  │ 278 M  │ 80–200 ms   │ Replaced             │
# │ ms-marco-MiniLM-L-6-v2  ◄ CHOSEN  │  22 M  │  15– 40 ms  │ 12× faster on CPU    │
# │ ms-marco-MiniLM-L-12-v2           │  33 M  │  25– 60 ms  │ Slightly better, ↑ms │
# └────────────────────────────────────┴────────┴─────────────┴──────────────────────┘
# Rationale: v3 targets <5 s end-to-end latency on CPU.  Saving ~150 ms on the
# reranker stage is critical budget for cache + embed + generation.

RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BERT_SCORE_MODEL = "distilbert-base-uncased"

SEC_SECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"item\s*1a[.\s]+risk\s+factor",                "Item 1A. Risk Factors"),
    (r"item\s*1b[.\s]+unresolved\s+staff\s+comment", "Item 1B. Unresolved Staff Comments"),
    (r"item\s*1[.\s]+business",                      "Item 1. Business"),
    (r"item\s*2[.\s]+propert",                       "Item 2. Properties"),
    (r"item\s*3[.\s]+legal\s+proceed",               "Item 3. Legal Proceedings"),
    (r"item\s*4[.\s]+mine\s+safety",                 "Item 4. Mine Safety Disclosures"),
    (r"item\s*5[.\s]+market",                        "Item 5. Market for Registrant"),
    (r"item\s*6[.\s]+selected\s+financial",          "Item 6. Selected Financial Data"),
    (r"item\s*7a[.\s]+quantitative",                 "Item 7A. Quantitative Disclosures"),
    (r"item\s*7[.\s]+management",                    "Item 7. MD&A"),
    (r"item\s*8[.\s]+financial\s+statement",         "Item 8. Financial Statements"),
    (r"item\s*9a[.\s]+controls",                     "Item 9A. Controls and Procedures"),
    (r"item\s*9b[.\s]+other",                        "Item 9B. Other Information"),
    (r"item\s*9[.\s]+changes",                       "Item 9. Changes in Accountants"),
    (r"item\s*10[.\s]+director",                     "Item 10. Directors"),
    (r"item\s*11[.\s]+executive\s+comp",             "Item 11. Executive Compensation"),
    (r"item\s*12[.\s]+security\s+ownership",         "Item 12. Security Ownership"),
    (r"item\s*13[.\s]+certain\s+relation",           "Item 13. Certain Relationships"),
    (r"item\s*14[.\s]+principal\s+account",          "Item 14. Principal Accountant"),
    (r"item\s*15[.\s]+exhibit",                      "Item 15. Exhibits"),
]

SYSTEM_PROMPT = (
    "You are a precise financial analyst assistant. Answer questions ONLY from "
    "the provided SEC 10-K context chunks.\n\n"
    "Rules:\n"
    "1. Every factual claim must be followed by its exact citation tag, e.g. "
    "[Item 7. MD&A, ~page 47, ¶3].\n"
    "2. If context is insufficient, respond: "
    "\"INSUFFICIENT_CONTEXT: The retrieved passages do not contain the needed information.\"\n"
    "3. Never fabricate numbers, dates, or company details.\n"
    "4. Be concise; use bullet points for multi-part answers."
)


# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """
    One text chunk produced during ingestion.

    ★ v3 additions
    --------------
    page_number     : Estimated 1-based page number (EntityLinker).
    paragraph_index : Paragraph counter within the current section.
    citation        : Human-readable citation string, e.g.
                      ``"[Item 7. MD&A, ~page 47, ¶3]"``.
    """
    chunk_id:        str
    text:            str
    ticker:          str
    year:            int
    sec_section:     str
    chunk_index:     int
    is_table:        bool = False
    word_count:      int  = 0
    # ★ v3 citation metadata
    page_number:     int  = 0
    paragraph_index: int  = 0
    citation:        str  = ""

    def to_payload(self) -> Dict[str, Any]:
        """Return a flat dict suitable for Qdrant point payload."""
        return {
            "chunk_id":        self.chunk_id,
            "text":            self.text,
            "ticker":          self.ticker,
            "year":            self.year,
            "sec_section":     self.sec_section,
            "chunk_index":     self.chunk_index,
            "is_table":        self.is_table,
            "word_count":      self.word_count,
            "page_number":     self.page_number,
            "paragraph_index": self.paragraph_index,
            "citation":        self.citation,
        }


@dataclass
class QueryResult:
    """Single retrieved chunk with its retrieval / reranking score."""
    chunk_id:    str
    text:        str
    ticker:      str
    year:        int
    sec_section: str
    score:       float
    is_table:    bool
    citation:    str = ""   # ★ v3: e.g. "[Item 1A. Risk Factors, ~page 47, ¶3]"


@dataclass
class RAGResponse:
    """Full response returned by ``FinRAG.query()``."""
    answer:            str
    chunks:            List[QueryResult]
    model:             str
    latency_s:         float
    latency_breakdown: Optional[LatencyBreakdown] = None
    financial_highlights: Optional[Dict[str, Any]] = None  # ★ v3


@dataclass
class EvalSample:
    """One question-answer pair used by FinRAGEvaluator."""
    question:         str
    ground_truth:     str
    generated_answer: str
    contexts:         List[str]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data Ingestion — SECIngester
# ══════════════════════════════════════════════════════════════════════════════

class SECIngester:
    """
    Downloads and parses SEC 10-K filings.

    ★ v3: EntityLinker is wired in during ``_section_aware_chunk()`` to tag
    every Chunk with ``page_number``, ``paragraph_index``, and ``citation``.
    """

    def __init__(self) -> None:
        self._check_deps()

    @staticmethod
    def _check_deps() -> None:
        try:
            import edgar            # noqa: F401
        except ImportError:
            raise SystemExit("edgartools not installed. Run: pip install edgartools")
        try:
            from unstructured.partition.html import partition_html  # noqa: F401
        except ImportError:
            raise SystemExit("unstructured[html] not installed.")

    # ── public API ────────────────────────────────────────────────────────────

    def download_and_parse(self, ticker: str, year: int) -> List[Chunk]:
        """Download a 10-K from EDGAR, parse it, and return Chunk objects."""
        logger.info(f"Ingesting {ticker} 10-K for fiscal year {year} …")
        html      = self._fetch_html(ticker, year)
        elements  = self._parse_html(html)
        chunks    = self._section_aware_chunk(elements, ticker, year)
        logger.success(f"Produced {len(chunks)} chunks for {ticker}/{year}")
        return chunks

    # ── private helpers ───────────────────────────────────────────────────────

    def _fetch_html(self, ticker: str, year: int) -> str:
        import edgar
        edgar.set_identity("FinRAG research@finrag.local")
        os.environ.setdefault("EDGAR_USER_AGENT", "FinRAG research@finrag.local")

        company  = edgar.Company(ticker)
        filings  = company.get_filings(form="10-K")

        target = None
        for f in filings:
            fy = getattr(f.filing_date, "year", None) or int(str(f.filing_date)[:4])
            if fy in (year, year + 1):
                target = f
                break
        if target is None:
            logger.warning("No exact match; using most recent 10-K.")
            target = filings[0]

        logger.info(f"Filing: {target.accession_no}  filed {target.filing_date}")
        doc  = target.document
        html_doc = target.primary_document if doc is None else doc
        if html_doc is None:
            raise ValueError(f"Cannot retrieve HTML for {ticker}/{year}")

        html_content = (
            html_doc.html()        if hasattr(html_doc, "html")
            else html_doc.content  if hasattr(html_doc, "content")
            else str(html_doc)
        )
        if not html_content or len(html_content) < 1_000:
            raise ValueError(f"HTML too short ({len(html_content)} chars)")
        logger.info(f"Downloaded {len(html_content):,} chars")
        return html_content

    def _parse_html(self, html: str) -> List[Dict[str, Any]]:
        from unstructured.partition.html import partition_html
        elements = partition_html(text=html)
        out = []
        for el in elements:
            text = (el.text or "").strip()
            if text:
                out.append({
                    "text":     text,
                    "is_table": type(el).__name__ == "Table",
                })
        logger.debug(f"Parsed {len(out)} raw elements")
        return out

    def _section_aware_chunk(
        self,
        elements: List[Dict[str, Any]],
        ticker:   str,
        year:     int,
    ) -> List[Chunk]:
        """
        Split elements into fixed-size Chunks and tag each with citation
        metadata via EntityLinker.

        ★ v3: EntityLinker tracks position so every Chunk knows its
        approximate page, paragraph, and section.
        """
        chunks:          List[Chunk] = []
        current_section: str         = "Preamble"
        chunk_index:     int         = 0
        linker                       = EntityLinker()   # ★ v3

        for el in elements:
            text       = el["text"]
            is_table   = el["is_table"]
            word_count = len(text.split())

            # ── section-heading detection ─────────────────────────────────
            detected = linker.detect_section(text)
            if detected:
                current_section = detected
                linker.update_section(detected)
                # Advance, then skip very-short headings (they're not chunks)
                linker.advance(word_count, is_new_section=True)
                if word_count < 15:
                    continue

            # ── build citation at current position (before advancing) ──────
            citation = linker.build_citation()

            if is_table:
                chunk_id = self._make_id(ticker, year, chunk_index)
                chunks.append(Chunk(
                    chunk_id=chunk_id, text=text, ticker=ticker, year=year,
                    sec_section=current_section, chunk_index=chunk_index,
                    is_table=True, word_count=word_count,
                    page_number=linker.estimated_page,
                    paragraph_index=linker.paragraph_index,
                    citation=citation,
                ))
                chunk_index += 1
            else:
                words = text.split()
                start = 0
                while start < len(words):
                    end       = min(start + CHUNK_SIZE, len(words))
                    chunk_txt = " ".join(words[start:end])
                    if len(chunk_txt.strip()) >= 30:
                        chunk_id = self._make_id(ticker, year, chunk_index)
                        chunks.append(Chunk(
                            chunk_id=chunk_id, text=chunk_txt, ticker=ticker,
                            year=year, sec_section=current_section,
                            chunk_index=chunk_index, is_table=False,
                            word_count=len(chunk_txt.split()),
                            page_number=linker.estimated_page,
                            paragraph_index=linker.paragraph_index,
                            citation=citation,   # all sub-chunks share parent citation
                        ))
                        chunk_index += 1
                    start += CHUNK_SIZE - CHUNK_OVERLAP

            # Advance linker after processing (skip if already advanced above)
            if not detected:
                linker.advance(word_count)

        return chunks

    @staticmethod
    def _make_id(ticker: str, year: int, index: int) -> str:
        return hashlib.md5(f"{ticker}_{year}_{index}".encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Embeddings — EmbeddingEngine
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Wraps BAAI/bge-small-en-v1.5 via sentence-transformers (CPU only).

    Vectors are L2-normalised; cosine similarity == dot product.
    """

    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        logger.info(f"Loading embedding model: {model_name} (CPU) …")
        self._model = SentenceTransformer(model_name, device="cpu")
        logger.success("Embedding model loaded.")

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Batch embed a list of texts."""
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 200,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query with the BGE query prefix."""
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed([prefixed])[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Vector Store — VectorStore (Qdrant)
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """Persistent Qdrant collection.  Never wipes existing data on restart."""

    def __init__(
        self,
        path:       str = QDRANT_PATH,
        collection: str = COLLECTION_NAME,
        embed_dim:  int = EMBED_DIM,
    ) -> None:
        self._collection = collection
        self._client     = QdrantClient(path=path)
        self._ensure_collection(embed_dim)

    def _ensure_collection(self, dim: int) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{self._collection}'")
        else:
            count = self._client.count(self._collection).count
            logger.info(f"Reusing '{self._collection}' ({count:,} vectors)")

    def upsert_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        points = [
            PointStruct(
                id=int(c.chunk_id, 16),
                vector=v,
                payload=c.to_payload(),
            )
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 256):
            self._client.upsert(self._collection, points=points[i : i + 256])
        logger.success(f"Upserted {len(points)} vectors.")

    def search(
        self,
        query_vec: List[float],
        top_k:     int           = TOP_K,
        ticker:    Optional[str] = None,
        year:      Optional[int] = None,
    ) -> List[QueryResult]:
        flt = self._build_filter(ticker, year)
        resp = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=top_k,
            query_filter=flt,
            with_payload=True,
        )
        return [self._to_result(h) for h in resp.points]

    @staticmethod
    def _build_filter(ticker: Optional[str], year: Optional[int]) -> Optional[Filter]:
        conds = []
        if ticker:
            conds.append(FieldCondition(key="ticker", match=MatchValue(value=ticker.upper())))
        if year:
            conds.append(FieldCondition(key="year",   match=MatchValue(value=int(year))))
        return Filter(must=conds) if conds else None

    @staticmethod
    def _to_result(hit: Any) -> QueryResult:
        p = hit.payload
        return QueryResult(
            chunk_id=p.get("chunk_id", ""),
            text=p.get("text", ""),
            ticker=p.get("ticker", ""),
            year=p.get("year", 0),
            sec_section=p.get("sec_section", ""),
            score=hit.score,
            is_table=p.get("is_table", False),
            citation=p.get("citation", ""),          # ★ v3
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. BM25 Sparse Index
# ══════════════════════════════════════════════════════════════════════════════

class BM25Index:
    """
    Append-safe persistent BM25 index backed by a pickle file.

    Never overwrites documents already indexed; only adds new chunk_ids.
    """

    def __init__(self, path: str = BM25_INDEX_PATH) -> None:
        self._path  = path
        self._docs: List[str]      = []
        self._meta: List[Dict]     = []
        self._index: Optional[Any] = None
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            self._docs = data.get("docs", [])
            self._meta = data.get("meta", [])
            self._rebuild()
            logger.info(f"BM25: loaded {len(self._docs):,} documents")
        else:
            logger.info("BM25: fresh index")

    def _save(self) -> None:
        with open(self._path, "wb") as f:
            pickle.dump({"docs": self._docs, "meta": self._meta}, f)

    def _rebuild(self) -> None:
        if self._docs and BM25Okapi is not None:
            self._index = BM25Okapi([d.lower().split() for d in self._docs])

    # ── write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if BM25Okapi is None:
            logger.warning("rank_bm25 not installed — BM25 disabled.")
            return
        existing = {m["chunk_id"] for m in self._meta}
        new      = [c for c in chunks if c.chunk_id not in existing]
        if not new:
            logger.info("BM25: no new chunks.")
            return
        for c in new:
            self._docs.append(c.text)
            self._meta.append({
                "chunk_id":    c.chunk_id,
                "ticker":      c.ticker,
                "year":        c.year,
                "sec_section": c.sec_section,
                "is_table":    c.is_table,
                "citation":    c.citation,     # ★ v3
                "text":        c.text,
            })
        self._rebuild()
        self._save()
        logger.success(f"BM25: added {len(new)} docs (total {len(self._docs):,})")

    # ── read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query:  str,
        top_k:  int           = TOP_K,
        ticker: Optional[str] = None,
        year:   Optional[int] = None,
    ) -> List[QueryResult]:
        if self._index is None or not self._docs:
            return []
        scores = self._index.get_scores(query.lower().split())
        filtered = [
            (i, scores[i]) for i, m in enumerate(self._meta)
            if (not ticker or m["ticker"].upper() == ticker.upper())
            and (not year   or m["year"]           == int(year))
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in filtered[:top_k]:
            m = self._meta[i]
            results.append(QueryResult(
                chunk_id=m["chunk_id"], text=m["text"],
                ticker=m["ticker"],    year=m["year"],
                sec_section=m["sec_section"], score=float(score),
                is_table=m["is_table"],
                citation=m.get("citation", ""),     # ★ v3
            ))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. Reranker  ★ v3: ms-marco-MiniLM-L-6-v2 (22 M, 15–40 ms CPU)
# ══════════════════════════════════════════════════════════════════════════════

class Reranker:
    """
    Cross-encoder reranker.

    ★ v3 model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
    * 22 M parameters   (was 278 M for bge-reranker-base in v2)
    * 15–40 ms for 18 candidates on CPU  (was 80–200 ms)
    * 12× faster, small quality trade-off acceptable for <5 s budget
    * Trained on 100 M (query, passage) pairs from MS-MARCO

    The cross-encoder reads the full (query, chunk) pair up to 512 tokens,
    catching relevance signals that the bi-encoder misses.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        if not _CROSS_ENCODER_AVAILABLE or CrossEncoder is None:
            raise ImportError(
                "sentence-transformers CrossEncoder not available. "
                "Ensure sentence-transformers >= 2.x."
            )
        logger.info(f"Loading reranker: {model_name} (CPU) …")
        self._model      = CrossEncoder(model_name, device="cpu", max_length=512)
        self._model_name = model_name
        logger.success(f"Reranker loaded: {model_name}")

    def rerank(
        self,
        query:      str,
        candidates: List[QueryResult],
        top_n:      int = TOP_K,
    ) -> List[QueryResult]:
        """
        Score all (query, chunk) pairs and return the top *top_n* chunks.

        The ``.score`` attribute of each returned ``QueryResult`` is overwritten
        with the cross-encoder logit (higher = more relevant).
        """
        if not candidates:
            return []

        t0     = time.perf_counter()
        pairs  = [(query, c.text[:512]) for c in candidates]
        scores: List[float] = self._model.predict(
            pairs, batch_size=16, show_progress_bar=False
        ).tolist()

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        top3 = [f"{s:.3f}" for s, _ in ranked[:3]]
        sel  = [c.chunk_id for _, c in ranked[:top_n]]
        logger.info(
            f"Reranker | {len(candidates)}→{top_n} | "
            f"{elapsed_ms:.0f}ms | top3={top3} | selected={sel}"
        )

        results = []
        for rerank_score, chunk in ranked[:top_n]:
            chunk.score = float(rerank_score)
            results.append(chunk)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. Retriever  ★ v3: HyDE default=False, precomputed_vec, profile_stage
# ══════════════════════════════════════════════════════════════════════════════

class Retriever:
    """
    Hybrid retrieval pipeline.

    Stage 1 (optional)  HyDE — generate a hypothetical answer and embed it.
    Stage 2             Dense — Qdrant KNN on query/HyDE vector.
    Stage 3             Sparse — BM25 keyword search.
    Stage 4             RRF — Reciprocal Rank Fusion (k=60).
    Stage 5 (optional)  Cross-Encoder Reranker — MiniLM-L-6-v2.

    ★ v3 changes
    ------------
    * ``use_hyde`` default flipped to ``False`` (opt-in per request).
    * ``precomputed_vec`` optional parameter: if the API already embedded the
      query for the Redis cache lookup, pass it here to skip re-embedding.
    * Per-stage timing via ``profile_stage()`` context manager.
    * Returns ``(chunks, LatencyBreakdown)`` tuple.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embed_engine: EmbeddingEngine,
        bm25_index:   BM25Index,
        reranker:     Optional[Reranker] = None,
    ) -> None:
        self._vs       = vector_store
        self._embed    = embed_engine
        self._bm25     = bm25_index
        self._reranker = reranker

    # ── HyDE ──────────────────────────────────────────────────────────────────

    def _generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical answer using Ollama for HyDE retrieval."""
        if ollama_client is None:
            return query
        prompt = (
            "You are a financial analyst. Write a concise, factual 3-sentence "
            "paragraph that would answer the following question about a company's "
            "SEC 10-K filing. Do not hedge; write as if you know the answer.\n\n"
            f"Question: {query}\n\nHypothetical Answer:"
        )
        try:
            resp = ollama_client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 200},
            )
            hypo = resp["message"]["content"].strip()
            logger.debug(f"HyDE: {hypo[:100]} …")
            return hypo
        except Exception as exc:
            logger.warning(f"HyDE failed ({exc}); using raw query.")
            return query

    # ── RRF ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf(
        dense:  List[QueryResult],
        sparse: List[QueryResult],
        top_k:  int = TOP_K,
        k:      int = 60,
    ) -> List[QueryResult]:
        """Reciprocal Rank Fusion of two ranked lists."""
        scores: Dict[str, float]       = {}
        by_id:  Dict[str, QueryResult] = {}

        for rank, hit in enumerate(dense):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            by_id[hit.chunk_id]  = hit
        for rank, hit in enumerate(sparse):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if hit.chunk_id not in by_id:
                by_id[hit.chunk_id] = hit

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result = []
        for chunk_id, score in ranked[:top_k]:
            r       = by_id[chunk_id]
            r.score = score
            result.append(r)
        return result

    # ── main entry ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        top_k:           int                 = TOP_K,
        ticker:          Optional[str]       = None,
        year:            Optional[int]       = None,
        use_hyde:        bool                = False,    # ★ v3: default OFF
        use_reranker:    bool                = True,
        precomputed_vec: Optional[List[float]] = None,  # ★ v3: avoid re-embed
    ) -> Tuple[List[QueryResult], LatencyBreakdown]:
        """
        Run the full hybrid retrieval pipeline.

        Args:
            query:           Raw query string (always required).
            top_k:           Number of final chunks to return.
            ticker:          Filter results to this ticker symbol.
            year:            Filter results to this fiscal year.
            use_hyde:        Generate a hypothetical answer and embed it
                             instead of the raw query (default **False**).
            use_reranker:    Apply the cross-encoder reranker (default True).
            precomputed_vec: Pre-computed query embedding — avoids double
                             embedding when the API layer already ran
                             embed_query() for the Redis cache check.

        Returns:
            ``(chunks, LatencyBreakdown)`` — LatencyBreakdown has hyde_ms,
            dense_ms, sparse_ms, fusion_ms, and rerank_ms populated.
        """
        lat      = LatencyBreakdown()
        n_cands  = RERANK_CANDIDATES if (use_reranker and self._reranker) else top_k * 2

        # ── Stage 1: HyDE (optional) ──────────────────────────────────────
        with profile_stage(lat, "hyde"):
            if use_hyde:
                search_text = self._generate_hypothetical_answer(query)
                search_vec  = self._embed.embed_query(search_text)
            else:
                search_text = query
                search_vec  = precomputed_vec or self._embed.embed_query(query)
            # If hyde is on, precomputed_vec (for cache) ≠ search_vec (for retrieval)

        if not use_hyde:
            lat.hyde_ms = 0.0  # report 0 if disabled

        # ── Stage 2: Dense retrieval ──────────────────────────────────────
        with profile_stage(lat, "dense"):
            dense_hits = self._vs.search(search_vec, top_k=n_cands, ticker=ticker, year=year)

        # ── Stage 3: Sparse retrieval ─────────────────────────────────────
        with profile_stage(lat, "sparse"):
            sparse_hits = self._bm25.search(query, top_k=n_cands, ticker=ticker, year=year)

        # ── Stage 4: RRF fusion ───────────────────────────────────────────
        with profile_stage(lat, "fusion"):
            n_rrf    = RERANK_CANDIDATES if (use_reranker and self._reranker) else top_k
            fused    = self._rrf(dense_hits, sparse_hits, top_k=n_rrf)

        logger.info(
            f"Retrieval | dense={len(dense_hits)} sparse={len(sparse_hits)} "
            f"fused={len(fused)} [hyde={use_hyde}]"
        )

        # ── Stage 5: Cross-encoder reranking ──────────────────────────────
        with profile_stage(lat, "rerank"):
            if use_reranker and self._reranker is not None:
                final = self._reranker.rerank(query, fused, top_n=top_k)
            else:
                final = fused[:top_k]
                lat.rerank_ms = 0.0

        lat.log(logger)
        return final, lat


# ══════════════════════════════════════════════════════════════════════════════
# 7. Generator  ★ v3: astream() for async token streaming
# ══════════════════════════════════════════════════════════════════════════════

class Generator:
    """
    Wraps the local Ollama instance for LLM generation.

    Methods
    -------
    generate(query, chunks)
        Synchronous; returns the full answer string.
    astream(query, chunks)
        Async generator; yields individual tokens.  Runs Ollama's synchronous
        streaming API in a background thread and bridges tokens to an
        ``asyncio.Queue`` for non-blocking consumption.
    """

    def __init__(self, model: str = OLLAMA_MODEL) -> None:
        self._model = model
        if ollama_client is None:
            logger.warning("ollama not installed — generation will return placeholder.")
        else:
            self._verify_model()

    def _verify_model(self) -> None:
        try:
            names = [m.model for m in ollama_client.list().models]
            if not any(self._model.split(":")[0] in n for n in names):
                logger.warning(f"Model '{self._model}' not in Ollama. Run: ollama pull {self._model}")
        except Exception as exc:
            logger.warning(f"Cannot contact Ollama: {exc}")

    # ── shared context builder ────────────────────────────────────────────────

    def _build_messages(self, query: str, chunks: List[QueryResult]) -> List[Dict[str, str]]:
        context_parts = []
        for c in chunks:
            header = (
                f"[chunk_id: {c.chunk_id}] "
                f"{c.citation}  "               # ★ v3: human-readable citation
                f"[{c.ticker} {c.year} | {c.sec_section}]"
                f"{' [TABLE]' if c.is_table else ''}"
            )
            context_parts.append(f"{header}\n{c.text}")

        user_content = (
            "Context:\n"
            + "\n\n---\n\n".join(context_parts)
            + f"\n\n---\n\nQuestion: {query}\n\nAnswer (cite each claim):"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

    # ── synchronous ──────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: List[QueryResult]) -> str:
        """Synchronous generation — returns complete answer string."""
        if ollama_client is None:
            return (
                "[Ollama unavailable] Context:\n"
                + "\n---\n".join(f"{c.citation} {c.text[:200]}" for c in chunks)
            )
        messages = self._build_messages(query, chunks)
        try:
            resp = ollama_client.chat(
                model=self._model,
                messages=messages,
                options={"temperature": 0.1, "num_predict": 1024},
            )
            return resp["message"]["content"].strip()
        except Exception as exc:
            logger.error(f"Generation error: {exc}")
            return f"GENERATION_ERROR: {exc}"

    # ── async streaming ★ v3 ─────────────────────────────────────────────────

    async def astream(
        self,
        query:  str,
        chunks: List[QueryResult],
    ) -> AsyncGenerator[str, None]:
        """
        Async token generator backed by Ollama's synchronous streaming API.

        Implementation
        --------------
        Ollama's ``chat(..., stream=True)`` is synchronous.  We run it in a
        daemon background thread and bridge each token to an ``asyncio.Queue``
        using ``loop.call_soon_threadsafe()``.  The async generator then
        yields tokens from the queue without blocking the event loop.

        Yields
        ------
        Individual token strings from the model.  Yields ``None``-sentinel
        internally (not exposed to callers) to signal completion.

        Usage::

            async for token in generator.astream(query, chunks):
                yield f"data: {json.dumps({'token': token})}\\n\\n"
        """
        if ollama_client is None:
            yield "[Ollama unavailable — install ollama and run: ollama pull qwen2.5:3b]"
            return

        messages = self._build_messages(query, chunks)
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def _sync_stream() -> None:
            """Run in a background daemon thread."""
            try:
                for response in ollama_client.chat(
                    model=self._model,
                    messages=messages,
                    stream=True,
                    options={"temperature": 0.1, "num_predict": 1024},
                ):
                    token: str = response["message"]["content"]
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                logger.error(f"Streaming error: {exc}")
                loop.call_soon_threadsafe(queue.put_nowait, f"[STREAM_ERROR: {exc}]")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        # Launch background thread — daemon so it dies with the process
        threading.Thread(target=_sync_stream, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:  # sentinel → done
                break
            yield token


# ══════════════════════════════════════════════════════════════════════════════
# 8. Evaluation Suite (scaffolding; extend with real QA pairs)
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalEvaluator:
    """
    Standard IR metrics for the retrieval stage.

    Metrics
    -------
    Recall@K     |rel ∩ top-K| / |rel|           — did we find all needed chunks?
    Precision@K  |rel ∩ top-K| / K               — how much noise in context?
    MRR          1 / rank_of_first_relevant        — rank of first useful chunk
    Hit Rate@K   1 if any rel ∈ top-K             — minimum viability bar
    nDCG@K       DCG / IDCG (binary relevance)    — best single retrieval score
    """

    @staticmethod
    def recall_at_k(retrieved: List[QueryResult], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        top_ids = {r.chunk_id for r in retrieved[:k]}
        return len(relevant_ids & top_ids) / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved: List[QueryResult], relevant_ids: Set[str], k: int) -> float:
        if not retrieved or k == 0:
            return 0.0
        return sum(1 for r in retrieved[:k] if r.chunk_id in relevant_ids) / k

    @staticmethod
    def mrr(retrieved: List[QueryResult], relevant_ids: Set[str]) -> float:
        for i, r in enumerate(retrieved, 1):
            if r.chunk_id in relevant_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def hit_rate(retrieved: List[QueryResult], relevant_ids: Set[str], k: int) -> float:
        return 1.0 if {r.chunk_id for r in retrieved[:k]} & relevant_ids else 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[QueryResult], relevant_ids: Set[str], k: int) -> float:
        dcg  = sum(
            1.0 / math.log2(i + 1)
            for i, r in enumerate(retrieved[:k], 1)
            if r.chunk_id in relevant_ids
        )
        ideal = min(len(relevant_ids), k)
        idcg  = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
        return dcg / idcg if idcg > 0.0 else 0.0

    def compute_all(
        self,
        retrieved:    List[QueryResult],
        relevant_ids: Set[str],
        k:            int = TOP_K,
    ) -> Dict[str, float]:
        return {
            f"recall@{k}":    self.recall_at_k(retrieved, relevant_ids, k),
            f"precision@{k}": self.precision_at_k(retrieved, relevant_ids, k),
            "mrr":            self.mrr(retrieved, relevant_ids),
            f"hit_rate@{k}":  self.hit_rate(retrieved, relevant_ids, k),
            f"ndcg@{k}":      self.ndcg_at_k(retrieved, relevant_ids, k),
        }

    @staticmethod
    def build_relevant_ids(
        chunks: List[QueryResult],
        ground_truth: str,
        token_overlap_threshold: float = 0.30,
    ) -> Set[str]:
        """
        Heuristic fallback when explicit relevant_chunk_ids are not supplied.

        Levels
        ------
        1. Exact substring match (case-insensitive).
        2. Numeric equivalence via NumericUtils.match().
        3. Token-overlap Jaccard ≥ threshold.
        """
        relevant: Set[str] = set()
        if not ground_truth:
            return relevant

        gt_lower  = ground_truth.lower().strip()
        gt_tokens = set(re.sub(r"[^\w\s]", "", gt_lower).split())

        for chunk in chunks:
            cl = chunk.text.lower()
            # Level 1
            if gt_lower in cl:
                relevant.add(chunk.chunk_id)
                continue
            # Level 2
            if NumericUtils.match(chunk.text, ground_truth):
                relevant.add(chunk.chunk_id)
                continue
            # Level 3
            ct = set(re.sub(r"[^\w\s]", "", cl).split())
            if gt_tokens and len(gt_tokens & ct) / len(gt_tokens) >= token_overlap_threshold:
                relevant.add(chunk.chunk_id)

        return relevant


class GenerationEvaluator:
    """
    Answer-quality metrics for the generation stage.

    Metrics (10 total across retrieval + generation)
    -------
    Semantic Similarity  — BGE cosine similarity (holistic meaning match)
    ROUGE-L              — Longest common subsequence F1
    ROUGE-1/2            — Keyword / key-phrase overlap
    BLEU                 — n-gram precision (baseline; penalises paraphrase)
    BERTScore F1         — Token-level semantic matching via DistilBERT
    Numeric Match        — ±1% numeric equivalence ($245,122M == 245122)
    """

    def __init__(self, embed_engine: EmbeddingEngine) -> None:
        self._embed = embed_engine
        self._rouge_scorer = (
            _rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            if _ROUGE_AVAILABLE else None
        )

    def semantic_similarity(self, pred: str, ref: str) -> float:
        if not pred or not ref:
            return 0.0
        try:
            vecs = self._embed.embed([pred, ref])
            return float(sum(a * b for a, b in zip(vecs[0], vecs[1])))
        except Exception as exc:
            logger.warning(f"Semantic similarity failed: {exc}")
            return float("nan")

    def rouge_scores(self, pred: str, ref: str) -> Dict[str, float]:
        nan3 = {"rouge_1": float("nan"), "rouge_2": float("nan"), "rouge_l": float("nan")}
        if not _ROUGE_AVAILABLE or self._rouge_scorer is None:
            return nan3
        try:
            r = self._rouge_scorer.score(ref, pred)
            return {
                "rouge_1": r["rouge1"].fmeasure,
                "rouge_2": r["rouge2"].fmeasure,
                "rouge_l": r["rougeL"].fmeasure,
            }
        except Exception:
            return nan3

    def bleu_score(self, pred: str, ref: str) -> float:
        if not _NLTK_AVAILABLE:
            return float("nan")
        try:
            rt = word_tokenize(ref.lower())
            pt = word_tokenize(pred.lower())
            return float(sentence_bleu([rt], pt, smoothing_function=_SmoothingFn().method1))
        except Exception:
            return float("nan")

    def bert_score(self, pred: str, ref: str) -> Dict[str, float]:
        empty = {"bert_p": float("nan"), "bert_r": float("nan"), "bert_f1": float("nan")}
        if not _BERT_SCORE_AVAILABLE:
            return empty
        try:
            P, R, F1 = _bert_score_fn(
                [pred], [ref],
                model_type=BERT_SCORE_MODEL,
                lang="en",
                verbose=False,
                device="cpu",
            )
            return {
                "bert_p":  float(P.mean()),
                "bert_r":  float(R.mean()),
                "bert_f1": float(F1.mean()),
            }
        except Exception as exc:
            logger.warning(f"BERTScore failed: {exc}")
            return empty

    def compute_all(self, pred: str, ref: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        metrics["semantic_similarity"] = self.semantic_similarity(pred, ref)
        metrics.update(self.rouge_scores(pred, ref))
        metrics["bleu"]          = self.bleu_score(pred, ref)
        metrics.update(self.bert_score(pred, ref))
        metrics["numeric_match"] = float(NumericUtils.match(pred, ref))
        return metrics


class FinRAGEvaluator:
    """
    Wraps RetrievalEvaluator + GenerationEvaluator + RAGAS into a single
    run_financebench() harness.

    Question dict format::

        {
            "question":           str,          # required
            "answer":             str,          # required (ground truth)
            "relevant_chunk_ids": List[str],    # optional explicit labels
        }
    """

    def __init__(self, finrag: "FinRAG") -> None:
        self._finrag = finrag

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower().strip()))

    def exact_match_accuracy(self, samples: List[EvalSample]) -> float:
        if not samples:
            return 0.0
        hits = sum(1 for s in samples
                   if self._normalise(s.generated_answer) == self._normalise(s.ground_truth))
        return hits / len(samples)

    def ragas_evaluate(self, samples: List[EvalSample]) -> Optional[Dict[str, float]]:
        if not _RAGAS_AVAILABLE:
            logger.warning("ragas not installed — skipping RAGAS metrics.")
            return None
        dataset = Dataset.from_dict({
            "question":     [s.question         for s in samples],
            "answer":       [s.generated_answer  for s in samples],
            "contexts":     [s.contexts          for s in samples],
            "ground_truth": [s.ground_truth      for s in samples],
        })
        metrics = [answer_relevancy, faithfulness]
        if _RAGAS_CONTEXT_AVAILABLE:
            metrics += [context_recall, context_precision]   # type: ignore[name-defined]
        try:
            result = _ragas_evaluate(dataset, metrics=metrics)
            return {k: float(result.get(k, float("nan"))) for k in (
                "answer_relevancy", "faithfulness",
                *( ["context_recall", "context_precision"] if _RAGAS_CONTEXT_AVAILABLE else [] )
            )}
        except Exception as exc:
            logger.error(f"RAGAS failed: {exc}")
            return None

    def run_financebench(
        self,
        questions: List[Dict[str, Any]],
        ticker:    str,
        year:      int,
        k:         int = TOP_K,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation suite on a FinanceBench-style question set.

        Returns a nested report dict with keys:
        ``num_questions``, ``exact_match_accuracy``, ``numeric_exact_match``,
        ``retrieval`` (avg), ``generation`` (avg), ``ragas``, ``latency_summary``.
        """
        ret_ev   = RetrievalEvaluator()
        gen_ev   = GenerationEvaluator(self._finrag.embed_engine)
        samples:   List[EvalSample]       = []
        ret_mets:  List[Dict[str, float]] = []
        gen_mets:  List[Dict[str, float]] = []
        lat_dicts: List[Dict[str, float]] = []

        for idx, q in enumerate(questions):
            question     = q["question"]
            ground_truth = q["answer"]
            explicit_ids = set(q.get("relevant_chunk_ids", []))

            logger.info(f"Eval Q{idx+1}/{len(questions)}: {question[:70]} …")
            resp = self._finrag.query(question, ticker=ticker, year=year)

            rel_ids = explicit_ids or ret_ev.build_relevant_ids(resp.chunks, ground_truth)
            ret_mets.append(ret_ev.compute_all(resp.chunks, rel_ids, k=k))
            gen_mets.append(gen_ev.compute_all(resp.answer, ground_truth))
            if resp.latency_breakdown:
                lat_dicts.append(resp.latency_breakdown.to_dict())
            samples.append(EvalSample(
                question=question, ground_truth=ground_truth,
                generated_answer=resp.answer,
                contexts=[c.text for c in resp.chunks],
            ))

        def _avg(dicts: List[Dict[str, float]]) -> Dict[str, float]:
            if not dicts:
                return {}
            return {k: statistics.mean(d[k] for d in dicts if k in d and not math.isnan(d.get(k, float("nan"))))
                    for k in dicts[0]}

        latency_summary: Dict[str, Any] = {}
        if lat_dicts:
            for key in lat_dicts[0]:
                vals = sorted(d[key] for d in lat_dicts)
                p95i = max(0, int(len(vals) * 0.95) - 1)
                latency_summary[key] = {
                    "median_ms": round(statistics.median(vals), 1),
                    "mean_ms":   round(statistics.mean(vals), 1),
                    "p95_ms":    round(vals[p95i], 1),
                }

        numeric_em = (
            sum(1 for s in samples if NumericUtils.match(s.generated_answer, s.ground_truth))
            / len(samples)
        ) if samples else 0.0

        report = {
            "num_questions":        len(samples),
            "exact_match_accuracy": self.exact_match_accuracy(samples),
            "numeric_exact_match":  numeric_em,
            "retrieval":            _avg(ret_mets),
            "generation":           _avg(gen_mets),
            "ragas":                self.ragas_evaluate(samples),
            "latency_summary":      latency_summary,
        }
        logger.info(f"\nEval report:\n{json.dumps(report, indent=2, default=str)}")
        return report


# ══════════════════════════════════════════════════════════════════════════════
# 9. Orchestrator — FinRAG
# ══════════════════════════════════════════════════════════════════════════════

class FinRAG:
    """
    Top-level orchestrator that wires all components together.

    Quick start::

        finrag = FinRAG()
        finrag.ingest("MSFT", 2024)
        resp = finrag.query("What are the primary risk factors?", ticker="MSFT", year=2024)
        print(resp.answer)
        resp.latency_breakdown.log()

    Args:
        use_reranker:    Enable cross-encoder reranking (default True).
        reranker_model:  Override the default MiniLM-L-6-v2 model.
    """

    def __init__(
        self,
        qdrant_path:    str  = QDRANT_PATH,
        embed_model:    str  = EMBED_MODEL_NAME,
        ollama_model:   str  = OLLAMA_MODEL,
        use_reranker:   bool = True,
        reranker_model: str  = RERANKER_MODEL,
    ) -> None:
        logger.info("Initialising FinRAG v3 …")

        self.embed_engine  = EmbeddingEngine(model_name=embed_model)
        self.vector_store  = VectorStore(path=qdrant_path)
        self.bm25_index    = BM25Index()

        self.reranker: Optional[Reranker] = None
        if use_reranker:
            if _CROSS_ENCODER_AVAILABLE:
                try:
                    self.reranker = Reranker(model_name=reranker_model)
                except Exception as exc:
                    logger.warning(f"Reranker load failed ({exc}); running without reranker.")
            else:
                logger.warning("CrossEncoder not available — reranker disabled.")

        self.retriever = Retriever(
            self.vector_store, self.embed_engine, self.bm25_index,
            reranker=self.reranker,
        )
        self.generator = Generator(model=ollama_model)
        self.ingester  = SECIngester()
        self.evaluator = FinRAGEvaluator(self)

        reranker_status = (
            f"enabled ({self.reranker._model_name})" if self.reranker else "disabled"
        )
        logger.success(f"FinRAG v3 ready | reranker={reranker_status}")

    # ── ingest ────────────────────────────────────────────────────────────────

    def ingest(self, ticker: str, year: int) -> int:
        """Download, parse, embed, and index a 10-K filing."""
        t0     = time.perf_counter()
        chunks = self.ingester.download_and_parse(ticker, year)
        if not chunks:
            logger.warning(f"No chunks for {ticker}/{year}.")
            return 0
        logger.info(f"Embedding {len(chunks)} chunks …")
        vectors = self.embed_engine.embed([c.text for c in chunks])
        self.vector_store.upsert_chunks(chunks, vectors)
        self.bm25_index.add_chunks(chunks)
        logger.success(f"Ingested {len(chunks)} chunks in {time.perf_counter()-t0:.1f}s")
        return len(chunks)

    # ── query (synchronous) ───────────────────────────────────────────────────

    def query(
        self,
        query:           str,
        ticker:          Optional[str]       = None,
        year:            Optional[int]       = None,
        top_k:           int                 = TOP_K,
        use_hyde:        bool                = False,   # ★ v3 default
        use_reranker:    bool                = True,
        precomputed_vec: Optional[List[float]] = None,
    ) -> RAGResponse:
        """
        Synchronous end-to-end RAG query.

        If ``precomputed_vec`` is supplied (e.g. from the API's cache-check
        embedding), the retrieval stage skips re-embedding.
        """
        t0 = time.perf_counter()

        chunks, lat = self.retriever.retrieve(
            query,
            top_k=top_k,
            ticker=ticker,
            year=year,
            use_hyde=use_hyde,
            use_reranker=use_reranker,
            precomputed_vec=precomputed_vec,
        )

        if not chunks:
            return RAGResponse(
                answer=(
                    "INSUFFICIENT_CONTEXT: No relevant passages found "
                    f"for ticker={ticker}, year={year}."
                ),
                chunks=[],
                model=self.generator._model,
                latency_s=time.perf_counter() - t0,
                latency_breakdown=lat,
            )

        t_gen  = time.perf_counter()
        answer = self.generator.generate(query, chunks)
        lat.generation_ms = (time.perf_counter() - t_gen) * 1_000

        highlights = extract_financial_highlights(chunks)

        return RAGResponse(
            answer=answer,
            chunks=chunks,
            model=self.generator._model,
            latency_s=time.perf_counter() - t0,
            latency_breakdown=lat,
            financial_highlights=highlights,
        )


# ── CLI convenience ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FinRAG v3 pipeline smoke-test")
    parser.add_argument("--ticker",    default="MSFT")
    parser.add_argument("--year",      default=2024, type=int)
    parser.add_argument("--no-ingest", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--hyde",      action="store_true")
    parser.add_argument("--eval",      action="store_true")
    args = parser.parse_args()

    rag = FinRAG(use_reranker=not args.no_rerank)
    if not args.no_ingest:
        rag.ingest(args.ticker, args.year)

    sample_q = "What were the total revenues and operating income for the fiscal year?"
    resp = rag.query(sample_q, ticker=args.ticker, year=args.year, use_hyde=args.hyde)
    print(f"\nAnswer:\n{resp.answer}\n")
    if resp.latency_breakdown:
        resp.latency_breakdown.log()

    if resp.financial_highlights:
        print(f"\nFinancial Highlights: {json.dumps(resp.financial_highlights, indent=2, default=str)}")

    if args.eval:
        rag.evaluator.run_financebench(
            questions=[
                {"question": "Total revenue?",        "answer": "245122"},
                {"question": "Primary risk factors?", "answer": "cybersecurity"},
            ],
            ticker=args.ticker,
            year=args.year,
        )

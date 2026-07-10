"""
utils.py — FinRAG v3 Utilities
================================
Four independent, CPU-only, stateless modules.  No ML dependencies — this
file is independently testable with ``pytest`` and imports only stdlib.

Modules
-------
  1. LatencyProfiler  — ``LatencyBreakdown`` dataclass + ``profile_stage``
                        context manager for per-stage timing.
  2. EntityLinker     — Attaches human-readable citation strings to chunks
                        during SEC 10-K ingestion.
  3. FinancialExtractor — Parses SEC financial tables into structured
                        ``{metric: {year: value}}`` dictionaries.
  4. NumericUtils     — Normalises and compares financial number strings.

Usage example::

    from utils import LatencyBreakdown, profile_stage, EntityLinker
    from utils import extract_financial_highlights, NumericUtils

    lat = LatencyBreakdown()
    with profile_stage(lat, "dense"):
        results = vector_store.search(vec)
    lat.log()

    assert NumericUtils.match("$245,122 million", "245122")  # True
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Module 1 — Latency Profiler
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class LatencyBreakdown:
    """
    Per-stage latency in milliseconds for a single end-to-end RAG query.

    ★ v3 additions: ``cache_ms`` (Redis lookup) + ``embed_ms`` (query vector).

    Properties
    ----------
    retrieval_ms
        Sum of hyde + dense + sparse + fusion + rerank.
    total_ms
        Sum of all stages (cache + embed + retrieval + generation).
    """

    cache_ms:      float = 0.0   # Redis semantic cache lookup
    embed_ms:      float = 0.0   # Query embedding (bge-small, ~15 ms CPU)
    hyde_ms:       float = 0.0   # HyDE hypothetical generation (opt-in)
    dense_ms:      float = 0.0   # Qdrant KNN vector search
    sparse_ms:     float = 0.0   # BM25 keyword search
    fusion_ms:     float = 0.0   # Reciprocal Rank Fusion
    rerank_ms:     float = 0.0   # Cross-encoder reranking
    generation_ms: float = 0.0   # Ollama LLM generation

    @property
    def retrieval_ms(self) -> float:
        return self.hyde_ms + self.dense_ms + self.sparse_ms + self.fusion_ms + self.rerank_ms

    @property
    def total_ms(self) -> float:
        return self.cache_ms + self.embed_ms + self.retrieval_ms + self.generation_ms

    def to_dict(self) -> Dict[str, float]:
        return {
            "cache_ms":      self.cache_ms,
            "embed_ms":      self.embed_ms,
            "hyde_ms":       self.hyde_ms,
            "dense_ms":      self.dense_ms,
            "sparse_ms":     self.sparse_ms,
            "fusion_ms":     self.fusion_ms,
            "rerank_ms":     self.rerank_ms,
            "generation_ms": self.generation_ms,
            "retrieval_ms":  self.retrieval_ms,
            "total_ms":      self.total_ms,
        }

    def log(self, logger: Any = None) -> None:
        """Pretty-print each stage to *logger* (loguru) or stdout."""
        msg = (
            f"Latency | Cache={self.cache_ms:.0f}ms  "
            f"Embed={self.embed_ms:.0f}ms  "
            f"HyDE={self.hyde_ms:.0f}ms  "
            f"Dense={self.dense_ms:.0f}ms  "
            f"Sparse={self.sparse_ms:.0f}ms  "
            f"RRF={self.fusion_ms:.0f}ms  "
            f"Rerank={self.rerank_ms:.0f}ms  "
            f"Gen={self.generation_ms:.0f}ms  "
            f"→ Total={self.total_ms:.0f}ms"
        )
        (logger.info if logger else print)(msg)


@contextlib.contextmanager
def profile_stage(
    breakdown: LatencyBreakdown,
    stage: str,
) -> Generator[None, None, None]:
    """
    Context manager that measures elapsed milliseconds and writes the result
    to ``breakdown.<stage>_ms``.

    Args:
        breakdown: ``LatencyBreakdown`` instance to mutate.
        stage:     Attribute prefix — e.g. ``"dense"`` writes
                   ``breakdown.dense_ms``.

    Usage::

        lat = LatencyBreakdown()
        with profile_stage(lat, "dense"):
            hits = vector_store.search(query_vec)
        # lat.dense_ms is now set
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1_000.0
        attr = f"{stage}_ms"
        if hasattr(breakdown, attr):
            setattr(breakdown, attr, elapsed_ms)


# ══════════════════════════════════════════════════════════════════════════════
# Module 2 — Entity Linker
# ══════════════════════════════════════════════════════════════════════════════

#: Estimated words per printed page in a dense 10-K filing.
_WORDS_PER_PAGE: int = 400

#: Compiled regex → canonical section label.  Longer patterns checked first
#: (item 7a before item 7; item 1a/1b before item 1) to avoid early-match.
_SEC_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"item\s*1a[.\s]+risk\s+factor",                  re.I), "Item 1A. Risk Factors"),
    (re.compile(r"item\s*1b[.\s]+unresolved\s+staff\s+comment",   re.I), "Item 1B. Unresolved Staff Comments"),
    (re.compile(r"item\s*1[.\s]+business",                        re.I), "Item 1. Business"),
    (re.compile(r"item\s*2[.\s]+propert",                         re.I), "Item 2. Properties"),
    (re.compile(r"item\s*3[.\s]+legal\s+proceed",                 re.I), "Item 3. Legal Proceedings"),
    (re.compile(r"item\s*4[.\s]+mine\s+safety",                   re.I), "Item 4. Mine Safety Disclosures"),
    (re.compile(r"item\s*5[.\s]+market",                          re.I), "Item 5. Market for Registrant"),
    (re.compile(r"item\s*6[.\s]+selected\s+financial",            re.I), "Item 6. Selected Financial Data"),
    (re.compile(r"item\s*7a[.\s]+quantitative",                   re.I), "Item 7A. Quantitative Disclosures"),
    (re.compile(r"item\s*7[.\s]+management",                      re.I), "Item 7. MD&A"),
    (re.compile(r"item\s*8[.\s]+financial\s+statement",           re.I), "Item 8. Financial Statements"),
    (re.compile(r"item\s*9a[.\s]+controls",                       re.I), "Item 9A. Controls and Procedures"),
    (re.compile(r"item\s*9b[.\s]+other",                          re.I), "Item 9B. Other Information"),
    (re.compile(r"item\s*9[.\s]+changes",                         re.I), "Item 9. Changes in Accountants"),
    (re.compile(r"item\s*10[.\s]+director",                       re.I), "Item 10. Directors"),
    (re.compile(r"item\s*11[.\s]+executive\s+comp",               re.I), "Item 11. Executive Compensation"),
    (re.compile(r"item\s*12[.\s]+security\s+ownership",           re.I), "Item 12. Security Ownership"),
    (re.compile(r"item\s*13[.\s]+certain\s+relation",             re.I), "Item 13. Certain Relationships"),
    (re.compile(r"item\s*14[.\s]+principal\s+account",            re.I), "Item 14. Principal Accountant"),
    (re.compile(r"item\s*15[.\s]+exhibit",                        re.I), "Item 15. Exhibits"),
]


class EntityLinker:
    """
    Stateful position tracker for a single SEC 10-K document.

    Called **once per document** during ingestion.  After each element is
    processed, call ``advance()`` to move the internal cursor.  Use
    ``build_citation()`` *before* advancing to tag the chunk with its
    correct position.

    Citation format::

        "[Item 1A. Risk Factors, ~page 47, ¶3]"

    Page numbers are estimated from cumulative word count ÷ ``_WORDS_PER_PAGE``
    and are explicitly labelled ``~page`` to signal approximation.

    Thread-safety
    -------------
    Create one ``EntityLinker`` per ingestion job; **do not share** instances
    across concurrent calls.
    """

    def __init__(self) -> None:
        self._cumulative_words:     int = 0
        self._current_section:      str = "Preamble"
        self._paragraph_in_section: int = 0

    # ── section detection ─────────────────────────────────────────────────────

    @staticmethod
    def detect_section(text: str) -> Optional[str]:
        """
        Return the canonical SEC section label if *text* is a section heading.

        Returns ``None`` when no pattern matches (i.e. ordinary body text).
        """
        for pattern, label in _SEC_PATTERNS:
            if pattern.search(text):
                return label
        return None

    # ── state management ──────────────────────────────────────────────────────

    def update_section(self, new_section: str) -> None:
        """Switch to *new_section* and reset the paragraph counter to 0."""
        self._current_section      = new_section
        self._paragraph_in_section = 0

    def advance(self, word_count: int, is_new_section: bool = False) -> None:
        """
        Advance the position cursor after processing one text element.

        Args:
            word_count:     Number of words in the element.
            is_new_section: True when this element triggered a section change;
                            resets paragraph counter to 1.
        """
        self._cumulative_words += word_count
        self._paragraph_in_section = 1 if is_new_section else self._paragraph_in_section + 1

    # ── read-only properties ─────────────────────────────────────────────────

    @property
    def estimated_page(self) -> int:
        """1-based estimated page number derived from cumulative word count."""
        return max(1, self._cumulative_words // _WORDS_PER_PAGE + 1)

    @property
    def current_section(self) -> str:
        return self._current_section

    @property
    def paragraph_index(self) -> int:
        return self._paragraph_in_section

    # ── citation builder ──────────────────────────────────────────────────────

    def build_citation(
        self,
        section_override:   Optional[str] = None,
        paragraph_override: Optional[int] = None,
    ) -> str:
        """
        Build a human-readable citation string at the **current** cursor
        position (call this *before* ``advance()``).

        Args:
            section_override:   Override the current section label.
            paragraph_override: Override the current paragraph index.

        Returns:
            e.g. ``"[Item 7. MD&A, ~page 47, ¶3]"``
        """
        sec  = section_override  or self._current_section
        para = paragraph_override if paragraph_override is not None else self._paragraph_in_section
        return f"[{sec}, ~page {self.estimated_page}, ¶{para}]"


# ══════════════════════════════════════════════════════════════════════════════
# Module 3 — Financial Table Extractor
# ══════════════════════════════════════════════════════════════════════════════

_MULT_PATTERNS: List[Tuple[re.Pattern[str], float]] = [
    (re.compile(r"in\s+trillions?",  re.I), 1e12),
    (re.compile(r"in\s+billions?",   re.I), 1e9),
    (re.compile(r"in\s+millions?",   re.I), 1e6),
    (re.compile(r"in\s+thousands?",  re.I), 1e3),
]

# Maps regex → canonical metric key.  Order matters: match more specific
# patterns before generic ones.
_METRIC_MAP: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"free\s+cash\s+flow",                              re.I), "free_cash_flow"),
    (re.compile(r"(total\s+)?revenue|net\s+revenue|total\s+sales",  re.I), "revenue"),
    (re.compile(r"gross\s+(profit|margin)",                          re.I), "gross_profit"),
    (re.compile(r"operating\s+income|income\s+from\s+operations",   re.I), "operating_income"),
    (re.compile(r"net\s+income|net\s+earnings",                     re.I), "net_income"),
    (re.compile(r"(basic|diluted)\s+eps|earnings\s+per\s+(diluted\s+)?share", re.I), "eps_diluted"),
    (re.compile(r"total\s+assets",                                   re.I), "total_assets"),
    (re.compile(r"total\s+(long.?term\s+)?debt",                    re.I), "total_debt"),
    (re.compile(r"^cash(\s+and\s+cash\s+equivalents)?$",            re.I), "cash"),
    (re.compile(r"capital\s+expenditures?|capex",                   re.I), "capex"),
    (re.compile(r"research\s+(and\s+)?development|\bR&D\b",         re.I), "r_and_d"),
    (re.compile(r"operating\s+expenses?",                            re.I), "operating_expenses"),
    (re.compile(r"income\s+tax",                                    re.I), "income_tax"),
    (re.compile(r"shares?\s+outstanding",                            re.I), "shares_outstanding"),
    (re.compile(r"dividend.+per\s+share",                           re.I), "dividends_per_share"),
]

_YEAR_RE = re.compile(r"\b(20\d{2})\b")

_FINANCIAL_SECTIONS = frozenset({
    "Item 6. Selected Financial Data",
    "Item 7. MD&A",
    "Item 7A. Quantitative Disclosures",
    "Item 8. Financial Statements",
})


def _parse_cell(raw: str) -> Optional[float]:
    """Parse one table cell to float; returns None for ``—`` / ``N/A``."""
    s = raw.strip()
    if not s or s in {"—", "–", "-", "N/A", "n/a", "nm", "NM"}:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[\$\(\)\s,]", "", s)
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def _detect_multiplier(header_block: str) -> float:
    """Return the numeric scale from an 'in millions' style table header."""
    for pattern, mult in _MULT_PATTERNS:
        if pattern.search(header_block):
            return mult
    return 1.0


def _canonicalise_metric(label: str) -> Optional[str]:
    """Map a raw row label to a canonical metric key, or ``None``."""
    for pattern, key in _METRIC_MAP:
        if pattern.search(label.strip()):
            return key
    return None


def extract_financial_highlights(chunks: List[Any]) -> Dict[str, Any]:
    """
    Extract structured financial metrics from SEC table chunks.

    Accepts any sequence of objects with ``.text``, ``.is_table``, and
    ``.sec_section`` attributes — both ``Chunk`` (ingestion) and
    ``QueryResult`` (retrieval) objects are compatible.

    Algorithm
    ---------
    1. Filter to table chunks or chunks from Items 6, 7, 8.
    2. Scan the first five lines for ≥ 2 four-digit year tokens (column headers).
    3. Detect the ``in millions / billions`` multiplier from the header block.
    4. For each body line, map the left label to a canonical metric; pair the
       right-hand values with the year columns.
    5. Merge across chunks — later chunks overwrite earlier values for the
       same metric/year, so more-specific tables win.

    Returns
    -------
    ::

        {
          "revenue":          {"2024": 245_122_000_000.0, "2023": 211_915_000_000.0},
          "operating_income": {"2024": 109_433_000_000.0, "2023":  88_523_000_000.0},
          "_meta": {
              "multiplier":     1_000_000.0,
              "source_section": "Item 7. MD&A",
              "chunks_parsed":  2,
          },
        }

    Returns ``{"_meta": {...}}`` (no metric keys) when no parseable tables
    are found in the input chunks.
    """
    result: Dict[str, Any] = {}
    chunks_parsed  = 0
    multiplier     = 1.0
    source_section = "Unknown"

    for chunk in chunks:
        is_table    = getattr(chunk, "is_table",    False)
        sec_section = getattr(chunk, "sec_section", "")
        text        = getattr(chunk, "text",        "")

        if not (is_table or sec_section in _FINANCIAL_SECTIONS):
            continue
        if not text:
            continue

        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue

        # ── locate year column headers ────────────────────────────────────
        year_cols: List[str] = []
        for line in lines[:5]:
            found = _YEAR_RE.findall(line)
            if len(found) >= 2:
                year_cols = found
                break
        if not year_cols:
            continue

        # ── detect scale multiplier ───────────────────────────────────────
        header_block = "\n".join(lines[:8])
        mult         = _detect_multiplier(header_block)
        if mult != 1.0:
            multiplier     = mult
            source_section = sec_section

        chunks_parsed += 1

        # ── parse metric rows ─────────────────────────────────────────────
        for line in lines:
            # Financial tables use ≥2 spaces or tabs as column delimiters
            parts = re.split(r"\s{2,}|\t", line.strip())
            if len(parts) < 2:
                continue

            canonical = _canonicalise_metric(parts[0])
            if canonical is None:
                continue

            value_cells = [p.strip() for p in parts[1:] if p.strip()]
            entry: Dict[str, float] = {}
            for i, year in enumerate(year_cols):
                if i >= len(value_cells):
                    break
                parsed = _parse_cell(value_cells[i])
                if parsed is not None:
                    entry[year] = parsed * mult

            if entry:
                existing = result.get(canonical, {})
                existing.update(entry)   # later / more-specific chunks win
                result[canonical] = existing

    result["_meta"] = {
        "multiplier":     multiplier,
        "source_section": source_section,
        "chunks_parsed":  chunks_parsed,
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Module 4 — Numeric Utilities
# ══════════════════════════════════════════════════════════════════════════════


class NumericUtils:
    """
    Stateless utilities for normalising and comparing financial number strings.

    All methods are ``@staticmethod``; instantiation is unnecessary.

    Examples
    --------
    >>> NumericUtils.match("$245,122 million", "245122")
    True
    >>> NumericUtils.match("15%", "0.15")
    True
    >>> NumericUtils.extract("($3.5 billion)")
    -3500000000.0
    """

    _SCALE_WORDS: List[Tuple[str, float]] = [
        ("trillion", 1e12),
        ("billion",  1e9),
        ("million",  1e6),
        ("thousand", 1e3),
    ]

    @staticmethod
    def extract(text: str) -> Optional[float]:
        """
        Normalise the primary numeric value from a financial string.

        Handles
        -------
        * ``"$245,122 million"``   →  245_122_000_000.0
        * ``"(1,234)"``            →  -1_234.0  (accounting parentheses)
        * ``"15%"``                →  0.15
        * ``"$3.5B"``              →  3_500_000_000.0
        * ``"1.2 trillion"``       →  1_200_000_000_000.0
        * ``"N/A"`` / ``"—"``     →  None
        """
        if not text:
            return None

        s = text.strip().lower()
        if s in {"n/a", "—", "–", "", "-", "nm"}:
            return None

        # Accounting negatives: "(1,234)" → negative
        is_negative = bool(re.fullmatch(r"\(.*\)", s))
        if is_negative:
            s = s.strip("()")

        # Strip shorthand suffixes (B/M/T/K) before general parsing
        suffix_map = {"b": 1e9, "m": 1e6, "t": 1e12, "k": 1e3}
        m_suffix = re.search(r"(\d)\s*([bmtk])$", s)
        shorthand_mult = 1.0
        if m_suffix:
            shorthand_mult = suffix_map.get(m_suffix.group(2), 1.0)
            s = s[: m_suffix.start(2)].strip()

        s = re.sub(r"[\$£€¥]", "", s)
        s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)   # strip thousands separators

        multiplier = shorthand_mult
        for word, mult in NumericUtils._SCALE_WORDS:
            if re.search(rf"\b{word}\b", s):
                multiplier = mult
                s = re.sub(rf"\b{word}\b", "", s)
                break

        is_pct = "%" in s
        s = s.replace("%", "").strip()

        m = re.search(r"-?\d+\.?\d*", s)
        if not m:
            return None
        try:
            val = float(m.group()) * multiplier
            if is_pct:
                val /= 100.0
            return -abs(val) if is_negative else val
        except ValueError:
            return None

    @staticmethod
    def match(pred: str, ref: str, rtol: float = 0.01) -> bool:
        """
        ``True`` when *pred* and *ref* encode the same financial number (±*rtol*).

        Args:
            pred:  Predicted / generated number string.
            ref:   Reference (ground-truth) number string.
            rtol:  Relative tolerance (default 1 %).

        Examples::

            match("$245,122 million", "245122")  → True
            match("$3.5B",  "3.5 billion")        → True
            match("15%",    "0.15")               → True
            match("1000",   "999")                → False
        """
        p = NumericUtils.extract(pred)
        r = NumericUtils.extract(ref)
        if p is None or r is None:
            return False
        if r == 0.0:
            return abs(p) < 1e-9
        return abs(p - r) / abs(r) <= rtol

    @staticmethod
    def format_display(value: float, multiplier: float = 1.0) -> str:
        """
        Format a raw float for display in a financial report.

        Args:
            value:      The raw float at full scale.
            multiplier: The unit the caller wants to express it in
                        (e.g. 1e6 → display in millions).

        Examples::

            format_display(245_122_000_000.0, 1e6)  → "$245,122.0M"
            format_display(3_500_000_000.0,   1e9)  → "$3.5B"
        """
        suffix = {1e12: "T", 1e9: "B", 1e6: "M", 1e3: "K"}.get(multiplier, "")
        v = value / multiplier if multiplier not in (0.0, 1.0) else value
        return f"${v:,.1f}{suffix}" if abs(v) >= 1_000 else f"${v:.4g}{suffix}"

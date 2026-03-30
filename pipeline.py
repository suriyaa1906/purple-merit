"""
rag/pipeline.py
RAG Pipeline: ingestion → chunking → embedding → FAISS vector store → retrieval

Chunking strategy:
  - chunk_size=512 tokens (~400 words): large enough to preserve policy context,
    small enough to stay semantically focused on one rule/section.
  - overlap=64 tokens (~50 words): prevents information loss at chunk boundaries,
    ensures a clause that spans two chunks is captured by at least one.
  - Metadata preserved: doc_id, section header, page estimate, chunk index.
"""

import os
import re
import json
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ── Optional heavy deps (graceful fallback for environments without GPU) ──────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[pipeline] FAISS not installed. Using in-memory cosine search fallback.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[pipeline] openai not installed. Embeddings will be mocked.")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyChunk:
    """A single retrievable unit of policy text with full provenance."""
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    char_start: int
    char_end: int
    chunk_index: int
    total_chunks_in_doc: int
    embedding: Optional[list] = field(default=None, repr=False)

    def citation_ref(self) -> str:
        return f"[{self.doc_id} § {self.section}]"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding")  # don't serialise embedding to JSON
        return d


@dataclass
class RetrievalResult:
    chunk: PolicyChunk
    score: float  # cosine similarity 0–1

    @property
    def citation(self) -> str:
        return self.chunk.citation_ref()


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def _extract_section_header(text_before: str) -> str:
    """Find the most recent markdown heading before a chunk starts."""
    headers = re.findall(r'^#{1,4}\s+(.+)$', text_before, re.MULTILINE)
    return headers[-1].strip() if headers else "General"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 0.75 words ≈ 4 chars."""
    return len(text) // 4


def chunk_document(
    text: str,
    doc_id: str,
    doc_title: str,
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[PolicyChunk]:
    """
    Split a policy document into overlapping chunks.

    Strategy:
    1. Split on double-newlines (paragraph boundaries) first.
    2. Accumulate paragraphs into a chunk until chunk_size_tokens is reached.
    3. On overflow, save the chunk and slide the window back by overlap_tokens.
    4. Preserve section header context in metadata.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks: list[PolicyChunk] = []
    buffer_paras: list[str] = []
    buffer_tokens = 0
    buffer_char_start = 0
    char_cursor = 0
    para_char_starts: list[int] = []

    # Pre-compute character positions
    pos = 0
    para_positions: list[int] = []
    for para in paragraphs:
        para_positions.append(pos)
        pos += len(para) + 2  # +2 for the newlines we stripped

    def _save_chunk(buf_paras, c_start, c_end, idx):
        joined = "\n\n".join(buf_paras)
        # Determine section from text preceding this chunk
        text_before = text[:c_start]
        section = _extract_section_header(text_before)
        cid = hashlib.md5(f"{doc_id}:{idx}:{joined[:50]}".encode()).hexdigest()[:12]
        return PolicyChunk(
            chunk_id=cid,
            doc_id=doc_id,
            doc_title=doc_title,
            section=section,
            text=joined,
            char_start=c_start,
            char_end=c_end,
            chunk_index=idx,
            total_chunks_in_doc=0,  # patched after
        )

    chunk_idx = 0
    para_idx = 0
    while para_idx < len(paragraphs):
        para = paragraphs[para_idx]
        para_tokens = _estimate_tokens(para)
        p_start = para_positions[para_idx]

        if not buffer_paras:
            buffer_char_start = p_start

        if buffer_tokens + para_tokens > chunk_size_tokens and buffer_paras:
            # Save current buffer
            c_end = para_positions[para_idx - 1] + len(paragraphs[para_idx - 1])
            chunks.append(_save_chunk(buffer_paras, buffer_char_start, c_end, chunk_idx))
            chunk_idx += 1

            # Slide back by overlap_tokens
            overlap_accum = 0
            keep_from = len(buffer_paras)
            for i in range(len(buffer_paras) - 1, -1, -1):
                overlap_accum += _estimate_tokens(buffer_paras[i])
                if overlap_accum >= overlap_tokens:
                    keep_from = i
                    break
            buffer_paras = buffer_paras[keep_from:]
            buffer_tokens = sum(_estimate_tokens(p) for p in buffer_paras)
            # Recalculate start
            if buffer_paras:
                # Find first remaining para in paragraphs list
                first_kept = buffer_paras[0]
                for pi, pp in enumerate(paragraphs):
                    if pp == first_kept:
                        buffer_char_start = para_positions[pi]
                        break

        buffer_paras.append(para)
        buffer_tokens += para_tokens
        para_idx += 1

    # Flush remaining
    if buffer_paras:
        c_end = para_positions[-1] + len(paragraphs[-1])
        chunks.append(_save_chunk(buffer_paras, buffer_char_start, c_end, chunk_idx))

    # Patch total_chunks_in_doc
    total = len(chunks)
    for c in chunks:
        c.total_chunks_in_doc = total

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """Wrapper around OpenAI text-embedding-ada-002 with mock fallback."""

    MODEL = "text-embedding-ada-002"
    DIM = 1536

    def __init__(self, api_key: Optional[str] = None):
        self.client = None
        if OPENAI_AVAILABLE:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if key:
                self.client = OpenAI(api_key=key)
        if not self.client:
            print("[EmbeddingModel] No API key — using deterministic mock embeddings.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.client:
            resp = self.client.embeddings.create(model=self.MODEL, input=texts)
            return [item.embedding for item in resp.data]
        # Deterministic mock: hash → pseudo-random unit vector
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            results.append(v.tolist())
        return results

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ─────────────────────────────────────────────────────────────────────────────
# Vector store
# ─────────────────────────────────────────────────────────────────────────────

class PolicyVectorStore:
    """
    FAISS-backed vector store for policy chunks.
    Falls back to brute-force cosine if FAISS is unavailable.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.chunks: list[PolicyChunk] = []
        self.index = None
        self._embeddings: Optional[np.ndarray] = None  # fallback matrix

        if FAISS_AVAILABLE:
            # IndexFlatIP = inner product (cosine after normalisation)
            self.index = faiss.IndexFlatIP(dim)

    def add_chunks(self, chunks: list[PolicyChunk]):
        """Add chunks (must have .embedding set) to the store."""
        vecs = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding.")
            v = np.array(chunk.embedding, dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            vecs.append(v)
            self.chunks.append(chunk)

        matrix = np.vstack(vecs)
        if FAISS_AVAILABLE:
            self.index.add(matrix)
        else:
            if self._embeddings is None:
                self._embeddings = matrix
            else:
                self._embeddings = np.vstack([self._embeddings, matrix])

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievalResult]:
        """Return top-k most similar chunks with cosine scores."""
        qv = np.array(query_embedding, dtype=np.float32)
        qv /= np.linalg.norm(qv)

        if FAISS_AVAILABLE:
            scores, indices = self.index.search(qv.reshape(1, -1), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                results.append(RetrievalResult(chunk=self.chunks[idx], score=float(score)))
            return results
        else:
            # Brute-force cosine
            sims = self._embeddings @ qv
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [RetrievalResult(chunk=self.chunks[i], score=float(sims[i])) for i in top_indices]

    def save(self, path: str):
        """Persist store to disk."""
        data = {
            "chunks": [c.to_dict() for c in self.chunks],
            "embeddings": [c.embedding for c in self.chunks],
        }
        with open(path + ".meta.json", "w") as f:
            json.dump(data, f, indent=2)
        if FAISS_AVAILABLE:
            faiss.write_index(self.index, path + ".faiss")
        else:
            np.save(path + ".embeddings.npy", self._embeddings)
        print(f"[VectorStore] Saved {len(self.chunks)} chunks to {path}")

    @classmethod
    def load(cls, path: str, dim: int = 1536) -> "PolicyVectorStore":
        """Load store from disk."""
        store = cls(dim=dim)
        with open(path + ".meta.json") as f:
            data = json.load(f)
        if FAISS_AVAILABLE:
            store.index = faiss.read_index(path + ".faiss")
        else:
            store._embeddings = np.load(path + ".embeddings.npy")
        for chunk_dict, emb in zip(data["chunks"], data["embeddings"]):
            chunk_dict["embedding"] = emb
            store.chunks.append(PolicyChunk(**chunk_dict))
        print(f"[VectorStore] Loaded {len(store.chunks)} chunks from {path}")
        return store


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    Orchestrates: load docs → chunk → embed → store in FAISS.

    Usage:
        pipeline = IngestionPipeline(policy_dir="policies/", store_path="data/store")
        pipeline.ingest()
        store = pipeline.store
    """

    POLICY_DOC_IDS = {
        "01_returns_refunds.md": ("POL-RR-001", "Returns & Refunds Policy"),
        "02_shipping_delivery.md": ("POL-SD-002", "Shipping & Delivery Policy"),
        "03_cancellations_promos_disputes.md": ("POL-CX-003/POL-PR-004/POL-DB-005",
                                                "Cancellation, Promotions & Disputes Policy"),
        "04_edge_cases_exceptions.md": ("POL-EX-006", "Edge Cases & Exception Policy"),
    }

    def __init__(
        self,
        policy_dir: str = "policies/",
        store_path: str = "data/vector_store",
        chunk_size: int = 512,
        overlap: int = 64,
        top_k: int = 5,
        api_key: Optional[str] = None,
    ):
        self.policy_dir = Path(policy_dir)
        self.store_path = store_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.embedder = EmbeddingModel(api_key=api_key)
        self.store = PolicyVectorStore(dim=self.embedder.DIM)

    def ingest(self, batch_size: int = 20):
        """Run full ingestion pipeline."""
        all_chunks: list[PolicyChunk] = []

        for filename, (doc_id, doc_title) in self.POLICY_DOC_IDS.items():
            fpath = self.policy_dir / filename
            if not fpath.exists():
                print(f"[Ingestion] WARNING: {fpath} not found — skipping.")
                continue
            text = fpath.read_text(encoding="utf-8")
            chunks = chunk_document(text, doc_id, doc_title, self.chunk_size, self.overlap)
            all_chunks.extend(chunks)
            print(f"[Ingestion] {doc_title}: {len(chunks)} chunks")

        print(f"[Ingestion] Total chunks: {len(all_chunks)}. Embedding in batches of {batch_size}...")

        # Embed in batches
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb

        self.store.add_chunks(all_chunks)

        # Persist
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        self.store.save(self.store_path)
        print(f"[Ingestion] Done. Store ready with {len(all_chunks)} chunks.")
        return self.store


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = IngestionPipeline(
        policy_dir="policies/",
        store_path="data/vector_store",
    )
    store = pipeline.ingest()

    # Smoke-test retrieval
    query = "Can I return a used lipstick if I don't like the colour?"
    q_emb = pipeline.embedder.embed_single(query)
    results = store.search(q_emb, top_k=3)
    print("\n=== Retrieval smoke-test ===")
    print(f"Query: {query}\n")
    for r in results:
        print(f"Score: {r.score:.3f} | {r.citation}")
        print(f"  {r.chunk.text[:200]}...\n")

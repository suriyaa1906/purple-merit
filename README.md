# E-Commerce Customer Support — Multi-Agent RAG System

A production-grade, **4-agent RAG pipeline** that resolves e-commerce support tickets using
policy documents as its only source of truth. Every claim is cited. No hallucinations allowed.

---

## Architecture

```
Ticket + Order JSON
        │
        ▼
┌─────────────────┐
│  1. TRIAGE      │  → classify, detect gaps, form retrieval queries
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────────────────┐
│  2. RETRIEVER   │◄────│  FAISS Vector Store               │
│  (Policy RAG)   │     │  chunk_size=512 | overlap=64     │
└────────┬────────┘     │  model: text-embedding-ada-002   │
         │              └──────────────────────────────────┘
         ▼
┌─────────────────┐
│  3. WRITER      │  → evidence-only resolution + customer draft
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. COMPLIANCE  │  → citation audit, hallucination check, rewrite/escalate
└────────┬────────┘
         │
    ┌────┴──────────────┐
    ▼                   ▼
FinalOutput         Escalation
(all 7 fields)      (human queue)
```

---

## Agents

### 1. Triage Agent
- Classifies ticket into 11 categories with confidence score
- Identifies missing information, asks ≤3 clarifying questions
- Generates 2–4 retrieval queries optimised for vector search
- Detects sentiment and urgency

### 2. Policy Retriever Agent
- Runs all retrieval queries against FAISS
- De-duplicates results across queries
- Returns top-5 unique chunks with cosine similarity scores and full citations
- Falls back gracefully: no evidence → immediate escalation path

### 3. Resolution Writer Agent
- Generates decisions: `approve | deny | partial | escalate | needs_info`
- Every claim must cite `[DOC_ID § SECTION]`
- Self-reports unsupported claims
- Cannot invent policy — must escalate if evidence is missing

### 4. Compliance / Safety Agent
- Checks citation coverage score (target ≥ 0.9)
- Cross-references unsupported claims list
- Detects policy violations (e.g. approving Final Sale return)
- Forces rewrite (up to 2x) then escalates if still failing

---

## RAG Pipeline Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 512 tokens (~400 words) | Large enough to preserve cross-sentence policy context |
| Overlap | 64 tokens (~50 words) | Prevents information loss at paragraph boundaries |
| Embedding model | text-embedding-ada-002 | 1536-dim, strong semantic similarity for policy text |
| Vector store | FAISS IndexFlatIP | Exact cosine search; fast enough for policy-scale corpus |
| Top-k retrieval | 5 unique chunks | Broad enough to catch edge-case clauses |
| Multi-query | Yes (2–4 queries) | Different angles of the same question improve recall |

---

## Output Format (all 7 fields)

```
1. Classification    — category + confidence %
2. Clarifying Qs     — max 3, only if needed
3. Decision          — approve / deny / partial / escalate / needs_info
4. Rationale         — policy-grounded, internal
5. Citations         — [DOC_ID § SECTION] for every claim
6. Customer Response — empathetic, clear, actionable
7. Internal Notes    — exception codes, compliance status, next steps
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (at least one required)
export ANTHROPIC_API_KEY="your-key"   # or OPENAI_API_KEY
# If neither is set, deterministic mock responses are used (for development)

# 3. Ingest policies into vector store
python main.py --mode ingest

# 4. Run 3 demo tickets
python main.py --mode demo

# 5. Run full 20-case evaluation
python main.py --mode eval

# 6. Process a single ticket
python main.py --mode ticket \
  --ticket "I want to return my jacket, delivered last week" \
  --order '{"order_id":"ORD-999","delivery_date":"2024-10-08","payment_method":"credit_card"}' \
  --tier standard
```

---

## Evaluation Dataset

| Tier | Count | Focus |
|------|-------|-------|
| Standard | 8 | Common return, refund, shipping, promo, cancel scenarios |
| Exception-heavy | 6 | Hygiene items, Final Sale, perishables, digital, Platinum window |
| Policy-conflict | 3 | Holiday vs Platinum, Defective vs Final Sale, Outage promo conflict |
| Not-in-policy | 3 | Price match, direct exchange, consumable component return |

### Metrics
- **Citation Coverage Rate**: fraction of cases where expected doc IDs appear in citations
- **Correct Decision Rate**: fraction of cases matching expected decision
- **Correct Escalation Rate**: fraction of should-escalate cases that actually escalated
- **Compliance Pass Rate**: fraction of cases passing the compliance audit

---

## Policy Documents

| File | Document ID | Coverage |
|------|-------------|----------|
| 01_returns_refunds.md | POL-RR-001 | Return windows, refund timelines, non-returnable items |
| 02_shipping_delivery.md | POL-SD-002 | Shipping methods, lost packages, international |
| 03_cancellations_promos_disputes.md | POL-CX-003/PR-004/DB-005 | Cancellations, promo codes, disputes |
| 04_edge_cases_exceptions.md | POL-EX-006 | Conflict resolution, agent authority, not-in-policy escalation |

---

## No-Hallucination Guarantees

1. **Writer agent constraint**: explicitly forbidden from inferring or inventing policy
2. **Self-reporting**: writer must list any unsupported claims it makes
3. **Compliance audit**: independent check cross-references claims against retrieved evidence
4. **Rewrite loop**: up to 2 rewrites before mandatory escalation
5. **Evidence gating**: if no evidence is retrieved, decision is automatically `escalate`
6. **Citation enforcement**: every claim must use `[DOC_ID § SECTION]` format

---

## Folder Structure

```
ecom_support_rag/
├── main.py                          # Entry point
├── requirements.txt
├── README.md
├── policies/
│   ├── 01_returns_refunds.md        # POL-RR-001
│   ├── 02_shipping_delivery.md      # POL-SD-002
│   ├── 03_cancellations_promos_disputes.md  # POL-CX-003/PR-004/DB-005
│   └── 04_edge_cases_exceptions.md  # POL-EX-006
├── rag/
│   └── pipeline.py                  # Chunking, embeddings, FAISS store
├── agents/
│   └── agents.py                    # All 4 agents + orchestrator
├── evaluation/
│   └── eval_pipeline.py             # 20 test cases + metrics
├── data/                            # Generated: FAISS index + metadata
└── outputs/                         # Generated: eval report + sample outputs
```

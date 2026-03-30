"""
agents/agents.py
Four-agent multi-agent RAG system for e-commerce customer support.

Agent architecture:
  1. TriageAgent       — classify, detect gaps, ask clarifying questions
  2. PolicyRetrieverAgent — query vector store, return ranked excerpts + citations
  3. ResolutionWriterAgent — generate evidence-only resolution + customer response
  4. ComplianceAgent   — verify citations, flag hallucinations, force rewrite/escalate
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pipeline import IngestionPipeline, PolicyVectorStore, EmbeddingModel


# ─────────────────────────────────────────────────────────────────────────────
# Shared types
# ─────────────────────────────────────────────────────────────────────────────

class IssueCategory(str, Enum):
    REFUND = "refund"
    RETURN = "return"
    SHIPPING = "shipping"
    CANCELLATION = "cancellation"
    PROMOTION = "promotion"
    DISPUTE = "dispute"
    DEFECTIVE = "defective"
    LOST_PACKAGE = "lost_package"
    SUBSCRIPTION = "subscription"
    EXCEPTION = "exception"
    UNKNOWN = "unknown"


class Decision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    PARTIAL = "partial"
    ESCALATE = "escalate"
    NEEDS_INFO = "needs_info"


@dataclass
class TicketContext:
    ticket_text: str
    order_data: dict
    conversation_history: list[dict] = field(default_factory=list)
    customer_tier: str = "standard"  # standard | silver | gold | platinum


@dataclass
class TriageResult:
    category: IssueCategory
    confidence: float          # 0.0 – 1.0
    sub_issues: list[str]
    missing_info: list[str]
    clarifying_questions: list[str]   # max 3
    retrieval_queries: list[str]      # formulated for vector store
    sentiment: str                    # neutral | frustrated | angry | satisfied
    urgency: str                      # low | medium | high


@dataclass
class PolicyEvidence:
    query: str
    results: list[RetrievalResult]
    total_retrieved: int

    @property
    def has_evidence(self) -> bool:
        return len(self.results) > 0

    def format_for_writer(self) -> str:
        """Format retrieved chunks as numbered evidence for the writer agent."""
        if not self.results:
            return "NO RELEVANT POLICY FOUND."
        lines = []
        for i, r in enumerate(self.results, 1):
            lines.append(
                f"[EVIDENCE {i}] Citation: {r.citation} | Score: {r.score:.3f}\n"
                f"{r.chunk.text}\n"
            )
        return "\n---\n".join(lines)


@dataclass
class ResolutionDraft:
    decision: Decision
    rationale: str                    # internal, policy-grounded
    citations: list[str]              # e.g. ["[POL-RR-001 § 4.2]", ...]
    customer_response: str            # customer-facing prose
    internal_notes: str               # for agent / CRM
    unsupported_claims: list[str]     # self-reported by writer
    evidence_used: list[int]          # evidence indices used


@dataclass
class ComplianceResult:
    passed: bool
    issues: list[str]
    action: str                       # "approved" | "rewrite" | "escalate"
    citation_coverage_score: float    # 0.0 – 1.0
    unsupported_claim_count: int
    final_response: Optional["FinalOutput"] = None


@dataclass
class FinalOutput:
    ticket_id: str
    category: IssueCategory
    category_confidence: float
    clarifying_questions: list[str]
    decision: Decision
    rationale: str
    citations: list[str]
    customer_response: str
    internal_notes: str
    compliance_passed: bool
    rewrite_count: int
    escalated: bool
    escalation_reason: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# LLM caller (Claude / OpenAI compatible)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_message: str, json_mode: bool = False) -> str:
    """
    Call an LLM. Tries Anthropic Claude first, falls back to OpenAI, then mock.
    Set ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.
    """
    import os

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return msg.content[0].text
        except Exception as e:
            print(f"[LLM] Anthropic error: {e}. Trying OpenAI...")

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"} if json_mode else None,
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[LLM] OpenAI error: {e}. Using mock response.")

    # Mock fallback for development/testing
    return _mock_llm_response(system_prompt, user_message)


def _mock_llm_response(system_prompt: str, user_message: str) -> str:
    """Deterministic mock — returns valid JSON for each agent type."""
    if "TRIAGE" in system_prompt:
        return json.dumps({
            "category": "return",
            "confidence": 0.85,
            "sub_issues": ["return eligibility", "refund method"],
            "missing_info": ["proof of purchase", "item condition"],
            "clarifying_questions": [
                "Is the item unopened or has it been used?",
                "When was the item delivered according to your tracking?",
                "Are you seeking a refund to your original payment method or store credit?"
            ],
            "retrieval_queries": [
                "return window policy standard customer",
                "refund method original payment",
                "condition requirements return"
            ],
            "sentiment": "neutral",
            "urgency": "medium"
        })
    elif "RESOLUTION WRITER" in system_prompt:
        return json.dumps({
            "decision": "approve",
            "rationale": "Customer is within the 30-day return window per POL-RR-001 Section 1.1.",
            "citations": ["[POL-RR-001 § 1.1]", "[POL-RR-001 § 2.1]"],
            "customer_response": (
                "Thank you for reaching out. Based on our returns policy, you are eligible "
                "to return your item within 30 days of delivery. Please initiate your return "
                "through your account portal and use the prepaid label if applicable."
            ),
            "internal_notes": "Standard return within window. No exceptions needed.",
            "unsupported_claims": [],
            "evidence_used": [1, 2]
        })
    elif "COMPLIANCE" in system_prompt:
        return json.dumps({
            "passed": True,
            "issues": [],
            "action": "approved",
            "citation_coverage_score": 1.0,
            "unsupported_claim_count": 0,
            "rewrite_instructions": None
        })
    return '{"error": "unknown agent"}'


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1: Triage
# ─────────────────────────────────────────────────────────────────────────────

TRIAGE_SYSTEM_PROMPT = """
You are the TRIAGE AGENT in an e-commerce customer support system.

YOUR ROLE:
- Analyse the support ticket and order context.
- Classify the primary issue into exactly one category.
- Identify any sub-issues.
- Spot missing information that would be needed to resolve the ticket.
- If key information is missing, generate UP TO 3 clarifying questions.
- Generate 2–4 retrieval queries optimised for searching a policy vector database.

CATEGORIES (choose one):
  refund | return | shipping | cancellation | promotion | dispute |
  defective | lost_package | subscription | exception | unknown

URGENCY SIGNALS (high): item lost, defective, customer very angry, time-sensitive.
SENTIMENT: neutral | frustrated | angry | satisfied

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{
  "category": "<category>",
  "confidence": <0.0-1.0>,
  "sub_issues": ["<sub-issue>", ...],
  "missing_info": ["<what is missing>", ...],
  "clarifying_questions": ["<question>"],  // MAX 3
  "retrieval_queries": ["<query>", ...],    // 2-4 queries
  "sentiment": "<neutral|frustrated|angry|satisfied>",
  "urgency": "<low|medium|high>"
}

RULES:
- Never infer or assume information not present in the ticket or order data.
- Never ask more than 3 clarifying questions.
- Retrieval queries should be keyword-rich and policy-focused (not conversational).
- If the category is genuinely unclear, use "unknown" and ask clarifying questions.
"""


class TriageAgent:
    """Agent 1: Classify ticket and prepare retrieval queries."""

    def __init__(self):
        self.name = "TriageAgent"

    def run(self, ctx: TicketContext) -> TriageResult:
        user_message = f"""
SUPPORT TICKET:
{ctx.ticket_text}

ORDER CONTEXT (JSON):
{json.dumps(ctx.order_data, indent=2)}

CUSTOMER TIER: {ctx.customer_tier}

CONVERSATION HISTORY:
{json.dumps(ctx.conversation_history, indent=2) if ctx.conversation_history else "None"}

Analyse this ticket and return the JSON triage result.
"""
        raw = call_llm(TRIAGE_SYSTEM_PROMPT, user_message, json_mode=True)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Extract JSON from markdown if model wrapped it
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}

        return TriageResult(
            category=IssueCategory(data.get("category", "unknown")),
            confidence=float(data.get("confidence", 0.5)),
            sub_issues=data.get("sub_issues", []),
            missing_info=data.get("missing_info", []),
            clarifying_questions=data.get("clarifying_questions", [])[:3],
            retrieval_queries=data.get("retrieval_queries", []),
            sentiment=data.get("sentiment", "neutral"),
            urgency=data.get("urgency", "medium"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2: Policy Retriever
# ─────────────────────────────────────────────────────────────────────────────

class PolicyRetrieverAgent:
    """
    Agent 2: Query the vector store with all retrieval queries from triage,
    de-duplicate results, and return ranked evidence with citations.
    """

    def __init__(self, store: PolicyVectorStore, embedder: EmbeddingModel, top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.name = "PolicyRetrieverAgent"

    def run(self, triage: TriageResult) -> PolicyEvidence:
        seen_chunk_ids = set()
        all_results: list[RetrievalResult] = []

        for query in triage.retrieval_queries:
            q_emb = self.embedder.embed_single(query)
            results = self.store.search(q_emb, top_k=self.top_k)
            for r in results:
                if r.chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(r.chunk.chunk_id)
                    all_results.append(r)

        # Re-rank by score, return top_k unique results
        all_results.sort(key=lambda r: r.score, reverse=True)
        top_results = all_results[:self.top_k]

        return PolicyEvidence(
            query=" | ".join(triage.retrieval_queries),
            results=top_results,
            total_retrieved=len(all_results),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3: Resolution Writer
# ─────────────────────────────────────────────────────────────────────────────

RESOLUTION_WRITER_SYSTEM_PROMPT = """
You are the RESOLUTION WRITER AGENT in an e-commerce customer support system.

YOUR ROLE:
- Read the customer ticket, order context, and the retrieved policy evidence.
- Generate a decision (approve / deny / partial / escalate / needs_info).
- Write a rationale grounded ONLY in the provided evidence.
- Write a professional, empathetic customer-facing response.
- List every citation you rely on.
- Flag any claims you are making that are NOT backed by evidence.

CRITICAL RULES — NO EXCEPTIONS:
1. You MUST cite every factual or policy claim using the format [DOC_ID § SECTION].
   e.g. "Per our returns policy [POL-RR-001 § 1.1], the standard window is 30 days."
2. You MUST NOT invent, assume, or extrapolate policy beyond what is in the evidence.
3. If the evidence does not cover the situation, set decision to "escalate" and explain
   what is missing in internal_notes. Do NOT make up an answer.
4. If two evidence items conflict, note the conflict in internal_notes and set
   decision to "escalate" unless the conflict resolution is explicitly stated.
5. evidence_used must list the [EVIDENCE N] numbers you relied on.
6. unsupported_claims must list any statement you made without a direct evidence source.
   If your response is fully cited, return [].

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{
  "decision": "<approve|deny|partial|escalate|needs_info>",
  "rationale": "<internal policy-grounded rationale>",
  "citations": ["[DOC_ID § SECTION]", ...],
  "customer_response": "<customer-facing text>",
  "internal_notes": "<agent-facing notes, exception codes, etc.>",
  "unsupported_claims": ["<claim not backed by evidence>", ...],
  "evidence_used": [1, 2, ...]
}
"""


class ResolutionWriterAgent:
    """Agent 3: Generate evidence-grounded resolution."""

    def __init__(self):
        self.name = "ResolutionWriterAgent"

    def run(
        self,
        ctx: TicketContext,
        triage: TriageResult,
        evidence: PolicyEvidence,
    ) -> ResolutionDraft:
        evidence_text = evidence.format_for_writer()
        if not evidence.has_evidence:
            evidence_text = (
                "⚠️ NO POLICY EVIDENCE RETRIEVED. "
                "You MUST set decision to 'escalate'. Do NOT invent policy."
            )

        user_message = f"""
SUPPORT TICKET:
{ctx.ticket_text}

ORDER CONTEXT:
{json.dumps(ctx.order_data, indent=2)}

CUSTOMER TIER: {ctx.customer_tier}

ISSUE CLASSIFICATION: {triage.category.value} (confidence: {triage.confidence:.0%})

RETRIEVED POLICY EVIDENCE:
{evidence_text}

Based ONLY on the evidence above, generate the resolution JSON.
If evidence is insufficient or absent, set decision to 'escalate'.
"""
        raw = call_llm(RESOLUTION_WRITER_SYSTEM_PROMPT, user_message, json_mode=True)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}

        return ResolutionDraft(
            decision=Decision(data.get("decision", "escalate")),
            rationale=data.get("rationale", "Unable to determine rationale."),
            citations=data.get("citations", []),
            customer_response=data.get("customer_response", "We are reviewing your request."),
            internal_notes=data.get("internal_notes", ""),
            unsupported_claims=data.get("unsupported_claims", []),
            evidence_used=data.get("evidence_used", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4: Compliance / Safety
# ─────────────────────────────────────────────────────────────────────────────

COMPLIANCE_SYSTEM_PROMPT = """
You are the COMPLIANCE & SAFETY AGENT in an e-commerce customer support system.

YOUR ROLE:
- Audit the Resolution Draft for hallucinations, missing citations, and policy violations.
- Determine whether to APPROVE, request a REWRITE, or ESCALATE.

CHECKS TO PERFORM:
1. CITATION COVERAGE: Every factual or policy claim in customer_response and rationale 
   must map to a citation. Calculate citation_coverage_score (0.0–1.0).
2. HALLUCINATION CHECK: Cross-reference unsupported_claims list. If non-empty and
   material to the decision, flag for rewrite.
3. POLICY VIOLATION: Does the decision contradict the policy evidence?
   e.g. approving a return for a Final Sale item, approving return outside window.
4. ESCALATION TRIGGERS:
   - No policy evidence exists for the situation
   - Conflicting policies not resolved
   - unsupported_claims list is non-empty with material claims
   - Decision would harm the customer based on incorrect policy reading

ACTION LOGIC:
- "approved": citation_coverage_score >= 0.9 AND unsupported_claims == [] AND no violations
- "rewrite": citation_coverage_score < 0.9 OR unsupported_claims non-empty (minor)
- "escalate": Major violation, conflicting policy, or no evidence

OUTPUT FORMAT — valid JSON only:
{
  "passed": <true|false>,
  "issues": ["<issue description>", ...],
  "action": "<approved|rewrite|escalate>",
  "citation_coverage_score": <0.0-1.0>,
  "unsupported_claim_count": <int>,
  "rewrite_instructions": "<specific instructions for rewrite, or null>"
}
"""


class ComplianceAgent:
    """
    Agent 4: Audit the resolution draft.
    Can approve, request rewrite (up to MAX_REWRITES), or escalate.
    """

    MAX_REWRITES = 2

    def __init__(self):
        self.name = "ComplianceAgent"

    def run(
        self,
        draft: ResolutionDraft,
        evidence: PolicyEvidence,
        rewrite_count: int = 0,
    ) -> ComplianceResult:
        evidence_text = evidence.format_for_writer()

        user_message = f"""
RESOLUTION DRAFT TO AUDIT:
decision: {draft.decision.value}
rationale: {draft.rationale}
citations: {json.dumps(draft.citations)}
customer_response: {draft.customer_response}
internal_notes: {draft.internal_notes}
unsupported_claims: {json.dumps(draft.unsupported_claims)}
evidence_used indices: {draft.evidence_used}

ORIGINAL POLICY EVIDENCE (for cross-reference):
{evidence_text}

REWRITE COUNT SO FAR: {rewrite_count}/{self.MAX_REWRITES}

{"⚠️ MAX REWRITES REACHED. If not passing, you MUST escalate." if rewrite_count >= self.MAX_REWRITES else ""}

Audit the draft and return your compliance JSON.
"""
        raw = call_llm(COMPLIANCE_SYSTEM_PROMPT, user_message, json_mode=True)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}

        action = data.get("action", "escalate")
        if rewrite_count >= self.MAX_REWRITES and action == "rewrite":
            action = "escalate"

        return ComplianceResult(
            passed=data.get("passed", False),
            issues=data.get("issues", []),
            action=action,
            citation_coverage_score=float(data.get("citation_coverage_score", 0.0)),
            unsupported_claim_count=int(data.get("unsupported_claim_count", 0)),
        )

    @property
    def rewrite_instructions(self) -> Optional[str]:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class SupportOrchestrator:
    """
    Orchestrates the four agents in sequence with rewrite/escalate loops.

    Flow:
      TicketContext
        → TriageAgent          (classify + retrieval queries)
        → PolicyRetrieverAgent (fetch evidence)
        → ResolutionWriterAgent (draft resolution)
        → ComplianceAgent      (audit; loop on rewrite, final escalate)
        → FinalOutput
    """

    def __init__(self, store: PolicyVectorStore, embedder: EmbeddingModel):
        self.triage = TriageAgent()
        self.retriever = PolicyRetrieverAgent(store, embedder)
        self.writer = ResolutionWriterAgent()
        self.compliance = ComplianceAgent()

    def process(self, ctx: TicketContext, ticket_id: str = "TICKET-001") -> FinalOutput:
        print(f"\n{'='*60}")
        print(f"Processing {ticket_id}")
        print(f"{'='*60}")

        # Step 1: Triage
        print("[1/4] Triage Agent running...")
        triage = self.triage.run(ctx)
        print(f"  Category: {triage.category.value} ({triage.confidence:.0%} confidence)")
        print(f"  Urgency: {triage.urgency} | Sentiment: {triage.sentiment}")
        if triage.clarifying_questions:
            print(f"  Clarifying Qs: {len(triage.clarifying_questions)}")

        # If critical info is missing and questions exist, pause for clarification
        if triage.clarifying_questions and triage.confidence < 0.6:
            print("  → Returning clarifying questions (low confidence)")
            return FinalOutput(
                ticket_id=ticket_id,
                category=triage.category,
                category_confidence=triage.confidence,
                clarifying_questions=triage.clarifying_questions,
                decision=Decision.NEEDS_INFO,
                rationale="Insufficient information to determine resolution.",
                citations=[],
                customer_response=(
                    "Thank you for contacting us. To assist you better, "
                    "could you please help us with the following:\n\n" +
                    "\n".join(f"• {q}" for q in triage.clarifying_questions)
                ),
                internal_notes="Low confidence classification. Awaiting customer clarification.",
                compliance_passed=False,
                rewrite_count=0,
                escalated=False,
                escalation_reason=None,
            )

        # Step 2: Retrieve
        print("[2/4] Policy Retriever running...")
        evidence = self.retriever.run(triage)
        print(f"  Retrieved {evidence.total_retrieved} unique chunks, using top {len(evidence.results)}")

        # Step 3 + 4: Write → Comply loop
        rewrite_count = 0
        draft = None
        compliance = None

        while True:
            # Step 3: Write
            print(f"[3/4] Resolution Writer running (attempt {rewrite_count + 1})...")
            draft = self.writer.run(ctx, triage, evidence)
            print(f"  Decision: {draft.decision.value}")
            print(f"  Citations: {draft.citations}")
            if draft.unsupported_claims:
                print(f"  ⚠️  Unsupported claims: {draft.unsupported_claims}")

            # Step 4: Comply
            print(f"[4/4] Compliance Agent auditing...")
            compliance = self.compliance.run(draft, evidence, rewrite_count)
            print(f"  Action: {compliance.action} | Coverage: {compliance.citation_coverage_score:.0%}")

            if compliance.action == "approved":
                print("  ✅ Compliance PASSED")
                break
            elif compliance.action == "rewrite" and rewrite_count < ComplianceAgent.MAX_REWRITES:
                print(f"  🔄 Rewrite requested ({rewrite_count + 1}/{ComplianceAgent.MAX_REWRITES})")
                rewrite_count += 1
            else:
                print("  🚨 ESCALATING")
                break

        escalated = compliance.action == "escalate"
        escalation_reason = None
        if escalated:
            escalation_reason = (
                "; ".join(compliance.issues) or
                "Compliance checks failed after max rewrites."
            )

        return FinalOutput(
            ticket_id=ticket_id,
            category=triage.category,
            category_confidence=triage.confidence,
            clarifying_questions=triage.clarifying_questions,
            decision=draft.decision if not escalated else Decision.ESCALATE,
            rationale=draft.rationale,
            citations=draft.citations,
            customer_response=draft.customer_response if not escalated else (
                "We're escalating your case to a senior specialist who will "
                "reach out within 1 business day to fully resolve your issue."
            ),
            internal_notes=(
                draft.internal_notes +
                (f"\n[ESCALATED] {escalation_reason}" if escalated else "")
            ),
            compliance_passed=compliance.passed,
            rewrite_count=rewrite_count,
            escalated=escalated,
            escalation_reason=escalation_reason,
        )

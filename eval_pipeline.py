"""
evaluation/eval_pipeline.py

Evaluation dataset: 20 test cases across four difficulty tiers.
  - 8 standard cases
  - 6 exception-heavy cases
  - 3 policy-conflict cases
  - 3 not-in-policy cases

Metrics:
  - Citation Coverage Rate    = citations / total_claims
  - Unsupported Claim Rate    = unsupported_claims / total_claims
  - Correct Escalation Rate   = correct_escalations / should_escalate_count
  - Correct Decision Rate     = correct_decisions / total_cases
  - Compliance Pass Rate      = compliance_passed / total_cases
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Test case definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    case_id: str
    tier: str          # standard | exception | conflict | not_in_policy
    description: str
    ticket_text: str
    order_data: dict
    customer_tier: str
    expected_decision: str     # approve | deny | partial | escalate | needs_info
    should_escalate: bool
    expected_citations_contain: list[str]    # doc IDs that MUST appear
    notes: str                               # evaluator notes


EVAL_CASES: list[EvalCase] = [

    # ── STANDARD CASES (8) ───────────────────────────────────────────────────

    EvalCase(
        case_id="STD-001",
        tier="standard",
        description="Standard return within 30-day window",
        ticket_text=(
            "Hi, I received my order last week and I'd like to return the blue jacket. "
            "It doesn't fit well. How do I start the return?"
        ),
        order_data={
            "order_id": "ORD-1001",
            "items": [{"name": "Blue Jacket", "sku": "JKT-BL-L", "price": 89.99}],
            "order_date": "2024-10-01",
            "delivery_date": "2024-10-08",
            "payment_method": "credit_card",
            "status": "delivered",
        },
        customer_tier="standard",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="Delivered Oct 8, today Oct 15 = 7 days. Well within 30-day window.",
    ),

    EvalCase(
        case_id="STD-002",
        tier="standard",
        description="Refund timeline inquiry",
        ticket_text=(
            "I returned my order 5 days ago and I still haven't received my refund. "
            "The return tracking shows it was delivered to your warehouse on Monday. "
            "When will I get my money back?"
        ),
        order_data={
            "order_id": "ORD-1002",
            "return_initiated": "2024-10-10",
            "return_delivered_to_warehouse": "2024-10-14",
            "payment_method": "credit_card",
            "refund_status": "pending_inspection",
        },
        customer_tier="standard",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="Return arrived. 5-10 business days from inspection. Inspection not yet complete.",
    ),

    EvalCase(
        case_id="STD-003",
        tier="standard",
        description="Lost package — carrier shows delivered but customer denies receipt",
        ticket_text=(
            "My tracking says delivered but I never received my package! "
            "I was home all day and there's nothing at my door. Order #ORD-1003."
        ),
        order_data={
            "order_id": "ORD-1003",
            "delivery_date": "2024-10-15",
            "carrier": "FedEx",
            "tracking_status": "Delivered",
            "tracking_number": "FX9876543210",
            "order_value": 120.00,
        },
        customer_tier="standard",
        expected_decision="escalate",   # investigation must be opened
        should_escalate=True,
        expected_citations_contain=["POL-SD-002"],
        notes="Lost package requires investigation per POL-SD-002 § 4.2. 3 business days wait first.",
    ),

    EvalCase(
        case_id="STD-004",
        tier="standard",
        description="Promo code not applied at checkout",
        ticket_text=(
            "I used promo code SAVE20 at checkout but the discount didn't show up "
            "in my order total. Can you apply it retroactively? Order ORD-1004."
        ),
        order_data={
            "order_id": "ORD-1004",
            "order_date": "2024-10-15",
            "status": "processing",
            "subtotal": 150.00,
            "discount_applied": 0,
            "promo_code_attempted": "SAVE20",
        },
        customer_tier="standard",
        expected_decision="deny",
        should_escalate=False,
        expected_citations_contain=["POL-PR-004"],
        notes="POL-PR-004 § 1.1 states codes cannot be applied retroactively.",
    ),

    EvalCase(
        case_id="STD-005",
        tier="standard",
        description="Cancellation request — order still processing",
        ticket_text=(
            "I just placed order ORD-1005 10 minutes ago and I made a mistake with "
            "the size. Can I cancel and reorder?"
        ),
        order_data={
            "order_id": "ORD-1005",
            "order_date": "2024-10-15T14:30:00",
            "current_time": "2024-10-15T14:42:00",
            "status": "processing",
        },
        customer_tier="standard",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-CX-003"],
        notes="Within 1-hour cancellation window. Status still processing.",
    ),

    EvalCase(
        case_id="STD-006",
        tier="standard",
        description="Price adjustment request within 7-day window",
        ticket_text=(
            "I bought a lamp for $120 on October 10th. I see it's now on sale for $90. "
            "Can I get a price adjustment? Order ORD-1006."
        ),
        order_data={
            "order_id": "ORD-1006",
            "purchase_date": "2024-10-10",
            "current_date": "2024-10-15",
            "item": "Designer Table Lamp",
            "price_paid": 120.00,
            "current_price": 90.00,
        },
        customer_tier="standard",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-PR-004"],
        notes="5 days since purchase, within 7-day window. Adjustment as store credit per § 2.1.",
    ),

    EvalCase(
        case_id="STD-007",
        tier="standard",
        description="Defective item — product stopped working",
        ticket_text=(
            "My blender (ORD-1007) stopped working after 2 weeks of normal use. "
            "The motor just stopped. It's clearly a defect. I have a video of it. "
            "I want a replacement."
        ),
        order_data={
            "order_id": "ORD-1007",
            "item": "Pro Blender 2000",
            "delivery_date": "2024-09-20",
            "current_date": "2024-10-15",
            "days_since_delivery": 25,
            "price": 189.99,
        },
        customer_tier="standard",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="Defective within 90-day window. Prepaid label required. Replacement at customer choice.",
    ),

    EvalCase(
        case_id="STD-008",
        tier="standard",
        description="International return — who pays shipping?",
        ticket_text=(
            "I'm in Canada and I need to return a dress from order ORD-1008. "
            "It's not defective, just not my style. Who pays for return shipping?"
        ),
        order_data={
            "order_id": "ORD-1008",
            "customer_country": "Canada",
            "item": "Summer Dress",
            "delivery_date": "2024-10-01",
            "current_date": "2024-10-15",
            "reason": "not to taste",
        },
        customer_tier="standard",
        expected_decision="partial",
        should_escalate=False,
        expected_citations_contain=["POL-SD-002"],
        notes="International customers pay return shipping for non-defective returns per POL-SD-002 § 5.4.",
    ),

    # ── EXCEPTION-HEAVY CASES (6) ─────────────────────────────────────────────

    EvalCase(
        case_id="EXC-001",
        tier="exception",
        description="Hygiene item — opened lipstick return request",
        ticket_text=(
            "I bought a luxury lipstick (ORD-2001) and after trying it on once it's "
            "completely the wrong shade. I want to return it for a refund. "
            "I only used it once to test the colour."
        ),
        order_data={
            "order_id": "ORD-2001",
            "item": "Matte Lipstick - Velvet Red",
            "category": "cosmetics",
            "seal_status": "opened",
            "delivery_date": "2024-10-12",
            "price": 34.99,
        },
        customer_tier="standard",
        expected_decision="deny",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="Opened cosmetics are non-returnable per POL-RR-001 § 4.2 (hygiene).",
    ),

    EvalCase(
        case_id="EXC-002",
        tier="exception",
        description="Final sale item — change of mind",
        ticket_text=(
            "I want to return the Final Sale jacket from order ORD-2002. "
            "I realise now the colour is not right for me."
        ),
        order_data={
            "order_id": "ORD-2002",
            "item": "Winter Jacket",
            "sale_type": "final_sale",
            "delivery_date": "2024-10-10",
            "price": 59.99,
        },
        customer_tier="gold",
        expected_decision="deny",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="Final Sale is non-returnable per § 4.1. Gold tier doesn't override Final Sale.",
    ),

    EvalCase(
        case_id="EXC-003",
        tier="exception",
        description="Perishable — flowers arrived wilted",
        ticket_text=(
            "The flowers I ordered for my mother's birthday arrived completely wilted "
            "and brown. They are totally unusable. Order ORD-2003 delivered yesterday."
        ),
        order_data={
            "order_id": "ORD-2003",
            "item": "Birthday Bouquet - Mixed",
            "category": "perishable/flowers",
            "delivery_date": "2024-10-14",
            "current_date": "2024-10-15",
            "hours_since_delivery": 20,
            "price": 65.00,
        },
        customer_tier="standard",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="Perishable reported within 48 hours. Photos required. Store credit or replacement per § 4.3.",
    ),

    EvalCase(
        case_id="EXC-004",
        tier="exception",
        description="Return request 45 days post-delivery — standard customer",
        ticket_text=(
            "I know it's been a while but I really need to return this shirt from "
            "ORD-2004. It's in perfect condition with tags still on. Can you help?"
        ),
        order_data={
            "order_id": "ORD-2004",
            "item": "Oxford Button Shirt",
            "delivery_date": "2024-09-01",
            "current_date": "2024-10-15",
            "days_since_delivery": 44,
            "condition": "unused_with_tags",
        },
        customer_tier="standard",
        expected_decision="deny",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="44 days > 30-day window for standard customers. Not holiday period. Deny unless agent exception.",
    ),

    EvalCase(
        case_id="EXC-005",
        tier="exception",
        description="Software license key — non-functional, never activated",
        ticket_text=(
            "I purchased a software license key (ORD-2005) but when I try to activate it "
            "I get an error saying the key is invalid. I've never used it. I want a refund."
        ),
        order_data={
            "order_id": "ORD-2005",
            "item": "Photo Editing Pro - License Key",
            "category": "digital/software",
            "activation_status": "never_activated",
            "error_reported": "invalid_key",
            "delivery_date": "2024-10-13",
        },
        customer_tier="standard",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="Digital non-refundable once activated. Key was never activated. Non-functional → replacement key per § 4.4.",
    ),

    EvalCase(
        case_id="EXC-006",
        tier="exception",
        description="Platinum member — 55-day return request",
        ticket_text=(
            "Hi, I'm a Platinum member. I have a pair of boots from 55 days ago that "
            "unfortunately gave me blisters. They're unworn beyond the initial try. "
            "Order ORD-2006. Can I still return?"
        ),
        order_data={
            "order_id": "ORD-2006",
            "item": "Leather Ankle Boots",
            "delivery_date": "2024-08-21",
            "current_date": "2024-10-15",
            "days_since_delivery": 55,
            "customer_tier": "platinum",
        },
        customer_tier="platinum",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001"],
        notes="Platinum members get 60-day window per § 1.2. 55 days < 60. Approve.",
    ),

    # ── POLICY CONFLICT CASES (3) ──────────────────────────────────────────────

    EvalCase(
        case_id="CON-001",
        tier="conflict",
        description="Holiday purchase + Platinum member — which window applies?",
        ticket_text=(
            "I'm a Platinum member and I bought a coat on December 5th (ORD-3001). "
            "Today is February 10th and I'd like to return it. It's still in perfect condition."
        ),
        order_data={
            "order_id": "ORD-3001",
            "item": "Winter Coat",
            "purchase_date": "2024-12-05",
            "delivery_date": "2024-12-10",
            "return_request_date": "2025-02-10",
            "days_since_delivery": 62,
            "customer_tier": "platinum",
        },
        customer_tier="platinum",
        expected_decision="deny",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001", "POL-EX-006"],
        notes=(
            "Holiday window: Jan 31. Platinum window: 60 days from Dec 10 = Feb 8. "
            "Both windows expired by Feb 10. POL-EX-006 § 1.2 says take the LATER deadline. "
            "Feb 8 > Jan 31. Feb 10 > Feb 8. DENY."
        ),
    ),

    EvalCase(
        case_id="CON-002",
        tier="conflict",
        description="Defective Final Sale item — which policy wins?",
        ticket_text=(
            "The Final Sale blender I ordered (ORD-3002) caught fire during normal use! "
            "Clearly a manufacturing defect and dangerous. I need a full refund immediately."
        ),
        order_data={
            "order_id": "ORD-3002",
            "item": "Compact Blender",
            "sale_type": "final_sale",
            "delivery_date": "2024-09-15",
            "incident_date": "2024-10-14",
            "defect_description": "Caught fire during normal operation",
            "safety_incident": True,
        },
        customer_tier="standard",
        expected_decision="approve",
        should_escalate=False,
        expected_citations_contain=["POL-RR-001", "POL-EX-006"],
        notes=(
            "Final Sale normally non-returnable (§ 4.1). BUT defective item exception explicitly "
            "supersedes Final Sale restrictions per § 4.1 ('except confirmed manufacturer defects'). "
            "POL-EX-006 § 1.1 confirms defective policy outranks Final Sale. APPROVE full refund."
        ),
    ),

    EvalCase(
        case_id="CON-003",
        tier="conflict",
        description="Promo code expired due to documented system outage",
        ticket_text=(
            "I tried to use my promo code FIRST30 during the sale on Oct 10th but "
            "your website was down for 4 hours (I have screenshots). By the time it came back "
            "the code had expired. Surely you can honour it given it was your system's fault?"
        ),
        order_data={
            "order_id": "ORD-3003",
            "promo_code": "FIRST30",
            "code_expiry": "2024-10-10",
            "outage_confirmed": True,
            "outage_duration_hours": 4,
            "outage_start": "2024-10-10T18:00:00",
            "outage_end": "2024-10-10T22:00:00",
            "order_date": "2024-10-10T22:30:00",
        },
        customer_tier="standard",
        expected_decision="escalate",
        should_escalate=True,
        expected_citations_contain=["POL-PR-004", "POL-EX-006"],
        notes=(
            "POL-PR-004 § 1.4 says agents cannot reactivate expired codes. BUT § 1.4 also says "
            "'if expired due to our system error (documented technical outage), an equivalent code "
            "may be issued at the discretion of the support manager.' Outage is confirmed. "
            "This requires manager-level exception. ESCALATE."
        ),
    ),

    # ── NOT-IN-POLICY CASES (3) ───────────────────────────────────────────────

    EvalCase(
        case_id="NIP-001",
        tier="not_in_policy",
        description="Customer demands price match with Amazon",
        ticket_text=(
            "Amazon has the exact same product I bought from you for $30 less. "
            "Order ORD-4001. I want you to match their price or I'll dispute the charge."
        ),
        order_data={
            "order_id": "ORD-4001",
            "item": "Wireless Headphones",
            "price_paid": 149.99,
            "competitor_price": 119.99,
            "competitor": "Amazon",
        },
        customer_tier="standard",
        expected_decision="escalate",
        should_escalate=True,
        expected_citations_contain=[],
        notes="Competitor price matching is not in any policy. Must escalate per POL-EX-006 § 4.2.",
    ),

    EvalCase(
        case_id="NIP-002",
        tier="not_in_policy",
        description="Customer wants direct exchange (not return + repurchase)",
        ticket_text=(
            "I bought size M jeans but need size L. Can you just swap them for me "
            "without me having to go through the full return process? Order ORD-4002."
        ),
        order_data={
            "order_id": "ORD-4002",
            "item": "Slim Fit Jeans - Size M",
            "delivery_date": "2024-10-10",
            "current_date": "2024-10-15",
            "reason": "wrong_size",
        },
        customer_tier="standard",
        expected_decision="escalate",
        should_escalate=True,
        expected_citations_contain=["POL-EX-006"],
        notes="Direct exchange not offered. Must return + repurchase. POL-EX-006 § 4.3 lists this as NIP.",
    ),

    EvalCase(
        case_id="NIP-003",
        tier="not_in_policy",
        description="Return of consumable part of non-consumable product separately",
        ticket_text=(
            "I bought a razor set (ORD-4003) but only need to return the blade heads, "
            "not the handle. The handle is fine. Can I return just the 5-pack of blade heads?"
        ),
        order_data={
            "order_id": "ORD-4003",
            "item": "Precision Razor Set (handle + 5 blade heads)",
            "bundle_type": "kit",
            "component_to_return": "blade_heads_only",
            "delivery_date": "2024-10-05",
        },
        customer_tier="standard",
        expected_decision="escalate",
        should_escalate=True,
        expected_citations_contain=["POL-EX-006"],
        notes="Returning consumable component of non-consumable product is explicitly listed as NIP per § 4.3.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    tier: str
    expected_decision: str
    actual_decision: str
    decision_correct: bool
    should_escalate: bool
    actually_escalated: bool
    escalation_correct: bool
    compliance_passed: bool
    citations_present: list[str]
    expected_citations_found: bool
    rewrite_count: int
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class EvalReport:
    total_cases: int
    correct_decisions: int
    correct_escalations: int
    should_escalate_count: int
    compliance_pass_count: int
    citation_coverage_rate: float      # how often expected doc IDs appeared
    correct_decision_rate: float
    correct_escalation_rate: float
    compliance_pass_rate: float
    avg_latency_seconds: float
    avg_rewrites: float
    tier_breakdown: dict
    case_results: list[CaseResult]


def run_evaluation(orchestrator, cases: list[EvalCase] = None) -> EvalReport:
    """Run all eval cases and compute metrics."""
    from agents.agents import TicketContext

    if cases is None:
        cases = EVAL_CASES

    results: list[CaseResult] = []

    for case in cases:
        print(f"\n{'─'*50}")
        print(f"Eval case: {case.case_id} [{case.tier}] — {case.description}")

        ctx = TicketContext(
            ticket_text=case.ticket_text,
            order_data=case.order_data,
            customer_tier=case.customer_tier,
        )

        t0 = time.time()
        error = None
        output = None
        try:
            output = orchestrator.process(ctx, ticket_id=case.case_id)
        except Exception as e:
            error = str(e)
            print(f"  ERROR: {e}")
        latency = time.time() - t0

        if output is None:
            results.append(CaseResult(
                case_id=case.case_id, tier=case.tier,
                expected_decision=case.expected_decision, actual_decision="error",
                decision_correct=False, should_escalate=case.should_escalate,
                actually_escalated=False, escalation_correct=False,
                compliance_passed=False, citations_present=[],
                expected_citations_found=False, rewrite_count=0,
                latency_seconds=latency, error=error,
            ))
            continue

        actual_decision = output.decision.value
        decision_correct = actual_decision == case.expected_decision
        actually_escalated = output.escalated
        escalation_correct = (case.should_escalate == actually_escalated)

        # Check citation coverage
        cit_str = " ".join(output.citations)
        expected_found = all(
            doc_id in cit_str for doc_id in case.expected_citations_contain
        )

        results.append(CaseResult(
            case_id=case.case_id,
            tier=case.tier,
            expected_decision=case.expected_decision,
            actual_decision=actual_decision,
            decision_correct=decision_correct,
            should_escalate=case.should_escalate,
            actually_escalated=actually_escalated,
            escalation_correct=escalation_correct,
            compliance_passed=output.compliance_passed,
            citations_present=output.citations,
            expected_citations_found=expected_found,
            rewrite_count=output.rewrite_count,
            latency_seconds=latency,
        ))

        print(f"  Decision: {actual_decision} (expected: {case.expected_decision}) {'✅' if decision_correct else '❌'}")
        print(f"  Escalated: {actually_escalated} (expected: {case.should_escalate}) {'✅' if escalation_correct else '❌'}")
        print(f"  Compliance: {'✅' if output.compliance_passed else '❌'} | Rewrites: {output.rewrite_count}")
        print(f"  Citations found: {'✅' if expected_found else '❌'} — {output.citations}")
        print(f"  Latency: {latency:.1f}s")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    total = len(results)
    correct_decisions = sum(r.decision_correct for r in results)
    should_esc = sum(r.should_escalate for r in results)
    correct_esc = sum(r.escalation_correct for r in results if r.should_escalate)
    compliance_pass = sum(r.compliance_passed for r in results)
    citations_ok = sum(r.expected_citations_found for r in results)
    avg_lat = sum(r.latency_seconds for r in results) / total if total else 0
    avg_rew = sum(r.rewrite_count for r in results) / total if total else 0

    tier_breakdown = {}
    for tier in ["standard", "exception", "conflict", "not_in_policy"]:
        tier_cases = [r for r in results if r.tier == tier]
        if tier_cases:
            tier_breakdown[tier] = {
                "total": len(tier_cases),
                "correct_decisions": sum(r.decision_correct for r in tier_cases),
                "accuracy": sum(r.decision_correct for r in tier_cases) / len(tier_cases),
            }

    report = EvalReport(
        total_cases=total,
        correct_decisions=correct_decisions,
        correct_escalations=correct_esc,
        should_escalate_count=should_esc,
        compliance_pass_count=compliance_pass,
        citation_coverage_rate=citations_ok / total if total else 0,
        correct_decision_rate=correct_decisions / total if total else 0,
        correct_escalation_rate=correct_esc / should_esc if should_esc else 1.0,
        compliance_pass_rate=compliance_pass / total if total else 0,
        avg_latency_seconds=avg_lat,
        avg_rewrites=avg_rew,
        tier_breakdown=tier_breakdown,
        case_results=results,
    )

    _print_report(report)
    return report


def _print_report(report: EvalReport):
    print(f"\n{'='*60}")
    print("EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Total cases:              {report.total_cases}")
    print(f"Correct Decision Rate:    {report.correct_decision_rate:.1%}  ({report.correct_decisions}/{report.total_cases})")
    print(f"Correct Escalation Rate:  {report.correct_escalation_rate:.1%}  ({report.correct_escalations}/{report.should_escalate_count})")
    print(f"Compliance Pass Rate:     {report.compliance_pass_rate:.1%}  ({report.compliance_pass_count}/{report.total_cases})")
    print(f"Citation Coverage Rate:   {report.citation_coverage_rate:.1%}")
    print(f"Avg Latency:              {report.avg_latency_seconds:.1f}s")
    print(f"Avg Rewrites per Case:    {report.avg_rewrites:.2f}")
    print(f"\nTier Breakdown:")
    for tier, stats in report.tier_breakdown.items():
        print(f"  {tier:20s}: {stats['correct_decisions']}/{stats['total']} correct ({stats['accuracy']:.0%})")
    print(f"{'='*60}")


def save_report(report: EvalReport, path: str = "outputs/eval_report.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "summary": {
            "total_cases": report.total_cases,
            "correct_decision_rate": report.correct_decision_rate,
            "correct_escalation_rate": report.correct_escalation_rate,
            "compliance_pass_rate": report.compliance_pass_rate,
            "citation_coverage_rate": report.citation_coverage_rate,
            "avg_latency_seconds": report.avg_latency_seconds,
            "tier_breakdown": report.tier_breakdown,
        },
        "cases": [
            {
                "case_id": r.case_id,
                "tier": r.tier,
                "decision_correct": r.decision_correct,
                "expected": r.expected_decision,
                "actual": r.actual_decision,
                "escalation_correct": r.escalation_correct,
                "compliance_passed": r.compliance_passed,
                "citations": r.citations_present,
                "latency": round(r.latency_seconds, 2),
            }
            for r in report.case_results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[Eval] Report saved to {path}")

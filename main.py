"""
main.py — Entry point for the e-commerce support RAG system.

Usage:
  python main.py --mode ingest          # Build vector store from policies
  python main.py --mode demo            # Run 3 sample tickets
  python main.py --mode eval            # Run full 20-case evaluation
  python main.py --mode ticket          # Process a single ticket interactively
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import IngestionPipeline, PolicyVectorStore, EmbeddingModel
from agents.agents import SupportOrchestrator, TicketContext, FinalOutput


# ─────────────────────────────────────────────────────────────────────────────
# Sample tickets for demo mode
# ─────────────────────────────────────────────────────────────────────────────

DEMO_TICKETS = [
    {
        "id": "DEMO-001",
        "description": "Standard return — within window",
        "ticket": (
            "Hi, I received my order (ORD-DEMO-1) last week and I'd like to return "
            "the blue jacket. It doesn't fit well. How do I start the return? "
            "I paid with my Visa."
        ),
        "order": {
            "order_id": "ORD-DEMO-1",
            "items": [{"name": "Blue Jacket", "price": 89.99}],
            "order_date": "2024-10-01",
            "delivery_date": "2024-10-08",
            "payment_method": "credit_card",
            "status": "delivered",
        },
        "customer_tier": "standard",
    },
    {
        "id": "DEMO-002",
        "description": "Defective Final Sale item — conflict scenario",
        "ticket": (
            "The Final Sale blender I ordered (ORD-DEMO-2) sparked and stopped working "
            "after my very first use. I'm worried it's a fire hazard. "
            "I know it's final sale but this is clearly defective. I need a full refund."
        ),
        "order": {
            "order_id": "ORD-DEMO-2",
            "item": "Compact Blender",
            "sale_type": "final_sale",
            "delivery_date": "2024-09-25",
            "defect_description": "Sparked and stopped working on first use",
            "price": 49.99,
        },
        "customer_tier": "standard",
    },
    {
        "id": "DEMO-003",
        "description": "Not-in-policy: competitor price match",
        "ticket": (
            "Hi, I just saw the exact same headphones I bought from you (ORD-DEMO-3) "
            "for $40 cheaper on BestBuy. Do you do price matching? Order was placed 3 days ago."
        ),
        "order": {
            "order_id": "ORD-DEMO-3",
            "item": "Wireless Headphones Pro",
            "price_paid": 199.99,
            "competitor_price": 159.99,
            "competitor": "BestBuy",
            "order_date": "2024-10-12",
        },
        "customer_tier": "gold",
    },
]


def format_final_output(output: FinalOutput) -> str:
    """Format FinalOutput as a readable ticket resolution report."""
    sep = "=" * 65
    thin = "-" * 65
    lines = [
        sep,
        f"  TICKET RESOLUTION: {output.ticket_id}",
        sep,
        "",
        f"1. CLASSIFICATION",
        f"   Category:    {output.category.value.upper()}",
        f"   Confidence:  {output.category_confidence:.0%}",
        "",
    ]

    if output.clarifying_questions:
        lines += [
            f"2. CLARIFYING QUESTIONS",
            *[f"   • {q}" for q in output.clarifying_questions],
            "",
        ]
    else:
        lines += ["2. CLARIFYING QUESTIONS: None needed", ""]

    lines += [
        f"3. DECISION: {output.decision.value.upper()}",
        "",
        f"4. RATIONALE (policy-based)",
        f"   {output.rationale}",
        "",
        f"5. CITATIONS",
    ]

    if output.citations:
        lines += [f"   {c}" for c in output.citations]
    else:
        lines += ["   ⚠️  No citations — review required" if not output.escalated else "   (Escalated — citations pending senior review)"]

    lines += [
        "",
        f"6. CUSTOMER RESPONSE DRAFT",
        thin,
        output.customer_response,
        thin,
        "",
        f"7. INTERNAL NOTES",
        f"   {output.internal_notes}",
        f"   Compliance: {'✅ PASSED' if output.compliance_passed else '❌ FAILED'}",
        f"   Rewrites:   {output.rewrite_count}",
        f"   Escalated:  {'YES — ' + (output.escalation_reason or 'reason unknown') if output.escalated else 'No'}",
        "",
        sep,
    ]
    return "\n".join(lines)


def build_orchestrator(store_path: str = "data/vector_store") -> SupportOrchestrator:
    """Load or build the vector store and return an orchestrator."""
    meta_path = store_path + ".meta.json"

    if Path(meta_path).exists():
        print(f"[main] Loading existing vector store from {store_path}...")
        embedder = EmbeddingModel()
        store = PolicyVectorStore.load(store_path)
    else:
        print(f"[main] No store found at {store_path}. Running ingestion...")
        pipeline = IngestionPipeline(
            policy_dir="policies/",
            store_path=store_path,
        )
        store = pipeline.ingest()
        embedder = pipeline.embedder

    return SupportOrchestrator(store=store, embedder=embedder)


# ─────────────────────────────────────────────────────────────────────────────
# Mode runners
# ─────────────────────────────────────────────────────────────────────────────

def run_ingest():
    print("[mode: ingest] Building vector store from policy documents...")
    pipeline = IngestionPipeline(
        policy_dir="policies/",
        store_path="data/vector_store",
    )
    store = pipeline.ingest()
    print(f"[ingest] Complete. {len(store.chunks)} chunks indexed.")


def run_demo():
    print("[mode: demo] Running 3 sample tickets...\n")
    orchestrator = build_orchestrator()

    Path("outputs").mkdir(exist_ok=True)
    all_outputs = []

    for ticket_def in DEMO_TICKETS:
        print(f"\nProcessing: {ticket_def['id']} — {ticket_def['description']}")
        ctx = TicketContext(
            ticket_text=ticket_def["ticket"],
            order_data=ticket_def["order"],
            customer_tier=ticket_def["customer_tier"],
        )
        output = orchestrator.process(ctx, ticket_id=ticket_def["id"])
        formatted = format_final_output(output)
        print(formatted)
        all_outputs.append(formatted)

    # Save sample outputs
    with open("outputs/sample_outputs.txt", "w") as f:
        f.write("\n\n".join(all_outputs))
    print("\n[demo] Sample outputs saved to outputs/sample_outputs.txt")


def run_eval():
    print("[mode: eval] Running full 20-case evaluation...\n")
    from evaluation.eval_pipeline import run_evaluation, save_report

    orchestrator = build_orchestrator()
    report = run_evaluation(orchestrator)
    save_report(report, "outputs/eval_report.json")


def run_single_ticket(ticket_text: str, order_json: str, tier: str = "standard"):
    """Process a single ticket from command line."""
    orchestrator = build_orchestrator()
    try:
        order_data = json.loads(order_json)
    except json.JSONDecodeError:
        print("[ERROR] Invalid order JSON. Using empty order context.")
        order_data = {}

    ctx = TicketContext(
        ticket_text=ticket_text,
        order_data=order_data,
        customer_tier=tier,
    )
    output = orchestrator.process(ctx, ticket_id="CLI-TICKET")
    print(format_final_output(output))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Commerce Support Multi-Agent RAG System")
    parser.add_argument(
        "--mode",
        choices=["ingest", "demo", "eval", "ticket"],
        default="demo",
        help="Run mode",
    )
    parser.add_argument("--ticket", type=str, help="Ticket text (for --mode ticket)")
    parser.add_argument("--order", type=str, default="{}", help="Order JSON string")
    parser.add_argument("--tier", type=str, default="standard", help="Customer tier")

    args = parser.parse_args()

    if args.mode == "ingest":
        run_ingest()
    elif args.mode == "demo":
        run_demo()
    elif args.mode == "eval":
        run_eval()
    elif args.mode == "ticket":
        if not args.ticket:
            print("[ERROR] --ticket text is required for mode=ticket")
            sys.exit(1)
        run_single_ticket(args.ticket, args.order, args.tier)

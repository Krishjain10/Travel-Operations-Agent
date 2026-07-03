"""
CaseClose — Autonomous Travel Ops Ticket Resolution Agent

Entrypoint. Processes support tickets end-to-end to a terminal state
(RESOLVED_REFUND / RESOLVED_VOUCHER / RESOLVED_REBOOK / ESCALATED).

Usage:
    python main.py --ticket TKT-001           # Process single ticket
    python main.py --all                       # Process all tickets
    python main.py --all --verbose             # Show full reasoning
    python main.py --ticket TKT-002 --verbose  # Single ticket, verbose
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.loop import process_ticket


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_tickets() -> list[dict]:
    with open(DATA_DIR / "tickets.json", "r") as f:
        return json.load(f)["tickets"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="CaseClose — Autonomous Travel Ops Ticket Resolution Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticket TKT-001              Process a single ticket
  python main.py --ticket TKT-002 --verbose    Tradeoff case with full reasoning
  python main.py --all                          Process all tickets
  python main.py --all --verbose                Full batch with reasoning output
        """,
    )
    parser.add_argument(
        "--ticket", "-t",
        type=str,
        help="Process a single ticket by ID (e.g. TKT-001)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all tickets in data/tickets.json",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed reasoning and tool calls in terminal",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging (API calls, token usage)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate arguments
    if not args.ticket and not args.all:
        parser.print_help()
        print("\nError: specify --ticket <ID> or --all")
        sys.exit(1)

    # Load tickets
    tickets = load_tickets()
    ticket_map = {t["id"]: t for t in tickets}

    # Determine which tickets to process
    if args.ticket:
        if args.ticket not in ticket_map:
            print(f"Error: Ticket '{args.ticket}' not found.")
            print(f"Available tickets: {', '.join(ticket_map.keys())}")
            sys.exit(1)
        tickets_to_process = [ticket_map[args.ticket]]
    else:
        tickets_to_process = tickets

    # Process tickets
    print(f"\n{'+' + '='*68 + '+'}")
    print(f"{'|'} {'CaseClose - Autonomous Ticket Resolution':^66} {'|'}")
    print(f"{'+' + '='*68 + '+'}")
    print(f"\n  Processing {len(tickets_to_process)} ticket(s)...\n")

    results = []
    total_start = time.time()

    for ticket in tickets_to_process:
        try:
            audit = process_ticket(ticket, verbose=args.verbose)
            results.append(audit)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Partial results below.")
            break
        except Exception as e:
            print(f"\n  X Error processing {ticket['id']}: {e}")
            logging.exception("Failed to process ticket %s", ticket["id"])
            results.append({
                "ticket_id": ticket["id"],
                "decision": "ERROR",
                "reasoning": str(e),
                "iterations_used": 0,
                "elapsed_seconds": 0,
            })

    total_elapsed = round(time.time() - total_start, 2)

    # Print summary table
    _print_results_table(results, total_elapsed)


def _print_results_table(results: list[dict], total_elapsed: float):
    """Print a formatted summary table of all processed tickets."""
    print(f"\n{'='*70}")
    print(f"  {'RESULTS SUMMARY':^66}")
    print(f"{'='*70}")
    print(
        f"  {'Ticket':<10} {'Issue Type':<22} {'Decision':<20} "
        f"{'Iters':>5} {'Time':>6}"
    )
    print(f"  {'-'*10} {'-'*22} {'-'*20} {'-'*5} {'-'*6}")

    for r in results:
        ticket_id = r.get("ticket_id", "?")
        issue = r.get("issue_type", "?") or "?"
        decision = r.get("decision", "?") or "?"
        iters = r.get("iterations_used", 0)
        elapsed = r.get("elapsed_seconds", 0)

        # Color-code decisions for terminal output
        decision_display = _format_decision(decision)

        print(
            f"  {ticket_id:<10} {issue:<22} {decision_display:<20} "
            f"{iters:>5} {elapsed:>5.1f}s"
        )

    print(f"  {'-'*66}")
    print(f"  Total time: {total_elapsed}s")
    print(f"  Audit logs: output/case_logs/\n")


def _format_decision(decision: str) -> str:
    """Format decision string with a status indicator."""
    indicators = {
        "RESOLVED_REFUND":  "[OK] REFUND",
        "RESOLVED_VOUCHER": "[OK] VOUCHER",
        "RESOLVED_REBOOK":  "[OK] REBOOK",
        "ESCALATED":        "[>>] ESCALATED",
        "ERROR":            "[!!] ERROR",
    }
    return indicators.get(decision, decision)


if __name__ == "__main__":
    main()

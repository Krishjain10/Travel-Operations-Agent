"""
Core ReAct loop for CaseClose.

Processes a single ticket end-to-end:
  classify → gather → compute → decide → act

The LLM drives tool selection. This module manages the conversation,
dispatches tool calls, and enforces the iteration cap.
"""

import json
import time
import logging

from agent import llm as llm_module
from agent.tools import TOOL_SCHEMAS, execute_tool
from agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger("caseclose.loop")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 6  # Hard cap — forced escalation if reached


def process_ticket(ticket: dict, verbose: bool = False) -> dict:
    """
    Process a single support ticket end-to-end to a terminal state.

    The LLM reads the ticket, decides which tools to call and in what order,
    reasons over the results, and finalizes the case. This function manages
    the conversation loop and enforces the iteration cap.

    Args:
        ticket:  A ticket record from tickets.json
        verbose: If True, print detailed reasoning to stdout

    Returns:
        Audit trace dict (also written to disk by finalize_case)
    """
    ticket_id = ticket["id"]
    start_time = time.time()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Processing: {ticket_id} - {ticket['subject']}")
        print(f"{'='*70}")

    # Initialize conversation with system prompt + the raw ticket
    messages = [
        llm_module.system_message(SYSTEM_PROMPT),
        llm_module.user_message(
            f"Process this support ticket to a terminal state.\n\n"
            f"**Ticket ID:** {ticket['id']}\n"
            f"**Subject:** {ticket['subject']}\n"
            f"**Channel:** {ticket.get('channel', 'unknown')}\n"
            f"**Submitted:** {ticket.get('submitted_at', 'unknown')}\n\n"
            f"**Customer message:**\n{ticket['body']}"
        ),
    ]

    # Track what the agent does for the audit trail
    audit = {
        "ticket_id": ticket_id,
        "issue_type": None,
        "facts_gathered": [],
        "refund_calculation": None,
        "reasoning": None,
        "decision": None,
        "customer_response": None,
        "policy_cited": None,
        "iterations_used": 0,
        "tools_called": [],
        "elapsed_seconds": 0,
    }

    # --- Main ReAct loop ---
    for iteration in range(1, MAX_ITERATIONS + 1):
        audit["iterations_used"] = iteration

        if verbose:
            print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        # Call the LLM
        response = llm_module.chat(
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        # --- Handle tool calls ---
        if response["tool_calls"]:
            # Append the assistant message (with tool_calls) to conversation
            # We need the raw message object for proper conversation threading
            raw_msg = response["raw_message"]
            messages.append(_serialize_assistant_message(raw_msg))

            # Track tool names this turn for terminal-state check
            tool_names_this_turn = [tc["name"] for tc in response["tool_calls"]]

            if verbose:
                print(f"  Tools called: {', '.join(tool_names_this_turn)}")

            # Execute each tool call and append results
            for tc in response["tool_calls"]:
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                tool_id = tc["id"]

                if verbose:
                    print(f"    -> {tool_name}({_truncate_args(tool_args)})")

                # Execute the tool
                result = execute_tool(tool_name, tool_args)

                # Append tool result to conversation
                messages.append(
                    llm_module.tool_result_message(tool_id, json.dumps(result, default=str))
                )

                # Track in audit
                audit["tools_called"].append({
                    "tool": tool_name,
                    "args_summary": _truncate_args(tool_args),
                    "iteration": iteration,
                })
                audit["facts_gathered"].append(
                    f"[Iteration {iteration}] {tool_name}: {_summarize_result(tool_name, result)}"
                )

                # Capture specific results for the audit trace
                _update_audit_from_tool(audit, tool_name, tool_args, result)

                if verbose and tool_name == "finalize_case":
                    print(f"\n  [OK] Case finalized: {result.get('decision', '?')}")

            # ---------------------------------------------------------------
            # TERMINAL STATE CHECK
            # Fixed: check by tool NAME, not by the trailing result value.
            # If the LLM calls get_policy + calculate_refund + finalize_case
            # in one turn, we detect finalize_case in the tool names list,
            # not by checking what the last tool returned.
            # ---------------------------------------------------------------
            if "finalize_case" in tool_names_this_turn:
                audit["elapsed_seconds"] = round(time.time() - start_time, 2)
                if verbose:
                    _print_summary(audit)
                return audit

        # --- Handle text response (reasoning step, no tool calls) ---
        elif response["content"]:
            messages.append(llm_module.assistant_message(response["content"]))

            if verbose:
                reasoning_preview = response["content"][:300]
                print(f"  LLM reasoning: {reasoning_preview}...")

            # Capture reasoning if it looks like a decision
            if any(kw in response["content"].upper() for kw in
                   ["RESOLVED_", "ESCALATED", "MY DECISION", "I CHOOSE", "I RECOMMEND"]):
                audit["reasoning"] = response["content"]

    # --- Hard cap reached -- force escalation ---
    if verbose:
        print(f"\n  [!] Iteration cap ({MAX_ITERATIONS}) reached. Force-escalating.")

    audit = _force_escalate(ticket, audit, messages)
    audit["elapsed_seconds"] = round(time.time() - start_time, 2)

    if verbose:
        _print_summary(audit)

    return audit


# ---------------------------------------------------------------------------
# Force escalation when iteration cap is reached
# ---------------------------------------------------------------------------

def _force_escalate(ticket: dict, audit: dict, messages: list) -> dict:
    """
    Force-escalate a case that hit the iteration cap.

    Still produces a structured handoff — even a forced escalation
    should save human time by including whatever was gathered.
    """
    ticket_id = ticket["id"]

    # Call finalize_case directly with ESCALATED status
    result = execute_tool("finalize_case", {
        "ticket_id": ticket_id,
        "issue_type": audit.get("issue_type", "unknown"),
        "decision": "ESCALATED",
        "reasoning": (
            f"Forced escalation: agent reached {MAX_ITERATIONS}-iteration cap "
            f"without reaching a terminal state. Partial investigation gathered: "
            f"{len(audit['facts_gathered'])} facts. "
            f"Tools called: {[tc['tool'] for tc in audit['tools_called']]}."
        ),
        "customer_response": (
            f"Dear Customer,\n\n"
            f"Thank you for your patience. Your case ({ticket_id}) requires "
            f"additional review by our specialist team. We have gathered the "
            f"relevant details and a team member will follow up with you "
            f"within 2 business hours.\n\n"
            f"We apologize for any inconvenience.\n\n"
            f"Best regards,\nTravel Support Team"
        ),
        "policy_cited": "N/A -- escalated for human review",
        "facts_gathered": [f.split("] ", 1)[-1] if "] " in f else f
                          for f in audit.get("facts_gathered", [])],
    })

    audit["decision"] = "ESCALATED"
    audit["reasoning"] = f"Forced escalation at iteration cap ({MAX_ITERATIONS})"
    return audit


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def _update_audit_from_tool(audit: dict, tool_name: str, args: dict, result: dict):
    """Extract specific data from tool results into the audit trace."""
    if tool_name == "calculate_refund" and "refund_calculation" in result:
        audit["refund_calculation"] = result["refund_calculation"]

    elif tool_name == "finalize_case":
        audit["decision"] = args.get("decision")
        audit["issue_type"] = args.get("issue_type")
        audit["reasoning"] = args.get("reasoning")
        audit["customer_response"] = args.get("customer_response")
        audit["policy_cited"] = args.get("policy_cited")

    elif tool_name == "draft_response" and "customer_response" in result:
        audit["customer_response"] = result["customer_response"]

    elif tool_name == "get_policy" and "policy" in result:
        audit["policy_cited"] = result["policy"]["policy_clause"]


def _serialize_assistant_message(raw_message) -> dict:
    """
    Convert the raw Groq message object into a dict suitable for
    appending to the messages list (preserving tool_calls).
    """
    msg = {"role": "assistant"}

    if raw_message.content:
        msg["content"] = raw_message.content
    else:
        msg["content"] = ""

    if raw_message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in raw_message.tool_calls
        ]

    return msg


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _truncate_args(args: dict) -> str:
    """Truncate tool arguments for display."""
    s = json.dumps(args, default=str)
    return s[:120] + "..." if len(s) > 120 else s


def _summarize_result(tool_name: str, result: dict) -> str:
    """One-line summary of a tool result for the audit trace."""
    if "error" in result:
        return f"ERROR: {result['error']}"

    if tool_name == "get_booking" and "booking" in result:
        b = result["booking"]
        return (
            f"Booking {b['booking_id']}: {b['flight']['airline']} "
            f"{b['flight']['flight_number']} {b['flight']['origin']}→"
            f"{b['flight']['destination']}, {b['fare_class']}, "
            f"${b['total_paid']}, status={b['flight']['status']}"
        )

    if tool_name == "get_policy" and "policy" in result:
        p = result["policy"]
        return (
            f"Policy: {p['refund_percent']}% cash / {p['voucher_percent']}% voucher, "
            f"change_fee=${p['change_fee']}, auto_max=${p['auto_approve_max']}"
        )

    if tool_name == "calculate_refund" and "refund_calculation" in result:
        r = result["refund_calculation"]
        return (
            f"Cash=${r['cash_refund_amount']}, Voucher=${r['voucher_equivalent_amount']}, "
            f"BizCost: cash=${r['cost_to_business_cash']} vs voucher=${r['cost_to_business_voucher']}, "
            f"exceeds_threshold={r['exceeds_auto_approve']}"
        )

    if tool_name == "check_itinerary_conflict" and "itinerary_check" in result:
        c = result["itinerary_check"]
        return f"Conflict={c['conflict']}: {c['detail'][:100]}"

    if tool_name == "estimate_urgency":
        return f"Priority={result.get('priority', '?')}: {result.get('reason', '')[:80]}"

    if tool_name == "search_alternate_flights":
        n = result.get("result_count", 0)
        return f"Found {n} alternate flight(s)"

    if tool_name == "finalize_case":
        return f"FINALIZED as {result.get('decision', '?')}"

    if tool_name == "draft_response":
        resp = result.get("customer_response", "")
        return f"Response drafted ({len(resp)} chars)"

    return json.dumps(result, default=str)[:100]


def _print_summary(audit: dict):
    """Print a compact summary of the processed ticket."""
    print(f"\n{'-'*50}")
    print(f"  Ticket:     {audit['ticket_id']}")
    print(f"  Issue:      {audit.get('issue_type', '?')}")
    print(f"  Decision:   {audit.get('decision', '?')}")
    print(f"  Iterations: {audit['iterations_used']}")
    print(f"  Time:       {audit.get('elapsed_seconds', '?')}s")
    if audit.get("refund_calculation"):
        rc = audit["refund_calculation"]
        print(f"  Cash:       ${rc.get('cash_refund_amount', '?')}")
        print(f"  Voucher:    ${rc.get('voucher_equivalent_amount', '?')}")
    print(f"{'-'*50}")

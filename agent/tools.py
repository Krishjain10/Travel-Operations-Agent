"""
Tool implementations and OpenAI-compatible function schemas for CaseClose.

Two parts:
  1. TOOL_SCHEMAS — sent to Groq so the LLM knows what tools exist
  2. execute_tool() — dispatcher that routes tool calls to implementations

The LLM decides which tools to call; this module executes them.
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime

from agent import business_logic
from agent import llm as llm_module
from agent.prompts import DRAFT_RESPONSE_SYSTEM

logger = logging.getLogger("caseclose.tools")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "case_logs"


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
_bookings_cache = None
_policies_cache = None


def _load_bookings() -> list[dict]:
    global _bookings_cache
    if _bookings_cache is None:
        with open(_DATA_DIR / "bookings.json", "r") as f:
            _bookings_cache = json.load(f)["bookings"]
    return _bookings_cache


def _load_policies() -> list[dict]:
    global _policies_cache
    if _policies_cache is None:
        with open(_DATA_DIR / "policies.json", "r") as f:
            _policies_cache = json.load(f)["policies"]
    return _policies_cache


# ===========================================================================
# TOOL SCHEMAS (OpenAI function-calling format, sent to Groq)
# ===========================================================================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": "Retrieve the full booking record for a ticket, including flight details, hotel reservation, passenger info, fare class, and total amount paid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket ID (e.g. 'TKT-001') to look up the associated booking.",
                    }
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy",
            "description": "Look up the applicable airline policy rule for a given airline, fare class, and cause. Returns refund percentages, voucher eligibility, change fees, and the specific policy clause text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "airline": {
                        "type": "string",
                        "description": "Airline name exactly as it appears in the booking (e.g. 'SkyWest Airlines', 'Pacific Air', 'Atlantic Airways').",
                    },
                    "fare_class": {
                        "type": "string",
                        "enum": ["economy", "premium_economy", "business", "first"],
                        "description": "Fare class of the booking.",
                    },
                    "cause": {
                        "type": "string",
                        "enum": [
                            "airline_cancellation",
                            "customer_cancellation",
                            "delay",
                            "missed_connection",
                            "schedule_change",
                        ],
                        "description": "The cause/reason for the support issue.",
                    },
                },
                "required": ["airline", "fare_class", "cause"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "DETERMINISTIC calculation of refund and voucher amounts based on a booking and policy. Returns exact dollar amounts for cash refund, voucher equivalent, change fee, and cost-to-business for both options. You MUST call this before making any financial decision — never compute amounts yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Ticket ID to calculate refund for. The booking and policy must have been retrieved first.",
                    },
                    "airline": {
                        "type": "string",
                        "description": "Airline name (must match a previously retrieved policy).",
                    },
                    "fare_class": {
                        "type": "string",
                        "description": "Fare class (must match a previously retrieved policy).",
                    },
                    "cause": {
                        "type": "string",
                        "description": "Cause type (must match a previously retrieved policy).",
                    },
                },
                "required": ["ticket_id", "airline", "fare_class", "cause"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_itinerary_conflict",
            "description": "Check if there are date/schedule conflicts between the flight and hotel in a booking. Detects orphaned hotel nights when flights are cancelled, rescheduled, or missed. Call this whenever a flight status is not 'CONFIRMED'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Ticket ID to check for itinerary conflicts.",
                    }
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_alternate_flights",
            "description": "Search for available alternate flights for rebooking. Use when the resolution involves rebooking the customer on a different flight (delays, missed connections, schedule changes, date changes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin airport code (e.g. 'LAX', 'JFK').",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination airport code (e.g. 'LHR', 'SFO').",
                    },
                    "near_date": {
                        "type": "string",
                        "description": "Target date for the alternate flight in YYYY-MM-DD format.",
                    },
                },
                "required": ["origin", "destination", "near_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_urgency",
            "description": "Assess the urgency/priority of a ticket based on its text content. Returns low/med/high priority with reasoning. Use this early in triage to inform resolution priority.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_text": {
                        "type": "string",
                        "description": "The raw ticket body text to analyze for urgency signals.",
                    }
                },
                "required": ["ticket_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_response",
            "description": "Generate a professional customer-facing response based on the resolution decision. The response will cite the specific policy clause. Call this after making your decision but before finalize_case.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Customer's name for the greeting.",
                    },
                    "issue_summary": {
                        "type": "string",
                        "description": "Brief summary of the customer's issue.",
                    },
                    "decision": {
                        "type": "string",
                        "enum": [
                            "RESOLVED_REFUND",
                            "RESOLVED_VOUCHER",
                            "RESOLVED_REBOOK",
                            "ESCALATED",
                        ],
                        "description": "The resolution decision.",
                    },
                    "decision_details": {
                        "type": "string",
                        "description": "Specific details: refund amount, voucher amount, rebooking flight info, or escalation reason.",
                    },
                    "policy_clause": {
                        "type": "string",
                        "description": "The exact policy clause text to cite in the response.",
                    },
                },
                "required": [
                    "customer_name",
                    "issue_summary",
                    "decision",
                    "decision_details",
                    "policy_clause",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_case",
            "description": "Write the case to a terminal state and generate the audit trace. This is the LAST tool you call — it closes the case. Call this exactly once per ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket ID being finalized.",
                    },
                    "issue_type": {
                        "type": "string",
                        "description": "Classified issue type (e.g. 'airline_cancellation', 'customer_cancellation', 'delay', etc.).",
                    },
                    "decision": {
                        "type": "string",
                        "enum": [
                            "RESOLVED_REFUND",
                            "RESOLVED_VOUCHER",
                            "RESOLVED_REBOOK",
                            "ESCALATED",
                        ],
                        "description": "Terminal state for the case.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Your stated reasoning for the decision, including any tradeoff logic.",
                    },
                    "customer_response": {
                        "type": "string",
                        "description": "The customer-facing response that was drafted.",
                    },
                    "policy_cited": {
                        "type": "string",
                        "description": "The policy clause that was applied.",
                    },
                    "refund_calculation": {
                        "type": "object",
                        "description": "The refund calculation results from calculate_refund.",
                    },
                    "facts_gathered": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key facts gathered during investigation.",
                    },
                },
                "required": [
                    "ticket_id",
                    "issue_type",
                    "decision",
                    "reasoning",
                    "customer_response",
                    "policy_cited",
                ],
            },
        },
    },
]


# ===========================================================================
# TOOL IMPLEMENTATIONS
# ===========================================================================

def execute_tool(name: str, args: dict) -> dict:
    """
    Dispatch a tool call to the correct implementation.

    Args:
        name: Tool function name
        args: Tool arguments (already parsed from JSON)

    Returns:
        Tool result as a dict (will be JSON-serialized back to the LLM)
    """
    dispatch = {
        "get_booking": _tool_get_booking,
        "get_policy": _tool_get_policy,
        "calculate_refund": _tool_calculate_refund,
        "check_itinerary_conflict": _tool_check_itinerary_conflict,
        "search_alternate_flights": _tool_search_alternate_flights,
        "estimate_urgency": _tool_estimate_urgency,
        "draft_response": _tool_draft_response,
        "finalize_case": _tool_finalize_case,
    }

    handler = dispatch.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        result = handler(args)
        logger.info("Tool %s → %s", name, json.dumps(result, default=str)[:200])
        return result
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
        return {"error": f"Tool {name} failed: {str(e)}"}


# ---------------------------------------------------------------------------
# Individual tool handlers
# ---------------------------------------------------------------------------

def _tool_get_booking(args: dict) -> dict:
    ticket_id = args["ticket_id"]
    for booking in _load_bookings():
        if booking["ticket_id"] == ticket_id:
            return {"booking": booking}
    return {"error": f"No booking found for ticket {ticket_id}"}


def _tool_get_policy(args: dict) -> dict:
    airline = args["airline"]
    fare_class = args["fare_class"]
    cause = args["cause"]

    for policy in _load_policies():
        if (
            policy["airline"] == airline
            and policy["fare_class"] == fare_class
            and policy["cause"] == cause
        ):
            return {"policy": policy}

    return {
        "error": (
            f"No policy found for airline='{airline}', "
            f"fare_class='{fare_class}', cause='{cause}'. "
            f"This may require escalation — the policy combination is not "
            f"covered in the current rule set."
        )
    }


def _tool_calculate_refund(args: dict) -> dict:
    """Delegates to business_logic.calculate_refund — deterministic, no LLM."""
    ticket_id = args["ticket_id"]
    airline = args["airline"]
    fare_class = args["fare_class"]
    cause = args["cause"]

    # Find the booking
    booking = None
    for b in _load_bookings():
        if b["ticket_id"] == ticket_id:
            booking = b
            break
    if not booking:
        return {"error": f"No booking found for ticket {ticket_id}"}

    # Find the policy
    policy = None
    for p in _load_policies():
        if (
            p["airline"] == airline
            and p["fare_class"] == fare_class
            and p["cause"] == cause
        ):
            policy = p
            break
    if not policy:
        return {"error": f"No matching policy found"}

    # Pure Python calculation — no LLM involved
    result = business_logic.calculate_refund(booking, policy)
    return {"refund_calculation": result}


def _tool_check_itinerary_conflict(args: dict) -> dict:
    """Delegates to business_logic.check_itinerary_conflict — deterministic."""
    ticket_id = args["ticket_id"]

    booking = None
    for b in _load_bookings():
        if b["ticket_id"] == ticket_id:
            booking = b
            break
    if not booking:
        return {"error": f"No booking found for ticket {ticket_id}"}

    result = business_logic.check_itinerary_conflict(booking)
    return {"itinerary_check": result}


def _tool_search_alternate_flights(args: dict) -> dict:
    """
    Mock alternate flight search. Returns 2-3 realistic options
    based on the route. In production, this would hit a GDS/airline API.
    """
    origin = args["origin"]
    destination = args["destination"]
    near_date = args["near_date"]

    # Pre-built alternates for demo routes
    mock_routes = {
        ("SEA", "LAX"): [
            {
                "flight_number": "PA-0461",
                "airline": "Pacific Air",
                "origin": "SEA",
                "destination": "LAX",
                "departure": f"{near_date}T14:00:00",
                "arrival": f"{near_date}T17:00:00",
                "fare": 195.00,
                "seats_available": 12,
                "fare_class": "economy",
            },
            {
                "flight_number": "PA-0463",
                "airline": "Pacific Air",
                "origin": "SEA",
                "destination": "LAX",
                "departure": f"{near_date}T18:30:00",
                "arrival": f"{near_date}T21:30:00",
                "fare": 175.00,
                "seats_available": 8,
                "fare_class": "economy",
            },
        ],
        ("JFK", "LHR"): [
            {
                "flight_number": "AA-7823",
                "airline": "Atlantic Airways",
                "origin": "JFK",
                "destination": "LHR",
                "departure": f"{near_date}T08:00:00",
                "arrival": f"{near_date}T20:00:00",
                "fare": 1250.00,
                "seats_available": 4,
                "fare_class": "business",
            },
            {
                "flight_number": "AA-7825",
                "airline": "Atlantic Airways",
                "origin": "JFK",
                "destination": "LHR",
                "departure": f"{near_date}T14:00:00",
                "arrival": f"{near_date}T02:00:00",
                "fare": 1180.00,
                "seats_available": 7,
                "fare_class": "business",
            },
            {
                "flight_number": "BA-178",
                "airline": "British Airways (partner)",
                "origin": "JFK",
                "destination": "LHR",
                "departure": f"{near_date}T10:30:00",
                "arrival": f"{near_date}T22:30:00",
                "fare": 1300.00,
                "seats_available": 2,
                "fare_class": "business",
            },
        ],
        ("BOS", "SFO"): [
            {
                "flight_number": "PA-1124",
                "airline": "Pacific Air",
                "origin": "BOS",
                "destination": "SFO",
                "departure": f"{near_date}T09:00:00",
                "arrival": f"{near_date}T12:30:00",
                "fare": 540.00,
                "seats_available": 15,
                "fare_class": "premium_economy",
            },
            {
                "flight_number": "PA-1126",
                "airline": "Pacific Air",
                "origin": "BOS",
                "destination": "SFO",
                "departure": f"{near_date}T13:00:00",
                "arrival": f"{near_date}T16:30:00",
                "fare": 510.00,
                "seats_available": 9,
                "fare_class": "premium_economy",
            },
            {
                "flight_number": "PA-1128",
                "airline": "Pacific Air",
                "origin": "BOS",
                "destination": "SFO",
                "departure": f"{near_date}T17:30:00",
                "arrival": f"{near_date}T21:00:00",
                "fare": 495.00,
                "seats_available": 22,
                "fare_class": "premium_economy",
            },
        ],
        ("DEN", "MIA"): [
            {
                "flight_number": "SW-3344",
                "airline": "SkyWest Airlines",
                "origin": "DEN",
                "destination": "MIA",
                "departure": f"{near_date}T16:00:00",
                "arrival": f"{near_date}T22:30:00",
                "fare": 290.00,
                "seats_available": 18,
                "fare_class": "economy",
            },
            {
                "flight_number": "SW-3346",
                "airline": "SkyWest Airlines",
                "origin": "DEN",
                "destination": "MIA",
                "departure": f"{near_date}T20:00:00",
                "arrival": f"{near_date}T02:30:00",
                "fare": 265.00,
                "seats_available": 25,
                "fare_class": "economy",
            },
        ],
    }

    key = (origin, destination)
    alternatives = mock_routes.get(key, [])

    if not alternatives:
        # Generate generic fallback options
        alternatives = [
            {
                "flight_number": "GEN-001",
                "airline": "Generic Airways",
                "origin": origin,
                "destination": destination,
                "departure": f"{near_date}T10:00:00",
                "arrival": f"{near_date}T14:00:00",
                "fare": 350.00,
                "seats_available": 10,
                "fare_class": "economy",
            },
        ]

    return {
        "origin": origin,
        "destination": destination,
        "search_date": near_date,
        "alternatives": alternatives,
        "result_count": len(alternatives),
    }


def _tool_estimate_urgency(args: dict) -> dict:
    """Delegates to business_logic.estimate_urgency — keyword heuristic."""
    ticket_text = args["ticket_text"]
    return business_logic.estimate_urgency(ticket_text)


def _tool_draft_response(args: dict) -> dict:
    """
    Uses a focused LLM call to draft the customer-facing response.

    This is a SEPARATE call from the main reasoning loop — it's a
    formatting task, not a judgment task.
    """
    prompt = f"""Draft a customer response with these details:

Customer Name: {args['customer_name']}
Issue Summary: {args['issue_summary']}
Decision: {args['decision']}
Decision Details: {args['decision_details']}
Policy Clause to Cite: {args['policy_clause']}

Write the response now. Address the customer directly. Be professional, empathetic, and concise (3-5 paragraphs). Cite the exact policy clause provided."""

    try:
        result = llm_module.chat(
            messages=[
                llm_module.system_message(DRAFT_RESPONSE_SYSTEM),
                llm_module.user_message(prompt),
            ],
            tools=None,  # No tool use for drafting
            temperature=0.4,
        )
        return {"customer_response": result["content"]}
    except Exception as e:
        # Fallback to template if LLM draft fails
        logger.warning("LLM draft_response failed: %s. Using template.", e)
        return {
            "customer_response": (
                f"Dear {args['customer_name']},\n\n"
                f"Thank you for contacting us regarding your concern. "
                f"After reviewing your case, we have determined the following resolution:\n\n"
                f"{args['decision_details']}\n\n"
                f"This action is taken in accordance with: {args['policy_clause']}\n\n"
                f"If you have any further questions, please don't hesitate to reach out.\n\n"
                f"Best regards,\nCaseClose Travel Support"
            )
        }


def _tool_finalize_case(args: dict) -> dict:
    """
    Write the case to a terminal state and save the audit trace.

    This creates a JSON file in output/case_logs/{ticket_id}.json
    containing the full audit trail.
    """
    ticket_id = args["ticket_id"]

    # Build the audit trace
    audit_trace = {
        "ticket_id": ticket_id,
        "issue_type": args.get("issue_type", "unknown"),
        "facts_gathered": args.get("facts_gathered", []),
        "refund_calculation": args.get("refund_calculation", {}),
        "reasoning": args.get("reasoning", ""),
        "decision": args["decision"],
        "customer_response": args.get("customer_response", ""),
        "policy_cited": args.get("policy_cited", ""),
        "finalized_at": datetime.now().isoformat(),
    }

    # Ensure output directory exists
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write audit trace
    output_path = _OUTPUT_DIR / f"{ticket_id}.json"
    with open(output_path, "w") as f:
        json.dump(audit_trace, f, indent=2, default=str)

    logger.info("Case %s finalized as %s → %s", ticket_id, args["decision"], output_path)

    return {
        "status": "finalized",
        "ticket_id": ticket_id,
        "decision": args["decision"],
        "audit_trace_path": str(output_path),
    }

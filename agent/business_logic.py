"""
Deterministic business logic for CaseClose.

ALL financial calculations live here — never in the LLM.
This is the correctness boundary: the LLM decides *which* action to take,
these functions compute the exact dollar amounts.
"""

from datetime import datetime, date

# --------------------------------------------------------------------------
# Voucher cost model
# --------------------------------------------------------------------------
# A voucher's real cost to the business is less than its face value because:
#   1. ~30% of vouchers are never redeemed (breakage)
#   2. Redeemed vouchers drive repeat bookings with margin
# Effective cost ≈ face_value × 0.40
# This is the number that makes the voucher-vs-cash tradeoff real.
VOUCHER_COST_FACTOR = 0.40


def calculate_refund(booking: dict, policy: dict) -> dict:
    """
    Compute all refund/voucher amounts for a booking under a given policy.

    Returns a dict with exact dollar amounts — the LLM uses these as-is,
    it never computes its own numbers.

    Args:
        booking: A booking record from bookings.json
        policy:  A matched policy rule from policies.json

    Returns:
        {
            "cash_refund_amount":        float,
            "voucher_equivalent_amount": float,
            "change_fee":                float,
            "cost_to_business_cash":     float,
            "cost_to_business_voucher":  float,
            "auto_approve_max":          float,
            "exceeds_auto_approve":      bool,
            "policy_clause":             str,
        }
    """
    total_paid = booking["total_paid"]
    refund_pct = policy["refund_percent"]
    voucher_pct = policy["voucher_percent"]

    cash_refund = round(total_paid * (refund_pct / 100), 2)
    voucher_amount = round(total_paid * (voucher_pct / 100), 2)
    change_fee = policy["change_fee"]

    cost_cash = cash_refund  # cash out the door
    cost_voucher = round(voucher_amount * VOUCHER_COST_FACTOR, 2)

    auto_max = policy["auto_approve_max"]

    return {
        "cash_refund_amount": cash_refund,
        "voucher_equivalent_amount": voucher_amount,
        "change_fee": change_fee,
        "cost_to_business_cash": cost_cash,
        "cost_to_business_voucher": cost_voucher,
        "auto_approve_max": auto_max,
        "exceeds_auto_approve": cash_refund > auto_max,
        "policy_clause": policy["policy_clause"],
    }


def check_itinerary_conflict(booking: dict) -> dict:
    """
    Detect mismatches between flight and hotel dates.

    Catches:
      - Flight cancelled/rescheduled but hotel still confirmed
      - Flight arrival date is after hotel check-in (orphaned nights)
      - Missed connections leaving hotel bookings stranded

    Args:
        booking: A booking record from bookings.json

    Returns:
        {
            "conflict":           bool,
            "detail":             str,
            "orphaned_nights":    int,
            "wasted_hotel_cost":  float,
        }
    """
    flight = booking.get("flight", {})
    hotel = booking.get("hotel", {})

    if not hotel or hotel.get("status") != "CONFIRMED":
        return {
            "conflict": False,
            "detail": "No confirmed hotel booking to conflict with.",
            "orphaned_nights": 0,
            "wasted_hotel_cost": 0.0,
        }

    flight_status = flight.get("status", "UNKNOWN")
    hotel_check_in = _parse_date(hotel["check_in"])
    hotel_check_out = _parse_date(hotel["check_out"])
    rate = hotel.get("rate_per_night", 0)

    # Case 1: Flight cancelled — hotel is now orphaned entirely
    if flight_status == "CANCELLED":
        total_nights = (hotel_check_out - hotel_check_in).days
        return {
            "conflict": True,
            "detail": (
                f"Flight {flight['flight_number']} is cancelled but hotel "
                f"'{hotel['name']}' in {hotel.get('city', 'N/A')} remains "
                f"confirmed for {total_nights} night(s) "
                f"({hotel['check_in']} to {hotel['check_out']}). "
                f"Customer may need to cancel the hotel separately."
            ),
            "orphaned_nights": total_nights,
            "wasted_hotel_cost": round(total_nights * rate, 2),
        }

    # Case 2: Flight rescheduled — check if arrival is after hotel check-in
    if flight_status == "RESCHEDULED":
        # Use the rescheduled arrival time
        arrival_str = flight.get("arrival", flight.get("departure", ""))
        if arrival_str:
            arrival_dt = _parse_datetime(arrival_str)
            arrival_date = arrival_dt.date()

            if arrival_date > hotel_check_in:
                orphaned = (arrival_date - hotel_check_in).days
                return {
                    "conflict": True,
                    "detail": (
                        f"Flight {flight['flight_number']} has been rescheduled. "
                        f"New arrival is {arrival_date.isoformat()} but hotel "
                        f"'{hotel['name']}' check-in is {hotel['check_in']}. "
                        f"{orphaned} night(s) will be wasted "
                        f"(${round(orphaned * rate, 2)} at ${rate}/night). "
                        f"Original departure was {flight.get('original_departure', 'N/A')}."
                    ),
                    "orphaned_nights": orphaned,
                    "wasted_hotel_cost": round(orphaned * rate, 2),
                }

    # Case 3: Missed connection — hotel at destination may be affected
    if flight_status in ("MISSED_CONNECTION", "DELAYED"):
        # Check if delayed arrival pushes past hotel check-in
        arrival_str = flight.get("arrival", "")
        if arrival_str:
            arrival_dt = _parse_datetime(arrival_str)
            arrival_date = arrival_dt.date()

            if arrival_date > hotel_check_in:
                orphaned = (arrival_date - hotel_check_in).days
                return {
                    "conflict": True,
                    "detail": (
                        f"Flight {flight['flight_number']} status is "
                        f"{flight_status}. Expected arrival {arrival_date.isoformat()} "
                        f"is after hotel '{hotel['name']}' check-in on "
                        f"{hotel['check_in']}. {orphaned} orphaned night(s)."
                    ),
                    "orphaned_nights": orphaned,
                    "wasted_hotel_cost": round(orphaned * rate, 2),
                }

    # No conflict detected
    return {
        "conflict": False,
        "detail": "No itinerary conflict detected. Flight and hotel dates are aligned.",
        "orphaned_nights": 0,
        "wasted_hotel_cost": 0.0,
    }


def estimate_urgency(ticket_text: str) -> dict:
    """
    Heuristic-based urgency scoring. No LLM call — pure keyword matching.

    Priority signals:
      HIGH: imminent travel (within 24-48h), stranded customer, legal threats,
            words like URGENT/ASAP/IMMEDIATELY/STRANDED
      MED:  upcoming travel (within a week), moderate frustration,
            words like frustrated/unacceptable/disappointed
      LOW:  future travel, informational queries, calm tone

    Args:
        ticket_text: Raw ticket body text

    Returns:
        {"priority": "low"|"med"|"high", "reason": str}
    """
    text_lower = ticket_text.lower()

    # --- HIGH priority signals ---
    high_keywords = [
        "urgent", "asap", "immediately", "stranded", "right now",
        "within the hour", "emergency", "stuck at", "legal",
        "aviation authority", "lawyer", "complaint",
    ]
    high_matches = [kw for kw in high_keywords if kw in text_lower]
    if high_matches:
        return {
            "priority": "high",
            "reason": f"High-urgency signals detected: {', '.join(high_matches)}",
        }

    # --- MEDIUM priority signals ---
    med_keywords = [
        "frustrated", "unacceptable", "disappointed", "messed up",
        "really", "significant", "poor", "terrible", "ridiculous",
        "reconsidering", "seriously",
    ]
    med_matches = [kw for kw in med_keywords if kw in text_lower]
    if med_matches:
        return {
            "priority": "med",
            "reason": f"Moderate frustration signals: {', '.join(med_matches)}",
        }

    # --- LOW priority (default) ---
    return {
        "priority": "low",
        "reason": "No urgency signals detected. Standard processing.",
    }


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _parse_date(date_str: str) -> date:
    """Parse a date string (YYYY-MM-DD) into a date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _parse_datetime(dt_str: str) -> datetime:
    """Parse an ISO datetime string into a datetime object."""
    # Handle both 'T' separated and space-separated formats
    return datetime.fromisoformat(dt_str)

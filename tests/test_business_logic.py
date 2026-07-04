"""
Unit tests for the deterministic business logic layer.

Tests the financial correctness boundary — the most critical code
in the project. These functions MUST be 100% correct because the
LLM trusts their output without verification.
"""

import pytest
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.business_logic import calculate_refund, check_itinerary_conflict, estimate_urgency


# ---------------------------------------------------------------------------
# Test calculate_refund
# ---------------------------------------------------------------------------

class TestCalculateRefund:
    """Tests for the core refund calculation engine."""

    def _make_booking(self, total_paid: float) -> dict:
        return {"total_paid": total_paid, "booking_id": "BKG-TEST"}

    def _make_policy(self, refund_pct: int, voucher_pct: int,
                     change_fee: float = 0, auto_max: float = 500) -> dict:
        return {
            "refund_percent": refund_pct,
            "voucher_percent": voucher_pct,
            "change_fee": change_fee,
            "auto_approve_max": auto_max,
            "policy_clause": "Test Policy §1.0",
        }

    def test_full_refund(self):
        """100% cash refund (airline cancellation)."""
        result = calculate_refund(
            self._make_booking(320),
            self._make_policy(100, 100),
        )
        assert result["cash_refund_amount"] == 320
        assert result["voucher_equivalent_amount"] == 320
        assert result["exceeds_auto_approve"] is False

    def test_partial_refund_with_tradeoff(self):
        """60% cash / 100% voucher (customer cancellation, premium economy)."""
        result = calculate_refund(
            self._make_booking(450),
            self._make_policy(60, 100),
        )
        assert result["cash_refund_amount"] == 270     # 450 * 0.60
        assert result["voucher_equivalent_amount"] == 450  # 450 * 1.00
        assert result["cost_to_business_cash"] == 270
        assert result["cost_to_business_voucher"] == 180   # 450 * 0.40

    def test_zero_refund_delay(self):
        """0% cash refund (delay — only voucher eligible)."""
        result = calculate_refund(
            self._make_booking(280),
            self._make_policy(0, 25),
        )
        assert result["cash_refund_amount"] == 0
        assert result["voucher_equivalent_amount"] == 70  # 280 * 0.25

    def test_exceeds_auto_approve_threshold(self):
        """Refund exceeds $500 auto-approve limit."""
        result = calculate_refund(
            self._make_booking(1200),
            self._make_policy(80, 100, auto_max=500),
        )
        assert result["cash_refund_amount"] == 960  # 1200 * 0.80
        assert result["exceeds_auto_approve"] is True

    def test_exactly_at_threshold(self):
        """Refund exactly at $500 — should NOT exceed."""
        result = calculate_refund(
            self._make_booking(500),
            self._make_policy(100, 100, auto_max=500),
        )
        assert result["cash_refund_amount"] == 500
        assert result["exceeds_auto_approve"] is False

    def test_one_dollar_over_threshold(self):
        """Refund at $501 — should exceed."""
        result = calculate_refund(
            self._make_booking(501),
            self._make_policy(100, 100, auto_max=500),
        )
        assert result["exceeds_auto_approve"] is True

    def test_voucher_cost_factor(self):
        """Voucher cost to business = face_value × 0.40."""
        result = calculate_refund(
            self._make_booking(1000),
            self._make_policy(50, 100),
        )
        assert result["cost_to_business_voucher"] == 400  # 1000 * 0.40

    def test_change_fee_included(self):
        """Change fee is passed through from policy."""
        result = calculate_refund(
            self._make_booking(450),
            self._make_policy(60, 100, change_fee=75),
        )
        assert result["change_fee"] == 75

    def test_policy_clause_passed_through(self):
        """Policy clause text is included in result."""
        result = calculate_refund(
            self._make_booking(100),
            self._make_policy(100, 100),
        )
        assert result["policy_clause"] == "Test Policy §1.0"


# ---------------------------------------------------------------------------
# Test check_itinerary_conflict
# ---------------------------------------------------------------------------

class TestCheckItineraryConflict:
    """Tests for itinerary conflict detection."""

    def test_no_hotel(self):
        """No hotel booking — no conflict possible."""
        booking = {
            "flight": {"flight_number": "SW-001", "status": "CONFIRMED"},
        }
        result = check_itinerary_conflict(booking)
        assert result["conflict"] is False

    def test_cancelled_flight_with_hotel(self):
        """Cancelled flight + confirmed hotel = conflict."""
        booking = {
            "flight": {"flight_number": "SW-001", "status": "CANCELLED"},
            "hotel": {
                "name": "Test Hotel",
                "status": "CONFIRMED",
                "check_in": "2026-07-15",
                "check_out": "2026-07-18",
                "rate_per_night": 200,
            },
        }
        result = check_itinerary_conflict(booking)
        assert result["conflict"] is True
        assert result["orphaned_nights"] == 3
        assert result["wasted_hotel_cost"] == 600

    def test_rescheduled_flight_creates_orphan(self):
        """Rescheduled flight arriving after hotel check-in."""
        booking = {
            "flight": {
                "flight_number": "PA-001",
                "status": "RESCHEDULED",
                "arrival": "2026-07-11T18:00:00",
                "original_departure": "2026-07-10T09:00:00",
            },
            "hotel": {
                "name": "Beach Resort",
                "status": "CONFIRMED",
                "check_in": "2026-07-10",
                "check_out": "2026-07-14",
                "rate_per_night": 220,
            },
        }
        result = check_itinerary_conflict(booking)
        assert result["conflict"] is True
        assert result["orphaned_nights"] == 1
        assert result["wasted_hotel_cost"] == 220

    def test_confirmed_flight_no_conflict(self):
        """Confirmed flight + confirmed hotel = no conflict."""
        booking = {
            "flight": {
                "flight_number": "SW-001",
                "status": "CONFIRMED",
                "arrival": "2026-07-15T14:00:00",
            },
            "hotel": {
                "name": "City Hotel",
                "status": "CONFIRMED",
                "check_in": "2026-07-15",
                "check_out": "2026-07-18",
                "rate_per_night": 150,
            },
        }
        result = check_itinerary_conflict(booking)
        assert result["conflict"] is False


# ---------------------------------------------------------------------------
# Test estimate_urgency
# ---------------------------------------------------------------------------

class TestEstimateUrgency:
    """Tests for the urgency heuristic."""

    def test_high_urgency_urgent_keyword(self):
        result = estimate_urgency("URGENT — I need help immediately!")
        assert result["priority"] == "high"

    def test_high_urgency_stranded(self):
        result = estimate_urgency("I am stranded at JFK airport")
        assert result["priority"] == "high"

    def test_medium_urgency_frustrated(self):
        result = estimate_urgency("This is really frustrating and unacceptable")
        assert result["priority"] == "med"

    def test_low_urgency_calm(self):
        result = estimate_urgency("Hi, I'd like to know my options for changing dates.")
        assert result["priority"] == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Agent Memory — Cross-ticket working memory for CaseClose.

Stores summaries of previously resolved tickets so the agent can:
  - Detect patterns (e.g., "3 SkyWest cancellations today")
  - Reference prior resolutions for consistency
  - Build situational awareness across a batch

Memory is ephemeral (lives for the duration of the process).
In production, this would be backed by a database.
"""

import json
import logging
from datetime import datetime

logger = logging.getLogger("caseclose.memory")


class AgentMemory:
    """
    Short-term working memory for the agent.

    Stores compact summaries of resolved tickets and generates
    a context block that can be injected into the system prompt.
    """

    def __init__(self):
        self.resolved_cases: list[dict] = []
        self.pattern_alerts: list[str] = []

    def store_resolution(self, audit: dict):
        """
        Store a compact summary of a resolved ticket.

        Args:
            audit: The full audit trace dict from process_ticket()
        """
        summary = {
            "ticket_id": audit.get("ticket_id"),
            "issue_type": audit.get("issue_type"),
            "decision": audit.get("decision"),
            "airline": self._extract_airline(audit),
            "reasoning_summary": (audit.get("reasoning") or "")[:200],
            "cash_amount": None,
            "voucher_amount": None,
            "resolved_at": datetime.now().isoformat(),
        }

        rc = audit.get("refund_calculation")
        if rc and isinstance(rc, dict):
            summary["cash_amount"] = rc.get("cash_refund_amount")
            summary["voucher_amount"] = rc.get("voucher_equivalent_amount")

        self.resolved_cases.append(summary)
        logger.info("Memory: stored resolution for %s", summary["ticket_id"])

        # Run pattern detection after storing
        self._detect_patterns()

    def get_context_block(self) -> str:
        """
        Generate a context block for injection into the system prompt.

        Returns a formatted string summarizing prior resolutions and
        any detected patterns. Returns empty string if no history.
        """
        if not self.resolved_cases:
            return ""

        lines = [
            "",
            "## Agent Memory — Prior Resolutions This Session",
            f"You have resolved {len(self.resolved_cases)} ticket(s) so far:",
            "",
        ]

        for case in self.resolved_cases:
            cash = f"${case['cash_amount']}" if case['cash_amount'] is not None else "N/A"
            voucher = f"${case['voucher_amount']}" if case['voucher_amount'] is not None else "N/A"
            lines.append(
                f"- **{case['ticket_id']}** ({case['issue_type']}): "
                f"{case['decision']} | Cash: {cash}, Voucher: {voucher} | "
                f"Airline: {case['airline']}"
            )

        if self.pattern_alerts:
            lines.append("")
            lines.append("### Detected Patterns")
            for alert in self.pattern_alerts:
                lines.append(f"- ⚠️ {alert}")

        lines.append("")
        lines.append(
            "Use this context to maintain consistency in your decisions "
            "and flag potential systemic issues."
        )

        return "\n".join(lines)

    def _detect_patterns(self):
        """
        Analyze resolved cases for patterns.
        Runs after every new resolution.
        """
        self.pattern_alerts = []

        # Pattern: Multiple issues with the same airline
        airline_counts: dict[str, int] = {}
        for case in self.resolved_cases:
            airline = case.get("airline", "Unknown")
            airline_counts[airline] = airline_counts.get(airline, 0) + 1

        for airline, count in airline_counts.items():
            if count >= 2:
                self.pattern_alerts.append(
                    f"Multiple issues ({count}) reported for {airline} this session. "
                    f"Possible systemic problem — consider flagging to operations."
                )

        # Pattern: Multiple escalations
        escalation_count = sum(
            1 for c in self.resolved_cases if c["decision"] == "ESCALATED"
        )
        if escalation_count >= 2:
            self.pattern_alerts.append(
                f"{escalation_count} tickets escalated this session. "
                f"High escalation rate may indicate policy gaps or "
                f"unusually complex batch."
            )

        # Pattern: High total cost
        total_cash = sum(
            c.get("cash_amount", 0) or 0 for c in self.resolved_cases
        )
        if total_cash > 1000:
            self.pattern_alerts.append(
                f"Cumulative cash refunds this session: ${total_cash:.2f}. "
                f"Approaching significant financial exposure."
            )

    def _extract_airline(self, audit: dict) -> str:
        """Try to extract airline name from audit facts."""
        facts = audit.get("facts_gathered", [])
        for fact in facts:
            fact_str = str(fact)
            for airline in ["SkyWest Airlines", "Pacific Air", "Atlantic Airways"]:
                if airline in fact_str:
                    return airline
        return "Unknown"

    @property
    def case_count(self) -> int:
        return len(self.resolved_cases)

    def __repr__(self) -> str:
        return (
            f"AgentMemory(cases={len(self.resolved_cases)}, "
            f"patterns={len(self.pattern_alerts)})"
        )

# CaseClose — Architecture Guide

## System Overview

CaseClose is an autonomous AI agent that resolves travel support tickets end-to-end.
It is NOT a chatbot — it owns each case to a terminal state (resolved or escalated)
without human intervention.

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py (CLI)                           │
│  ┌──────────┐  ┌──────────────────┐  ┌───────────────────────┐ │
│  │  Ticket   │→│   AgentMemory    │→│    process_ticket()    │ │
│  │  Loader   │  │ (cross-ticket   │  │    (ReAct Loop)       │ │
│  │           │  │  context)        │  │                       │ │
│  └──────────┘  └──────────────────┘  └───────────┬───────────┘ │
└──────────────────────────────────────────────────│─────────────┘
                                                   │
                    ┌──────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      agent/loop.py                              │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  PLAN    │───→│  GATHER  │───→│ COMPUTE  │───→│  DECIDE  │ │
│   │          │    │          │    │          │    │          │ │
│   │ LLM out- │    │ get_book │    │calculate │    │ Tradeoff │ │
│   │ puts its │    │ get_poli │    │_refund   │    │ analysis │ │
│   │ strategy │    │ estimate │    │          │    │          │ │
│   │ before   │    │ check_it │    │          │    │          │ │
│   │ acting   │    │          │    │          │    │          │ │
│   └──────────┘    └──────────┘    └──────────┘    └────┬─────┘ │
│                                                        │       │
│                                          ┌─────────────┘       │
│                                          ▼                     │
│                                   ┌──────────┐                 │
│                                   │   ACT    │                 │
│                                   │          │                 │
│                                   │ draft_   │                 │
│                                   │ response │                 │
│                                   │ finalize │                 │
│                                   │ _case    │                 │
│                                   └──────────┘                 │
│                                                                 │
│   Self-Correction: If any tool returns an error, the loop      │
│   injects guidance and the LLM adapts its approach.            │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Capabilities

### 1. Autonomous Planning
Before executing any tools, the agent outputs a numbered execution plan:
- What type of issue this is
- Which tools to call and in what order
- What data to gather
- What tradeoffs or edge cases to anticipate
- Under what conditions to escalate

**File:** `agent/prompts.py` → `PLANNING_PROMPT`

### 2. ReAct Reasoning Loop
The core loop follows the Reason + Act pattern:
1. LLM receives ticket + system prompt + memory context
2. LLM selects tools (parallel calling supported)
3. Tools execute and return results
4. LLM reasons over results
5. Repeat until terminal state or iteration cap

**File:** `agent/loop.py` → `process_ticket()`

### 3. Tool Calling
8 tools available via OpenAI function-calling format:

| Tool | Type | Purpose |
|------|------|---------|
| `get_booking` | Data retrieval | Fetch booking record |
| `get_policy` | Data retrieval | Look up airline policy rules |
| `calculate_refund` | **Deterministic** | Compute exact refund amounts |
| `check_itinerary_conflict` | **Deterministic** | Detect flight/hotel date mismatches |
| `search_alternate_flights` | Data retrieval | Find rebooking options |
| `estimate_urgency` | **Heuristic** | Assess ticket priority |
| `draft_response` | LLM generation | Draft customer-facing message |
| `finalize_case` | State mutation | Close the case + write audit |

**Critical design decision:** `calculate_refund` and `check_itinerary_conflict` are
pure Python — the LLM is explicitly prohibited from performing financial math.
This is the correctness boundary.

**File:** `agent/tools.py`

### 4. Deterministic Business Logic
All financial calculations are isolated in a separate module:

```
┌──────────────────────────────────────────────┐
│           business_logic.py                   │
│                                               │
│   calculate_refund()                          │
│     cash = total_paid × refund_percent        │
│     voucher = total_paid × voucher_percent    │
│     cost_voucher = voucher × 0.40 (breakage)  │
│     exceeds = cash > auto_approve_max         │
│                                               │
│   check_itinerary_conflict()                  │
│     CANCELLED → all hotel nights orphaned     │
│     RESCHEDULED → arrival > check_in?         │
│     DELAYED → arrival > check_in?             │
│                                               │
│   estimate_urgency()                          │
│     HIGH: urgent/stranded/legal keywords      │
│     MED: frustrated/unacceptable keywords     │
│     LOW: default                              │
└──────────────────────────────────────────────┘
```

**File:** `agent/business_logic.py`  
**Tests:** `tests/test_business_logic.py` (17 tests)

### 5. Agent Memory
Cross-ticket working memory that persists across a batch run:

- Stores summaries of resolved tickets
- Detects patterns:
  - Multiple issues with the same airline → "possible systemic problem"
  - High escalation rate → "policy gaps"
  - Cumulative high refund totals → "financial exposure"
- Injects context into the system prompt for subsequent tickets

**File:** `agent/memory.py` → `AgentMemory`

### 6. Guardrails & Safety
- **Financial threshold:** `$500 auto_approve_max` — any refund exceeding this auto-escalates
- **Iteration cap:** `MAX_ITERATIONS = 6` — force-escalates if agent doesn't converge
- **Arithmetic prohibition:** LLM cannot compute dollar amounts (enforced by prompt)
- **Turn separation:** `calculate_refund` and `finalize_case` cannot be called in the same turn

### 7. Self-Correction
When a tool returns an error, the loop:
1. Increments `self_corrections` counter
2. Logs the error in the thought trace
3. Injects guidance: *"Consider an alternative approach or escalate"*
4. The LLM adapts its strategy in the next iteration

### 8. Full Audit Trail
Every ticket produces a rich JSON audit log with:
- `execution_plan` — the agent's strategy before acting
- `thought_trace` — every LLM thought and tool result at every iteration
- `reasoning` — the final tradeoff analysis
- `tools_called` — every tool with args, iteration, and success/failure
- `total_tokens_used` — cumulative LLM token consumption
- `self_corrections` — number of error recovery adaptations

### 9. Provider Failover
The `ProviderManager` handles LLM reliability:
- Exponential backoff for transient 429s (2s → 4s → 8s → 16s → 32s)
- Automatic failover across multiple API keys
- Primary model → fallback model rotation
- Clean terminal output during failover

**File:** `agent/llm.py` → `ProviderManager`

## File Map

```
caseclose/
├── main.py                    # CLI entrypoint + batch orchestration
├── server.py                  # Dashboard HTTP server
├── dashboard.html             # Operations dashboard (fetches live data)
├── agent/
│   ├── loop.py                # Core ReAct loop (plan → gather → compute → decide → act)
│   ├── memory.py              # Cross-ticket working memory + pattern detection
│   ├── prompts.py             # System prompt, planning prompt, decision prompt
│   ├── tools.py               # Tool schemas + implementations
│   ├── business_logic.py      # Deterministic financial calculations
│   ├── llm.py                 # Provider manager with failover
│   └── config.json            # API keys (gitignored)
├── data/
│   ├── bookings.json          # Mock booking records
│   ├── policies.json          # Airline refund/voucher policies
│   └── tickets.json           # Support tickets (6 edge cases)
├── output/
│   └── case_logs/             # Full audit traces (JSON per ticket)
├── tests/
│   └── test_business_logic.py # 17 unit tests for financial logic
└── requirements.txt
```

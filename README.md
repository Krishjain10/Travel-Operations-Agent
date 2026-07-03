# CaseClose — Autonomous Travel Operations Agent

**CaseClose** is an autonomous AI agent built to execute end-to-end travel operations support tickets (cancellations, delays, rebookings). 

Instead of a standard Q&A chatbot, this agent uses a strict **ReAct (Reason + Act)** loop to autonomously classify issues, gather data via tool calling, analyze financial tradeoffs, and execute decisions while adhering to strict business constraints.

## Key Features

1. **Deterministic Boundaries**: LLMs are prohibited from performing financial arithmetic. All math and date conflict logic is offloaded to a secure, deterministic Python business logic layer.
2. **Tradeoff Optimization**: The agent evaluates the financial cost of cash vs. voucher refunds to optimize both business savings and customer satisfaction before making a final decision.
3. **Escalation Guardrails**: The agent enforces a strict `$500` `auto_approve_max` limit. Any calculation exceeding this threshold halts the autonomous workflow and drafts a structured escalation for human review.
4. **High Availability**: Incorporates a custom `ProviderManager` to handle LLM rate limits (`HTTP 429`) via exponential backoff and seamless multi-provider quota failover.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create `agent/config.json` with your Groq API key:
```json
{
  "groq_api_keys": [
    "gsk_your_key_here"
  ]
}
```
*Or set the environment variable: `export GROQ_API_KEY=gsk_your_key_here`*

### 3. Run the Agent
The agent processes support tickets found in `data/tickets.json`. 

Run a specific ticket with verbose reasoning enabled:
```bash
python main.py --ticket TKT-002 --verbose
```

Run the entire batch of edge cases:
```bash
python main.py --all
```

## Architecture

* `data/`: Mock data layer containing bookings, airline policies, and support tickets.
* `agent/business_logic.py`: Deterministic refund and conflict calculations.
* `agent/tools.py`: OpenAI-compatible schemas and Python dispatchers for the agent to use.
* `agent/loop.py`: The core ReAct loop (Gather -> Compute -> Decide -> Act).
* `agent/llm.py`: Provider manager with automated failover and 429 backoff.

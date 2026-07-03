"""
Multi-Provider API wrapper with tool-use support, 429 retry/backoff, and quota failover.

Provides:
  - Chat completions with native function calling
  - Exponential backoff for transient rate limits (429s)
  - Modular provider architecture (LLMProvider base class)
  - Automated failover across multiple API keys/providers if a quota is exceeded
"""

import os
import sys
import json
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from groq import Groq, RateLimitError, APIError, APIStatusError

logger = logging.getLogger("caseclose.llm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2  # doubles each retry: 2s, 4s, 8s, 16s, 32s


# ---------------------------------------------------------------------------
# Provider Interface
# ---------------------------------------------------------------------------
class LLMProvider(ABC):
    """Abstract base class for an LLM API provider."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider instance (e.g. 'Groq-Key1')."""
        pass

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None, tool_choice: str, model: str, temperature: float) -> dict:
        """
        Send a chat completion request.
        Must handle its own transient rate limits (e.g. short-term 429s).
        Must raise an exception if the key has hit a hard quota or is permanently unavailable.
        """
        pass


# ---------------------------------------------------------------------------
# Groq Provider Implementation
# ---------------------------------------------------------------------------
class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, instance_name: str):
        self.api_key = api_key
        self._name = instance_name
        self.client = Groq(api_key=api_key)

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages: list[dict], tools: list[dict] | None, tool_choice: str, model: str, temperature: float) -> dict:
        target_model = model or PRIMARY_MODEL
        tried_fallback = False

        while True:
            try:
                return self._call_with_retry(messages, tools, tool_choice, target_model, temperature)
            except APIStatusError as e:
                # 413 is Request Too Large (often a strict TPM limit on fallback model).
                # This is a hard failure for this provider/model combo.
                if e.status_code == 413:
                    logger.error("[%s] Hard limit reached (413): %s", self.name, e)
                    raise
                # 429 that exhausts retries will also surface here.
                raise
            except (APIError, Exception) as e:
                # If primary model fails with a non-rate-limit/non-quota error, try fallback model
                if not tried_fallback and target_model == PRIMARY_MODEL:
                    logger.warning(
                        "[%s] Primary model %s failed: %s. Trying fallback %s",
                        self.name, PRIMARY_MODEL, e, FALLBACK_MODEL,
                    )
                    target_model = FALLBACK_MODEL
                    tried_fallback = True
                    continue
                raise

    def _call_with_retry(self, messages: list[dict], tools: list[dict] | None, tool_choice: str, model: str, temperature: float) -> dict:
        for attempt in range(MAX_RETRIES + 1):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 4096,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice

                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message

                # Extract tool calls into a cleaner format
                tool_calls = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments),
                        })

                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                if usage:
                    logger.debug(
                        "[%s] Model=%s | Tokens: %d prompt + %d completion = %d total",
                        self.name, model, usage["prompt_tokens"], usage["completion_tokens"],
                        usage["total_tokens"],
                    )

                return {
                    "content": message.content,
                    "tool_calls": tool_calls,
                    "model": model,
                    "usage": usage,
                    "raw_message": message,  # Keep for appending to conversation
                }

            except RateLimitError as e:
                # Check if it's a hard quota limit (e.g. Tokens Per Day reached) vs a transient rate limit
                error_msg = str(e).lower()
                if "tokens per day" in error_msg and "limit 100000" in error_msg:
                    logger.debug("[%s] Daily quota exhausted. Giving up on this provider.", self.name)
                    raise

                if attempt == MAX_RETRIES:
                    logger.debug("[%s] Rate limit exceeded after %d retries. Giving up.", self.name, MAX_RETRIES)
                    raise

                wait = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "[%s] Rate limited (429). Retry %d/%d in %ds...",
                    self.name, attempt + 1, MAX_RETRIES, wait,
                )
                time.sleep(wait)


# ---------------------------------------------------------------------------
# Provider Manager
# ---------------------------------------------------------------------------
class ProviderManager:
    """Manages a pool of providers and handles failover."""
    
    def __init__(self):
        self.providers: list[LLMProvider] = []
        self.current_idx = 0
        self._load_providers()

    def _load_providers(self):
        """Load API keys from agent/config.json or environment."""
        config_path = Path(__file__).resolve().parent / "config.json"
        
        # Load from config file
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    keys = config.get("groq_api_keys", [])
                    for i, key in enumerate(keys):
                        self.providers.append(GroqProvider(api_key=key, instance_name=f"Groq-Key{i+1}"))
            except Exception as e:
                logger.error("Failed to load config.json: %s", e)

        # Also check ENV variable just in case
        env_key = os.environ.get("GROQ_API_KEY")
        if env_key and not any(isinstance(p, GroqProvider) and p.api_key == env_key for p in self.providers):
            self.providers.append(GroqProvider(api_key=env_key, instance_name="Groq-EnvKey"))

        if not self.providers:
            print("ERROR: No API keys configured in agent/config.json or GROQ_API_KEY env var.", file=sys.stderr)
            sys.exit(1)

        logger.info("Loaded %d LLM providers.", len(self.providers))

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto", model: str | None = None, temperature: float = 0.2) -> dict:
        """Routes the chat request to the active provider, with automatic failover."""
        attempts = 0
        max_attempts = len(self.providers)

        while attempts < max_attempts:
            provider = self.providers[self.current_idx]
            try:
                # Try to execute chat with the current provider
                return provider.chat(messages, tools, tool_choice, model, temperature)
            except Exception as e:
                next_idx = (self.current_idx + 1) % len(self.providers)
                print(f"  [API] {provider.name} quota exhausted. Seamlessly failing over to {self.providers[next_idx].name}...")
                logger.debug("Provider %s failed with error: %s", provider.name, e)
                # Fail over to the next provider
                self.current_idx = next_idx
                attempts += 1

        logger.error("All %d providers failed.", max_attempts)
        raise RuntimeError("All LLM providers are exhausted or unavailable.")

# Singleton manager instance
_manager = None

def _get_manager() -> ProviderManager:
    global _manager
    if _manager is None:
        _manager = ProviderManager()
    return _manager


# ---------------------------------------------------------------------------
# Core exported chat function
# ---------------------------------------------------------------------------
def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    model: str | None = None,
    temperature: float = 0.2,
) -> dict:
    """
    Send a chat completion request to the provider manager.
    Automatically handles provider rotation and failover.
    """
    manager = _get_manager()
    return manager.chat(messages, tools, tool_choice, model, temperature)


# ---------------------------------------------------------------------------
# Helper to build message dicts
# ---------------------------------------------------------------------------
def system_message(content: str) -> dict:
    return {"role": "system", "content": content}

def user_message(content: str) -> dict:
    return {"role": "user", "content": content}

def assistant_message(content: str) -> dict:
    return {"role": "assistant", "content": content}

def tool_result_message(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}

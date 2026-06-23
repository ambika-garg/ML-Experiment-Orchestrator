"""Base agent class providing LLM integration for all AI-powered agents.

Every LLM-backed agent inherits from BaseAgent, which manages:
  • LLM instantiation (Gemini via langchain-google-genai)
  • Structured JSON output parsing with retry logic
  • System prompt injection
  • Consistent logging
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ml_experiment_orchestrator.agents.demo_responses import (
    demo_json_response,
    demo_text_response,
)
from ml_experiment_orchestrator.config import settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all LLM-powered agents.

    Subclasses must implement:
      • ``system_prompt`` — the agent's identity & instruction prompt
      • ``run(state)`` — the main execution method
    """

    def __init__(self, temperature: float | None = None) -> None:
        self._temperature = temperature or settings.llm_temperature
        self._llm: ChatGoogleGenerativeAI | None = None
        self._name = self.__class__.__name__
        self._traces: list[dict[str, Any]] = []

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazily initialise the LLM client on first access."""
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for live LLM agents. "
                "Set DEMO_MODE=true to run with deterministic local agent responses."
            )
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=self._temperature,
            )
        return self._llm

    # ── Abstract Interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt defining this agent's role."""

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's task and return a partial state update."""

    # ── LLM Helpers ───────────────────────────────────────────────────────

    def invoke_llm(self, user_prompt: str) -> str:
        """Send a system + user message pair to the LLM and return the text."""
        import time

        if settings.demo_mode:
            logger.info("[%s] Demo mode enabled; using deterministic response", self._name)
            content = demo_text_response(self._name, user_prompt)
            # Create a mock trace for UI testing in demo mode
            prompt_tok = len(user_prompt.split()) * 4
            comp_tok = len(content.split()) * 4
            tot_tok = prompt_tok + comp_tok
            input_cost_rate = 0.075 / 1_000_000
            output_cost_rate = 0.30 / 1_000_000
            cost = (prompt_tok * input_cost_rate) + (comp_tok * output_cost_rate)
            
            self._traces.append({
                "agent_name": self._name,
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_tokens": tot_tok,
                "cost": round(cost, 8),
                "latency": round(0.4 + (prompt_tok / 1000.0), 3),
            })
            return content

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        logger.info("[%s] Invoking LLM …", self._name)
        
        start_time = time.perf_counter()
        response = self.llm.invoke(messages)
        latency = time.perf_counter() - start_time
        
        content = response.content
        logger.debug("[%s] Raw LLM response: %s", self._name, content[:500])

        # Track usage statistics
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = response.usage_metadata.get("input_tokens", 0)
            completion_tokens = response.usage_metadata.get("output_tokens", 0)
            total_tokens = response.usage_metadata.get("total_tokens", 0)
        elif hasattr(response, "response_metadata") and response.response_metadata:
            meta = response.response_metadata
            token_usage = meta.get("token_usage") or meta.get("usage")
            if isinstance(token_usage, dict):
                prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
                completion_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
                total_tokens = token_usage.get("total_tokens") or (prompt_tokens + completion_tokens)

        # Fallback to estimate if 0
        if total_tokens == 0:
            prompt_tokens = len(user_prompt.split()) * 4
            completion_tokens = len(content.split()) * 4
            total_tokens = prompt_tokens + completion_tokens

        # Cost calculation
        model_name = settings.llm_model
        input_cost_rate = 0.075 / 1_000_000
        output_cost_rate = 0.30 / 1_000_000

        if "pro" in model_name:
            input_cost_rate = 1.25 / 1_000_000
            output_cost_rate = 5.00 / 1_000_000

        cost = (prompt_tokens * input_cost_rate) + (completion_tokens * output_cost_rate)

        # Append to active traces
        self._traces.append({
            "agent_name": self._name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(cost, 8),
            "latency": round(latency, 3),
        })

        return content

    def invoke_llm_json(
        self, user_prompt: str, retries: int = 2
    ) -> dict[str, Any]:
        """Invoke LLM expecting a JSON response, with retry on parse failure.

        Strips markdown fences and attempts ``json.loads``.  On failure it
        re-prompts the LLM asking for valid JSON (up to *retries* attempts).
        """
        if settings.demo_mode:
            logger.info("[%s] Demo mode enabled; using deterministic JSON", self._name)
            res = demo_json_response(self._name, user_prompt)
            import json
            res_str = json.dumps(res)
            prompt_tok = len(user_prompt.split()) * 4
            comp_tok = len(res_str.split()) * 4
            tot_tok = prompt_tok + comp_tok
            input_cost_rate = 0.075 / 1_000_000
            output_cost_rate = 0.30 / 1_000_000
            cost = (prompt_tok * input_cost_rate) + (comp_tok * output_cost_rate)
            self._traces.append({
                "agent_name": self._name,
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_tokens": tot_tok,
                "cost": round(cost, 8),
                "latency": round(0.3 + (prompt_tok / 1200.0), 3),
            })
            return res

        for attempt in range(1 + retries):
            raw = self.invoke_llm(user_prompt)
            parsed = self._try_parse_json(raw)
            if parsed is not None:
                return parsed
            logger.warning(
                "[%s] JSON parse failed (attempt %d/%d)",
                self._name,
                attempt + 1,
                1 + retries,
            )
            if attempt < retries:
                user_prompt = (
                    "Your previous response was not valid JSON. "
                    "Please respond ONLY with a valid JSON object. "
                    "Do not include markdown fences or explanation text.\n\n"
                    f"Original request:\n{user_prompt}"
                )
        # If all retries fail, return an empty dict and log an error.
        logger.error(
            "[%s] Failed to get valid JSON after %d attempts", self._name, 1 + retries
        )
        return {}

    # ── Private Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """Attempt to parse JSON from LLM output, stripping fences."""
        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            r"\1",
            text,
            flags=re.DOTALL,
        )
        # Try the cleaned text first, then fall back to the raw text
        for candidate in (cleaned.strip(), text.strip()):
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
        return None

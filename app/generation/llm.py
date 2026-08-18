from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.ingestion.exceptions import LLMServiceError
from app.runtime_settings import settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    Generates answers via Ollama using LangChain's ``ChatOllama`` wrapper.

    When ``LANGSMITH_TRACING=true`` every ``generate()`` call appears in
    LangSmith as an LLM span with inputs, outputs, model name, and latency –
    automatically, with no extra instrumentation code.

    Parameters
    ----------
    model_name: Ollama model tag (defaults to ``settings.CHAT_MODEL``).
    host:       Ollama base URL (defaults to ``settings.OLLAMA_HOST``).
    client:     Optional pre-constructed ``ChatOllama`` instance (for testing).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name or settings.CHAT_MODEL
        self.host = host or settings.OLLAMA_HOST

        # Allow injection for tests; otherwise build the LangChain wrapper.
        self.client: Any = client or ChatOllama(
            model=self.model_name,
            base_url=self.host,
        )

        logger.info(
            "LLMService initialised (ChatOllama) | model=%s host=%s",
            self.model_name,
            self.host,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Generate a response from the LLM.

        LangChain automatically sends a trace to LangSmith (when configured)
        containing the full message list, model name, token counts, and
        latency for this invocation.

        Parameters
        ----------
        system_prompt: System-level instruction (grounding rules).
        user_prompt:   Context + question assembled by ``PromptBuilder``.

        Returns
        -------
        The model's text response.

        Raises
        ------
        LLMServiceError  on any Ollama / LangChain failure.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        logger.info(
            "LLM generate | model=%s context_chars=%d",
            self.model_name,
            len(user_prompt),
        )

        t0 = time.perf_counter()
        try:
            # ChatOllama.invoke() is what LangSmith hooks into for auto-tracing
            response = self.client.invoke(messages)
        except Exception as exc:
            logger.exception(
                "ChatOllama call failed | model=%s host=%s", self.model_name, self.host
            )
            raise LLMServiceError(
                f"Cannot reach Ollama LLM service at {self.host}: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        # AIMessage.content is always a str for chat models
        content: str = response.content if hasattr(response, "content") else str(response)

        if not content or not content.strip():
            raise LLMServiceError(
                f"Ollama returned an empty response for model '{self.model_name}'"
            )

        logger.info(
            "LLM generation complete | model=%s latency_ms=%.1f response_chars=%d",
            self.model_name,
            latency_ms,
            len(content),
        )
        return content.strip()

    def health_check(self) -> dict:
        """Return a dict with Ollama reachability status."""
        try:
            # A minimal invoke with a short prompt to confirm the model is up.
            # We use the underlying ollama list endpoint via a lightweight check.
            import httpx  # noqa: PLC0415 – optional fast path
            resp = httpx.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            return {"status": "ok", "model": self.model_name, "host": self.host}
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return {"status": "error", "detail": str(exc), "host": self.host}

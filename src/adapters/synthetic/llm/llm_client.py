"""Anthropic API client wrapper for LLM-based synthetic generation."""
from __future__ import annotations

from typing import Any

from src.adapters.synthetic.client.base import EvidenceSpec, SyntheticTextClient
from .prompt_builder import build_prompt, parse_llm_response


class LLMClient(SyntheticTextClient):
    """Calls the Anthropic Messages API to generate claim + evidence texts.

    Args:
        model:      Anthropic model ID.
        api_key:    Anthropic API key (or set ANTHROPIC_API_KEY env var).
        max_tokens: Maximum tokens per response (default 500).
        _client:    Injectable Anthropic client for testing.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 500,
        _client=None,
    ):
        if _client is not None:
            self._api = _client
        else:
            import anthropic
            self._api = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(
        self,
        specs: list[EvidenceSpec],
        template_name: str,
    ) -> dict[str, Any] | None:
        from src.adapters.synthetic.llm.prompt_builder import _SYSTEM_PROMPT
        prompt = build_prompt(specs, template_name)
        try:
            response = self._api.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
        except Exception:
            return None

        return parse_llm_response(raw_text)

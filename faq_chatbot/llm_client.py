"""A small, dependency-light client for LLM chat APIs.

Two wire protocols are supported:

* ``anthropic``        - the Anthropic Messages API
* ``openai`` / ``ollama`` - the OpenAI-compatible chat-completions API
  (also used by Ollama for locally running models)

Every failure is converted to :class:`LLMError` with a message that is safe to
show to a user - it never contains the API key.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import requests

from .config import Settings

ANTHROPIC_VERSION = "2023-06-01"
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


class LLMError(Exception):
    """Any problem talking to the LLM provider (network, auth, rate limit, bad response)."""


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        session: Optional[Any] = None,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self._sleep = sleep

    # ---- public API -----------------------------------------------------
    def generate(self, system_prompt: str, messages: Sequence[dict[str, str]]) -> str:
        """Send the conversation to the model and return the reply text."""
        url, headers, payload = self._build_request(system_prompt, list(messages))
        data = self._post_with_retries(url, headers, payload)
        return self._parse_response(data)

    # ---- request building -----------------------------------------------
    @property
    def _uses_anthropic_protocol(self) -> bool:
        return self.settings.provider == "anthropic"

    def _build_request(self, system_prompt: str, messages: list[dict[str, str]]):
        s = self.settings
        base = s.resolved_base_url
        headers = {"content-type": "application/json"}

        if self._uses_anthropic_protocol:
            url = f"{base}/v1/messages"
            headers["x-api-key"] = s.api_key
            headers["anthropic-version"] = ANTHROPIC_VERSION
            payload: dict[str, Any] = {
                "model": s.resolved_model,
                "max_tokens": s.max_tokens,
                "system": system_prompt,
                "messages": messages,
            }
        else:
            url = f"{base}/chat/completions"
            if s.api_key:
                headers["authorization"] = f"Bearer {s.api_key}"
            payload = {
                "model": s.resolved_model,
                "max_tokens": s.max_tokens,
                "messages": [{"role": "system", "content": system_prompt}, *messages],
            }

        if s.temperature is not None:
            payload["temperature"] = s.temperature
        return url, headers, payload

    # ---- HTTP + error handling ------------------------------------------
    def _post_with_retries(self, url: str, headers: dict, payload: dict) -> dict:
        last_problem = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=self.settings.timeout)
            except requests.exceptions.Timeout:
                last_problem = "the AI service took too long to respond"
            except requests.exceptions.ConnectionError:
                last_problem = "could not connect to the AI service"
            except requests.exceptions.RequestException as exc:
                raise LLMError(f"Request to the AI service failed ({type(exc).__name__}).") from exc
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise LLMError("The AI service returned a response that is not valid JSON.") from exc
                if status in (401, 403):
                    raise LLMError(
                        "The AI service rejected the API key (authentication failed). "
                        "Please check LLM_API_KEY in your .env file."
                    )
                if status in RETRYABLE_STATUS:
                    last_problem = (
                        "the AI service is rate-limiting requests"
                        if status == 429
                        else f"the AI service had a temporary problem (HTTP {status})"
                    )
                else:
                    raise LLMError(f"The AI service returned an error (HTTP {status}): {self._error_detail(response)}")

            if attempt < self.max_retries:
                self._sleep(2**attempt)  # 1s, 2s, ... exponential back-off

        raise LLMError(f"Giving up after {self.max_retries + 1} attempts: {last_problem}.")

    @staticmethod
    def _error_detail(response) -> str:
        """Best-effort extraction of the provider's error message (never raises)."""
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else body
            if isinstance(error, dict):
                error = error.get("message", error)
            return str(error)[:200]
        except Exception:  # noqa: BLE001 - we only want a short description
            return (getattr(response, "text", "") or "no details provided")[:200]

    # ---- response parsing -----------------------------------------------
    def _parse_response(self, data: dict) -> str:
        try:
            if self._uses_anthropic_protocol:
                blocks = data["content"]
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            else:
                text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError("The AI service returned a response in an unexpected format.") from exc

        text = (text or "").strip()
        if not text:
            raise LLMError("The AI service returned an empty response.")
        return text


def create_llm_client(settings: Settings) -> Optional[LLMClient]:
    """Return an LLM client for the configured provider, or None for retrieval-only mode."""
    if settings.provider == "none":
        return None
    return LLMClient(settings)

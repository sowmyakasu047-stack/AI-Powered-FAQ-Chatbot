"""Application configuration, read from environment variables (and an optional .env file).

Keeping configuration in one place means no API key or tunable value is ever
hard-coded in the rest of the code base.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Per-provider defaults. "ollama" and "openai" both speak the OpenAI-compatible
# chat-completions protocol; "anthropic" uses the Anthropic Messages API.
PROVIDER_DEFAULTS = {
    "anthropic": {"model": "claude-haiku-5-5", "base_url": "https://api.anthropic.com", "needs_key": True},
    "openai": {"model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1", "needs_key": True},
    "ollama": {"model": "llama3.2", "base_url": "http://localhost:11434/v1", "needs_key": False},
    "none": {"model": "", "base_url": "", "needs_key": False},
}


class ConfigError(ValueError):
    """Raised when an environment variable has an unusable value."""


def _number(env: Mapping[str, str], key: str, default, cast: Callable):
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return cast(str(raw).strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must be a valid {cast.__name__}, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """All runtime settings. `api_key` is excluded from repr so it never lands in logs."""

    provider: str = "anthropic"
    api_key: str = field(default="", repr=False)
    model: str = ""
    base_url: str = ""
    temperature: Optional[float] = 0.2
    max_tokens: int = 400
    timeout: float = 30.0
    faq_path: Path = PROJECT_ROOT / "data" / "faqs.json"
    top_k: int = 3
    min_score: float = 0.25
    max_history_messages: int = 8
    max_input_chars: int = 500

    # ---- derived values -------------------------------------------------
    @property
    def resolved_model(self) -> str:
        return self.model or PROVIDER_DEFAULTS.get(self.provider, {}).get("model", "")

    @property
    def resolved_base_url(self) -> str:
        return (self.base_url or PROVIDER_DEFAULTS.get(self.provider, {}).get("base_url", "")).rstrip("/")

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, *, load_dotenv_file: bool = True) -> "Settings":
        """Build settings from `env` (defaults to os.environ, after loading `.env`)."""
        if env is None:
            if load_dotenv_file:
                try:
                    from dotenv import load_dotenv

                    load_dotenv(PROJECT_ROOT / ".env")
                except ImportError:  # python-dotenv is optional at runtime
                    pass
            env = os.environ

        provider = env.get("LLM_PROVIDER", "anthropic").strip().lower() or "anthropic"
        api_key = (env.get("LLM_API_KEY") or env.get(f"{provider.upper()}_API_KEY") or "").strip()

        faq_path = Path(env.get("FAQ_PATH") or cls.faq_path)
        if not faq_path.is_absolute():
            faq_path = PROJECT_ROOT / faq_path

        raw_temp = env.get("LLM_TEMPERATURE", "0.2")
        temperature = None if str(raw_temp).strip() == "" else _number(env, "LLM_TEMPERATURE", 0.2, float)

        return cls(
            provider=provider,
            api_key=api_key,
            model=(env.get("LLM_MODEL") or "").strip(),
            base_url=(env.get("LLM_BASE_URL") or "").strip(),
            temperature=temperature,
            max_tokens=_number(env, "LLM_MAX_TOKENS", 400, int),
            timeout=_number(env, "LLM_TIMEOUT", 30.0, float),
            faq_path=faq_path,
            top_k=_number(env, "TOP_K", 3, int),
            min_score=_number(env, "MIN_SCORE", 0.25, float),
            max_history_messages=_number(env, "MAX_HISTORY_MESSAGES", 8, int),
            max_input_chars=_number(env, "MAX_INPUT_CHARS", 500, int),
        )

    # ---- validation -----------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty list means all good)."""
        problems: list[str] = []
        if self.provider not in PROVIDER_DEFAULTS:
            problems.append(
                f"LLM_PROVIDER '{self.provider}' is not supported. Choose one of: {', '.join(PROVIDER_DEFAULTS)}."
            )
        elif PROVIDER_DEFAULTS[self.provider]["needs_key"] and not self.api_key:
            problems.append(
                f"No API key found for provider '{self.provider}'. Set LLM_API_KEY in your .env file "
                "(or use LLM_PROVIDER=ollama / none)."
            )
        if not Path(self.faq_path).exists():
            problems.append(f"FAQ file not found: {self.faq_path}")
        if self.top_k < 1:
            problems.append("TOP_K must be at least 1.")
        if not 0 <= self.min_score <= 1:
            problems.append("MIN_SCORE must be between 0 and 1.")
        if self.max_history_messages < 0:
            problems.append("MAX_HISTORY_MESSAGES cannot be negative.")
        return problems

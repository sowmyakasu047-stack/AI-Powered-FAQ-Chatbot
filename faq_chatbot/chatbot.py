"""Conversation logic: retrieve FAQs -> build prompt -> call the LLM -> manage history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Settings
from .llm_client import LLMError, create_llm_client
from .prompts import (
    EMPTY_REPLY,
    LLM_ERROR_NOTICE,
    NO_ANSWER_REPLY,
    SYSTEM_PROMPT,
    TOO_LONG_REPLY,
    build_user_prompt,
)
from .retriever import FAQRetriever, RetrievalResult, load_faqs

SHORT_FOLLOWUP_WORDS = 4  # messages this short are treated as possible follow-ups


@dataclass
class BotReply:
    """What the UI needs to show for one turn."""

    text: str
    sources: list[RetrievalResult] = field(default_factory=list)
    used_llm: bool = False
    error: Optional[str] = None  # technical reason when the LLM failed (safe to display)


class FAQChatbot:
    def __init__(
        self,
        retriever: FAQRetriever,
        llm=None,
        *,
        top_k: int = 3,
        max_history_messages: int = 8,
        max_input_chars: int = 500,
    ):
        self.retriever = retriever
        self.llm = llm  # any object with .generate(system_prompt, messages); None = retrieval-only
        self.top_k = top_k
        self.max_history_messages = max_history_messages
        self.max_input_chars = max_input_chars
        self._history: list[dict[str, str]] = []

    # ---- conversation state ---------------------------------------------
    @property
    def history(self) -> list[dict[str, str]]:
        """A copy of the stored conversation: alternating 'user' / 'assistant' messages."""
        return [dict(message) for message in self._history]

    def reset(self) -> None:
        self._history.clear()

    def _remember(self, user_text: str, assistant_text: str) -> None:
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": assistant_text})
        excess = len(self._history) - self.max_history_messages
        if excess > 0:
            # Drop whole user/assistant pairs so the history always starts with a user message.
            del self._history[: excess + (excess % 2)]

    # ---- retrieval --------------------------------------------------------
    def _retrieve(self, text: str) -> list[RetrievalResult]:
        results = self.retriever.search(text, top_k=self.top_k)
        if results or len(text.split()) > SHORT_FOLLOWUP_WORDS:
            return results
        # Very short message with no match (e.g. "and for students?"): it may be a
        # follow-up, so retry together with the previous question.
        previous = next((m["content"] for m in reversed(self._history) if m["role"] == "user"), None)
        if previous:
            return self.retriever.search(f"{previous} {text}", top_k=self.top_k)
        return results

    # ---- main entry point -------------------------------------------------
    def ask(self, message: str) -> BotReply:
        text = " ".join((message or "").split())
        if not text:
            return BotReply(EMPTY_REPLY)
        if len(text) > self.max_input_chars:
            return BotReply(TOO_LONG_REPLY.format(limit=self.max_input_chars))

        results = self._retrieve(text)
        if not results:
            self._remember(text, NO_ANSWER_REPLY)
            return BotReply(NO_ANSWER_REPLY)

        if self.llm is None:  # retrieval-only mode
            answer = results[0].faq.answer
            self._remember(text, answer)
            return BotReply(answer, sources=results)

        messages = self._history + [{"role": "user", "content": build_user_prompt(text, results)}]
        try:
            answer = self.llm.generate(SYSTEM_PROMPT, messages)
        except LLMError as exc:
            # Degrade gracefully: show the best FAQ answer verbatim. The failed turn is
            # not stored, so the user can simply try again.
            return BotReply(LLM_ERROR_NOTICE + results[0].faq.answer, sources=results, error=str(exc))

        self._remember(text, answer)
        return BotReply(answer, sources=results, used_llm=True)


def build_chatbot(settings: Settings) -> FAQChatbot:
    """Wire everything together from settings (used by the Streamlit app)."""
    faqs = load_faqs(settings.faq_path)
    retriever = FAQRetriever(faqs, min_score=settings.min_score)
    return FAQChatbot(
        retriever,
        create_llm_client(settings),
        top_k=settings.top_k,
        max_history_messages=settings.max_history_messages,
        max_input_chars=settings.max_input_chars,
    )

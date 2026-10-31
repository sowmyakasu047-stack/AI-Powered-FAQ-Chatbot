"""Conversation logic: retrieve FAQs -> build prompt -> call the LLM -> manage history."""

from __future__ import annotations

import re
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

# "Do you have / sell / offer ...?" style questions. A model tends to answer these with a confident
# "no" even when the FAQ says nothing, so they get a guard (see FAQChatbot._asks_about_unknown_thing).
_EXISTENCE_QUESTION = re.compile(
    r"^\s*(?:(?:do|does)\s+(?:you|your|the\s+\w+)\s+(?:have|sell|offer|support|provide|accept|ship|include)"
    r"|(?:is|are)\s+there"
    r"|can\s+i\s+pay\s+(?:with|using|by))\b",
    re.IGNORECASE,
)

# Messages that start like this ("and for students?", "also for teams?") continue the previous topic.
_FOLLOWUP_OPENER = re.compile(r"^\s*(?:and|also|plus)\b", re.IGNORECASE)

_CONTEXT_BLOCK = re.compile(r"<faq_context>.*?</faq_context>", re.DOTALL | re.IGNORECASE)
_STRAY_TAG = re.compile(r"</?faq_context>", re.IGNORECASE)


def clean_answer(text: str) -> str:
    """Remove prompt scaffolding that small models sometimes echo back (the <faq_context> block)."""
    text = _CONTEXT_BLOCK.sub("", text)
    text = _STRAY_TAG.sub("", text)
    text = re.sub(r"^\s*Customer question:.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    return text.strip()


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
        """Search for `text`; for short follow-ups also search with the previous question.

        A short message such as "and for subscriptions?" depends on the earlier topic
        (refunds), so its own matches and the matches for "<previous question> <message>"
        are merged, keeping each FAQ's best score.
        """
        own = self.retriever.search(text, top_k=self.top_k)
        if len(text.split()) > SHORT_FOLLOWUP_WORDS:
            return own
        previous = next((m["content"] for m in reversed(self._history) if m["role"] == "user"), None)
        if not previous:
            return own

        combined = self.retriever.search(f"{previous} {text}", top_k=self.top_k)
        if _FOLLOWUP_OPENER.match(text):
            # An explicit follow-up ("and for subscriptions?"): the earlier topic comes first,
            # the follow-up's own matches fill the remaining slots.
            merged: list[RetrievalResult] = []
            seen: set[str] = set()
            for result in combined + own:
                if result.faq.id not in seen:
                    seen.add(result.faq.id)
                    merged.append(result)
            return merged[: self.top_k]

        # Otherwise merge both searches by best score, so a genuine change of topic still wins.
        best: dict[str, RetrievalResult] = {}
        for result in combined + own:
            if result.faq.id not in best or result.score > best[result.faq.id].score:
                best[result.faq.id] = result
        return sorted(best.values(), key=lambda r: r.score, reverse=True)[: self.top_k]

    def _previous_question(self) -> str | None:
        return next((m["content"] for m in reversed(self._history) if m["role"] == "user"), None)

    def _asks_about_unknown_thing(self, text: str, results: list[RetrievalResult]) -> bool:
        """True for 'do you have X?' questions where X (e.g. 'linux', 'laptops') is in none of the retrieved FAQs."""
        uncovered_terms = getattr(self.retriever, "uncovered_terms", None)
        return bool(_EXISTENCE_QUESTION.match(text) and uncovered_terms and uncovered_terms(text, results))

    # ---- main entry point -------------------------------------------------
    def ask(self, message: str) -> BotReply:
        text = " ".join((message or "").split())
        if not text:
            return BotReply(EMPTY_REPLY)
        if len(text) > self.max_input_chars:
            return BotReply(TOO_LONG_REPLY.format(limit=self.max_input_chars))

        results = self._retrieve(text)
        if not results or self._asks_about_unknown_thing(text, results):
            self._remember(text, NO_ANSWER_REPLY)
            return BotReply(NO_ANSWER_REPLY)

        if self.llm is None:  # retrieval-only mode
            answer = results[0].faq.answer
            self._remember(text, answer)
            return BotReply(answer, sources=results)

        previous = self._previous_question() if _FOLLOWUP_OPENER.match(text) else None
        messages = self._history + [{"role": "user", "content": build_user_prompt(text, results, previous)}]
        try:
            answer = self.llm.generate(SYSTEM_PROMPT, messages)
        except LLMError as exc:
            # Degrade gracefully: show the best FAQ answer verbatim. The failed turn is
            # not stored, so the user can simply try again.
            return BotReply(LLM_ERROR_NOTICE + results[0].faq.answer, sources=results, error=str(exc))

        answer = clean_answer(answer)
        if not answer:  # the model only echoed the prompt: fall back to the best FAQ answer
            answer = results[0].faq.answer
            self._remember(text, answer)
            return BotReply(answer, sources=results)

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
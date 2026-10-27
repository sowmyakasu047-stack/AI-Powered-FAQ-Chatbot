"""Test doubles so the suite never touches the network or a real LLM."""

from __future__ import annotations

from faq_chatbot.retriever import RetrievalResult


class FakeLLM:
    """Records every call and returns a canned reply (or raises a canned error)."""

    def __init__(self, reply: str = "Fake LLM answer.", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, list[dict]]] = []

    def generate(self, system_prompt, messages):
        self.calls.append((system_prompt, [dict(m) for m in messages]))
        if self.error:
            raise self.error
        return self.reply


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, json_data=None, text="", invalid_json=False):
        self.status_code = status_code
        self._json = json_data
        self.text = text
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("not json")
        return self._json


class FakeSession:
    """Returns queued responses (or raises queued exceptions) from .post()."""

    def __init__(self, *queue):
        self.queue = list(queue)
        self.requests: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeRetriever:
    """Returns results from a {query: [results]} mapping and records the queries."""

    def __init__(self, mapping: dict[str, list[RetrievalResult]]):
        self.mapping = mapping
        self.queries: list[str] = []

    def search(self, query, top_k=3, min_score=None):
        self.queries.append(query)
        return self.mapping.get(query, [])

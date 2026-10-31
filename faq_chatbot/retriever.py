"""FAQ loading and retrieval.

Retrieval combines two simple, explainable signals:

1. **Keyword overlap** - what share of the (IDF-weighted) query words appear in
   an FAQ's question or keyword list.
2. **TF-IDF cosine similarity** - how close the query is to the whole FAQ entry
   (question + keywords + answer) in a bag-of-words vector space.

No external ML library is needed, which keeps the logic easy to explain.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_FIELDS = ("id", "question", "answer", "keywords")

STOPWORDS = frozenset(
    """
    a an the is are was were be been am do does did can could would should will
    shall may might i me my mine we our you your it its of to in on at for from
    by with about as and or but if so than that this these those there here what
    which who whom when where why how please tell want need get got have has had
    not no yes any some more much many just also too very really ok okay hi
    hello hey thanks thank cant cannot dont doesnt didnt wont isnt im ive
    """.split()
)


# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #
def _stem(word: str) -> str:
    """A tiny suffix stripper so 'payments'/'payment' and 'cancelled'/'cancel' match."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us")):
        word = word[:-1]
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    if len(word) > 3 and word.endswith("y"):
        word = word[:-1] + "i"
    if len(word) > 3 and word[-1] == word[-2] and word[-1] != "s":
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case, split into words, drop stopwords and apply light stemming."""
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    tokens = []
    for word in words:
        word = word.replace("'", "")
        if word and word not in STOPWORDS:
            tokens.append(_stem(word))
    return tokens


# --------------------------------------------------------------------------- #
# Data model + loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FAQ:
    id: str
    question: str
    answer: str
    keywords: tuple[str, ...] = ()
    category: str = "General"


@dataclass(frozen=True)
class RetrievalResult:
    faq: FAQ
    score: float


def load_faqs(path: str | Path) -> list[FAQ]:
    """Load and validate the FAQ knowledge base from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"FAQ file {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"FAQ file {path} must contain a non-empty JSON list.")

    faqs, seen_ids = [], set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"FAQ entry #{index} must be a JSON object.")
        missing = [name for name in REQUIRED_FIELDS if name not in item]
        if missing:
            raise ValueError(f"FAQ entry #{index} is missing fields: {', '.join(missing)}")
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate FAQ id: {item['id']}")
        seen_ids.add(item["id"])
        faqs.append(
            FAQ(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                keywords=tuple(item["keywords"]),
                category=item.get("category", "General"),
            )
        )
    return faqs


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class FAQRetriever:
    """Ranks FAQ entries against a user query."""

    OVERLAP_WEIGHT = 0.6
    COSINE_WEIGHT = 0.4
    QUESTION_WEIGHT = 3
    KEYWORD_WEIGHT = 3
    ANSWER_WEIGHT = 1

    def __init__(self, faqs: Iterable[FAQ], min_score: float = 0.25):
        self.faqs = list(faqs)
        if not self.faqs:
            raise ValueError("FAQRetriever needs at least one FAQ.")
        self.min_score = min_score
        self._build_index()

    # ---- index ----------------------------------------------------------
    def _build_index(self) -> None:
        raw_weights: list[Counter] = []
        self._match_terms: list[set[str]] = []
        for faq in self.faqs:
            question_tokens = tokenize(faq.question)
            keyword_tokens = tokenize(" ".join(faq.keywords))
            answer_tokens = tokenize(faq.answer)

            weights: Counter = Counter()
            for token in question_tokens:
                weights[token] += self.QUESTION_WEIGHT
            for token in keyword_tokens:
                weights[token] += self.KEYWORD_WEIGHT
            for token in answer_tokens:
                weights[token] += self.ANSWER_WEIGHT

            raw_weights.append(weights)
            self._match_terms.append(set(question_tokens) | set(keyword_tokens))

        doc_freq: Counter = Counter()
        for weights in raw_weights:
            doc_freq.update(weights.keys())
        self._doc_freq = doc_freq
        self._n_docs = len(self.faqs)

        self._vectors: list[dict[str, float]] = []
        self._norms: list[float] = []
        for weights in raw_weights:
            vector = {term: weight * self._idf(term) for term, weight in weights.items()}
            self._vectors.append(vector)
            self._norms.append(math.sqrt(sum(v * v for v in vector.values())) or 1.0)

    def _idf(self, term: str) -> float:
        return math.log((self._n_docs + 1) / (self._doc_freq.get(term, 0) + 1)) + 1.0

    # Verbs that only express "does it exist?", so they are not the thing being asked about.
    EXISTENCE_VERBS = frozenset(tokenize("have sell offer support provide accept ship include pay use"))

    def uncovered_terms(self, query: str, results: Iterable["RetrievalResult"]) -> list[str]:
        """Words in `query` that appear in none of the retrieved FAQ entries (question, keywords or answer)."""
        index_by_id = {faq.id: i for i, faq in enumerate(self.faqs)}
        covered: set[str] = set()
        for result in results:
            covered.update(self._vectors[index_by_id[result.faq.id]])
        wanted = dict.fromkeys(tokenize(query))
        return [token for token in wanted if token not in covered and token not in self.EXISTENCE_VERBS]

    # ---- search ---------------------------------------------------------
    def score(self, query_tokens: list[str]) -> list[float]:
        """Return one combined score (0..1) per FAQ for the given query tokens."""
        unique = set(query_tokens)
        if not unique:
            return [0.0] * self._n_docs

        counts = Counter(query_tokens)
        query_vector = {term: counts[term] * self._idf(term) for term in unique}
        query_norm = math.sqrt(sum(v * v for v in query_vector.values())) or 1.0
        total_idf = sum(self._idf(term) for term in unique)

        scores = []
        for index in range(self._n_docs):
            vector = self._vectors[index]
            dot = sum(weight * vector.get(term, 0.0) for term, weight in query_vector.items())
            cosine = dot / (query_norm * self._norms[index])

            matched_idf = sum(self._idf(t) for t in unique if t in self._match_terms[index])
            overlap = matched_idf / total_idf

            scores.append(self.OVERLAP_WEIGHT * overlap + self.COSINE_WEIGHT * cosine)
        return scores

    def search(self, query: str, top_k: int = 3, min_score: float | None = None) -> list[RetrievalResult]:
        """Return up to `top_k` FAQs scoring at least `min_score`, best first."""
        threshold = self.min_score if min_score is None else min_score
        tokens = tokenize(query)
        if not tokens or top_k < 1:
            return []

        scored = [
            RetrievalResult(faq=faq, score=score)
            for faq, score in zip(self.faqs, self.score(tokens))
            if score >= threshold
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]
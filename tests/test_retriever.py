"""Retrieval tests (TC-04 .. TC-10): valid, paraphrased, irrelevant, ambiguous and empty queries."""

import pytest

# (user query, expected FAQ id) - phrased differently from the FAQ questions on purpose.
PARAPHRASED_QUERIES = [
    ("how can I register for a new account", "account-create"),
    ("I forgot my password", "password-reset"),
    ("can I get my money back", "refund-policy"),
    ("stop my monthly subscription", "cancel-subscription"),
    ("which credit cards do you take", "payment-methods"),
    ("is there a student discount", "discounts"),
    ("will I get a certificate when I finish", "certificates"),
    ("can I watch videos without internet", "offline-download"),
    ("video keeps buffering", "video-playback"),
    ("how do I talk to customer support", "support-contact"),
    ("I want to teach a course on your platform", "teach"),
    ("do you have plans for companies", "team-plans"),
    ("turn on 2FA", "two-factor"),
    ("close my account permanently", "delete-account"),
    ("what browsers work with the site", "system-requirements"),
    ("does the platform support screen readers", "accessibility"),
    ("how long will I have access to the course", "course-access-duration"),
    ("can I retake the final exam", "quiz-retakes"),
    ("my card was declined", "failed-payment"),
    ("I never got the confirmation email", "verify-email"),
]

IRRELEVANT_QUERIES = [
    "what is the capital of France",
    "who won the world cup",
    "write me a poem about the sea",
    "how do I bake a chocolate cake",
    "tell me a joke",
]


def test_tc04_exact_faq_question_returns_that_faq_first(faqs, retriever):
    """TC-04: asking a FAQ's own question retrieves that FAQ as the top result."""
    misses = [f.id for f in faqs if not (r := retriever.search(f.question)) or r[0].faq.id != f.id]
    assert misses == []


def test_tc05_paraphrased_queries_reach_the_right_faq(retriever):
    """TC-05: retrieval accuracy on reworded questions is at least 90% (top-3 contains the answer)."""
    failures = []
    for query, expected in PARAPHRASED_QUERIES:
        ids = [r.faq.id for r in retriever.search(query, top_k=3)]
        if expected not in ids:
            failures.append((query, expected, ids))
    accuracy = 1 - len(failures) / len(PARAPHRASED_QUERIES)
    assert accuracy >= 0.9, f"accuracy {accuracy:.0%}; misses: {failures}"


def test_tc05b_best_match_is_top_ranked_for_most_paraphrases(retriever):
    """TC-05b: the expected FAQ is ranked #1 for at least 80% of reworded questions."""
    top1 = sum(
        1
        for query, expected in PARAPHRASED_QUERIES
        if (results := retriever.search(query, top_k=1)) and results[0].faq.id == expected
    )
    assert top1 / len(PARAPHRASED_QUERIES) >= 0.8


@pytest.mark.parametrize("query", IRRELEVANT_QUERIES)
def test_tc06_irrelevant_queries_return_nothing(retriever, query):
    """TC-06: questions unrelated to the FAQ do not match anything."""
    assert retriever.search(query) == []


@pytest.mark.parametrize("query", ["", "   ", "???", "!!! ...", None])
def test_tc07_empty_or_punctuation_only_queries_return_nothing(retriever, query):
    """TC-07: empty, whitespace, punctuation-only or None input is handled without errors."""
    assert retriever.search(query) == []


def test_tc08_ambiguous_one_word_query_returns_ranked_candidates(retriever):
    """TC-08: a vague query ('account') does not crash and returns a sorted, bounded list."""
    results = retriever.search("account", top_k=3, min_score=0.0)
    assert 1 <= len(results) <= 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_tc09_search_ignores_case_and_punctuation(retriever):
    """TC-09: 'HOW DO I RESET MY PASSWORD???' behaves like the plain lower-case question."""
    plain = retriever.search("how do i reset my password")
    shouty = retriever.search("HOW DO I RESET MY PASSWORD???")
    assert plain and shouty
    assert plain[0].faq.id == shouty[0].faq.id == "password-reset"


def test_tc10_top_k_limits_results_and_scores_are_sorted(retriever):
    """TC-10: top_k is respected and results are ordered best-first."""
    one = retriever.search("subscription payment", top_k=1, min_score=0.0)
    five = retriever.search("subscription payment", top_k=5, min_score=0.0)
    assert len(one) == 1
    assert len(five) == 5
    assert five[0].faq.id == one[0].faq.id
    assert [r.score for r in five] == sorted((r.score for r in five), reverse=True)

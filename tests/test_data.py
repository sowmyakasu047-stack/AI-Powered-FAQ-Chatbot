"""Knowledge-base tests (TC-01 .. TC-03)."""

import json

import pytest

from faq_chatbot.retriever import load_faqs


def test_tc01_knowledge_base_has_at_least_30_faqs(faqs):
    """TC-01: the dataset meets the minimum size requirement."""
    assert len(faqs) >= 30


def test_tc02_every_entry_is_complete_and_ids_are_unique(faqs):
    """TC-02: each FAQ has an id, question, answer and keywords; ids are unique."""
    ids = [faq.id for faq in faqs]
    assert len(ids) == len(set(ids))
    for faq in faqs:
        assert faq.question.strip().endswith("?")
        assert len(faq.answer.strip()) > 20
        assert len(faq.keywords) >= 3


def test_tc03_bad_faq_files_fail_with_clear_errors(tmp_path):
    """TC-03: a missing, malformed or incomplete FAQ file raises a helpful error."""
    with pytest.raises(FileNotFoundError):
        load_faqs(tmp_path / "missing.json")

    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_faqs(broken)

    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps([{"id": "x", "question": "Q?"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_faqs(incomplete)

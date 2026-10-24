from pathlib import Path

import pytest

from faq_chatbot.retriever import FAQRetriever, load_faqs

FAQ_FILE = Path(__file__).resolve().parents[1] / "data" / "faqs.json"


@pytest.fixture(scope="session")
def faqs():
    return load_faqs(FAQ_FILE)


@pytest.fixture(scope="session")
def retriever(faqs):
    return FAQRetriever(faqs)

"""LLM client and configuration tests (TC-22 .. TC-28). A fake HTTP session is used: no network."""

import pytest
import requests

from faq_chatbot.config import Settings
from faq_chatbot.llm_client import LLMClient, LLMError, create_llm_client
from tests.fakes import FakeResponse, FakeSession

MESSAGES = [{"role": "user", "content": "Hello"}]
ANTHROPIC_OK = {"content": [{"type": "text", "text": "Hi there!"}]}
OPENAI_OK = {"choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]}


def client(settings, *responses, retries=2):
    session = FakeSession(*responses)
    return LLMClient(settings, session=session, max_retries=retries, sleep=lambda _s: None), session


def test_tc22_anthropic_request_and_response():
    """TC-22: Anthropic call has the right URL, auth headers, system prompt and payload; reply text is parsed."""
    settings = Settings(provider="anthropic", api_key="sk-test-123", max_tokens=123, temperature=0.1)
    llm, session = client(settings, FakeResponse(200, ANTHROPIC_OK))

    assert llm.generate("You are helpful.", MESSAGES) == "Hi there!"

    sent = session.requests[0]
    assert sent["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["headers"]["x-api-key"] == "sk-test-123"
    assert "anthropic-version" in sent["headers"]
    assert sent["json"]["system"] == "You are helpful."
    assert sent["json"]["messages"] == MESSAGES
    assert sent["json"]["max_tokens"] == 123
    assert sent["json"]["model"] == "claude-haiku-5-5"


def test_tc23_openai_compatible_request_and_local_model_needs_no_key():
    """TC-23: OpenAI-style call puts the system prompt first; a local Ollama model sends no auth header."""
    settings = Settings(provider="ollama", api_key="", temperature=None)
    llm, session = client(settings, FakeResponse(200, OPENAI_OK))

    assert llm.generate("Be brief.", MESSAGES) == "Hi there!"

    sent = session.requests[0]
    assert sent["url"] == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in sent["headers"]
    assert sent["json"]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert "temperature" not in sent["json"]


def test_tc24_bad_api_key_gives_clear_error_without_leaking_the_key():
    """TC-24: a 401 raises LLMError that points at the API key setting and never prints the key."""
    settings = Settings(provider="anthropic", api_key="sk-secret-value")
    llm, session = client(settings, FakeResponse(401, {"error": {"message": "invalid x-api-key"}}))

    with pytest.raises(LLMError) as excinfo:
        llm.generate("sys", MESSAGES)

    assert "API key" in str(excinfo.value)
    assert "sk-secret-value" not in str(excinfo.value)
    assert len(session.requests) == 1  # authentication errors are not retried


def test_tc25_timeouts_are_retried_then_reported():
    """TC-25: repeated timeouts end in a friendly LLMError after the configured number of attempts."""
    settings = Settings(provider="openai", api_key="k")
    llm, session = client(settings, *[requests.exceptions.Timeout()] * 3)

    with pytest.raises(LLMError, match="too long"):
        llm.generate("sys", MESSAGES)

    assert len(session.requests) == 3


def test_tc26_rate_limit_is_retried_and_then_succeeds():
    """TC-26: a 429 followed by a 200 returns the answer (retry with back-off)."""
    settings = Settings(provider="anthropic", api_key="k")
    llm, session = client(settings, FakeResponse(429, {}), FakeResponse(200, ANTHROPIC_OK))

    assert llm.generate("sys", MESSAGES) == "Hi there!"
    assert len(session.requests) == 2


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(200, invalid_json=True),
        FakeResponse(200, {"unexpected": "shape"}),
        FakeResponse(200, {"content": []}),
        FakeResponse(400, {"error": {"message": "bad request"}}),
    ],
    ids=["invalid-json", "wrong-shape", "empty-content", "http-400"],
)
def test_tc27_malformed_or_rejected_responses_raise_llm_error(response):
    """TC-27: bad JSON, unexpected structure, empty replies and 4xx errors all become LLMError."""
    llm, _ = client(Settings(provider="anthropic", api_key="k"), response)
    with pytest.raises(LLMError):
        llm.generate("sys", MESSAGES)


def test_tc28_configuration_validation_and_env_loading(tmp_path):
    """TC-28: missing keys are reported, local/none providers need none, and env vars override defaults."""
    missing_key = Settings.from_env({"LLM_PROVIDER": "anthropic"}, load_dotenv_file=False)
    assert any("API key" in p for p in missing_key.validate())

    assert Settings.from_env({"LLM_PROVIDER": "ollama"}).validate() == []
    assert Settings.from_env({"LLM_PROVIDER": "none"}).validate() == []
    assert any("not supported" in p for p in Settings.from_env({"LLM_PROVIDER": "bogus"}).validate())

    configured = Settings.from_env(
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "abc", "LLM_MODEL": "my-model", "TOP_K": "5", "LLM_TEMPERATURE": ""}
    )
    assert configured.api_key == "abc"
    assert configured.resolved_model == "my-model"
    assert configured.top_k == 5
    assert configured.temperature is None
    assert "abc" not in repr(configured)  # keys must never show up in logs

    with pytest.raises(ValueError, match="TOP_K"):
        Settings.from_env({"TOP_K": "many"})


def test_create_llm_client_returns_none_in_retrieval_only_mode():
    assert create_llm_client(Settings(provider="none")) is None
    assert isinstance(create_llm_client(Settings(provider="ollama")), LLMClient)

import pytest
from pydantic import ValidationError
import os

from communitydocs_rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_S
from communitydocs_rag.llm.schema import SimpleResult
from communitydocs_rag.llm.generate import generate_structured
from communitydocs_rag.llm.client import OllamaClient

# Test A: Pydantic schema validation should reject wrong types
def test_pydantic_rejects_wrong_types():
    # confidence should be a float in [0, 1], not a string.
    bad = {
        "title": "Example",
        "items": ["a", "b", "c"],
        "confidence": "high",
    }

    with pytest.raises(ValidationError):
        SimpleResult.model_validate(bad)

# Test B: If LLM returns invalid JSON, the generate_structured function should
# retry and eventually succeed if a valid output is returned.
class FakeClient:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def generate(self, prompt: str) -> str:
        # Track calls, return next output in sequence.
        out = self.outputs[self.calls]
        self.calls += 1
        return out
    
def test_repair_prompt_triggered_and_succeeds():
    # First output is invalid JSON -> triggers retry.
    invalid = "not json at all"
    # Second output is valid JSON matching schema -> should succeed.
    valid = '{"title":"OK","items":["x","y","z"],"confidence":0.8}'

    client = FakeClient([invalid, valid])

    result = generate_structured(
        client=client,
        schema=SimpleResult,
        user_task="Return a title, 3 items, and confidence.",
        max_retries=2,
    )

    assert result.title == "OK"
    assert result.items == ["x", "y", "z"]
    assert result.confidence == 0.8
    assert client.calls == 2

# Test C: If LLM returns valid JSON but fails schema validation,
# it should retry and succeed if a valid output is returned.
def test_retry_on_schema_validation_error_then_succeeds():
    # Valid JSON, but wrong type for confidence -> schema validation should fail.
    invalid_schema = '{"title":"Bad","items":["x"],"confidence":"high"}'
    valid = '{"title":"Good","items":["x","y","z"],"confidence":0.6}'

    client = FakeClient([invalid_schema, valid])

    result = generate_structured(
        client=client,
        schema=SimpleResult,
        user_task="Return a title, 3 items, and confidence.",
        max_retries=2,
    )

    assert result.title == "Good"
    assert client.calls == 2

# Test D: Integration test with real Ollama client. Requires 
# Ollama running and configured.
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run integration tests (requires Ollama running).",
)
def test_integration_real_ollama_structured_output():
    client = OllamaClient(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        timeout_s=OLLAMA_TIMEOUT_S,
    )

    result = generate_structured(
        client=client,
        schema=SimpleResult,
        user_task="Return JSON only with a short title, 3 short items, and confidence between 0 and 1.",
        max_retries=2,
    )

    assert isinstance(result.title, str)
    assert len(result.items) == 3
    assert 0.0 <= result.confidence <= 1.0
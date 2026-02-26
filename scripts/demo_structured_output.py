from communitydocs_rag.logging_setup import setup_logging
from communitydocs_rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_S
from communitydocs_rag.llm.client import OllamaClient
from communitydocs_rag.llm.generate import generate_structured
from communitydocs_rag.llm.schema import SimpleResult


def main() -> None:
    # 1) Logging (so you can see retries, timing, errors)
    setup_logging()

    # 2) Create the Ollama client using env-driven config
    client = OllamaClient(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        timeout_s=OLLAMA_TIMEOUT_S,
    )

    # 3) Define a tiny task that forces structured output
    task = (
        "Create a short title and exactly 3 short bullet items about why quiet cafés help conversation. "
        "Return a confidence number between 0 and 1. Output ONLY JSON."
    )

    # 4) Call the structured generator (JSON parse + Pydantic validation + retries)
    result = generate_structured(
        client=client,
        schema=SimpleResult,
        user_task=task,
        max_retries=2,
    )

    # 5) Print validated output as JSON (this is your demo artifact)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
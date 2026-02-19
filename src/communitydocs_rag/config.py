import os
from dotenv import load_dotenv


load_dotenv()

# Ollama configuration
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:7b-instruct",
)

OLLAMA_TIMEOUT_S = int(
    os.getenv(
        "OLLAMA_TIMEOUT_S",
        "60",
    )
)

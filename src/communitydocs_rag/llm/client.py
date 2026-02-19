import time
from typing import Any

import httpx

from communitydocs_rag.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_S,
)
from communitydocs_rag.logging_setup import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """
    Thin HTTP client for Ollama.
    
    Sends prompts to Ollama's generate endpoint, captures latency, and raises useful errors.
    """

    def __init__(self, base_url: str, model: str, timeout_s: int) -> None:
        """
        Initialise the Ollama client.
        
        Args:
            base_url: Ollama server base URL (e.g., "http://localhost:11434")
            model: Model name to use (e.g., "qwen2.5:7b-instruct")
            timeout_s: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        """
        Send a prompt to Ollama and return the text response.
        
        Args:
            prompt: The input prompt/question
            
        Returns:
            The generated text response from Ollama
            
        Raises:
            httpx.TimeoutException: If request exceeds timeout
            httpx.ConnectError: If unable to connect to Ollama
            ValueError: If response format is unexpected
        """
        # Log the start of the request
        logger.info(f"Ollama request: prompt={prompt[:50]}...")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Build the request URL and payload
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,  # Important: non-streaming for simpler parsing
            }
            
            # Make the POST request
            with httpx.Client() as client:
                response = client.post(
                    url,
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()  # Raise exception for bad status codes
            
            # Parse the JSON response
            result: dict[str, Any] = response.json()
            
            # Extract the generated text
            if "response" not in result:
                raise ValueError(
                    f"Unexpected Ollama response format. Expected 'response' field, got: {list(result.keys())}"
                )
            
            generated_text = result["response"]
            
            # Log success with elapsed time
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Ollama request completed in {elapsed_ms:.0f}ms")
            
            return generated_text
            
        except httpx.TimeoutException as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Ollama request timed out after {elapsed_ms:.0f}ms: {e}")
            raise
            
        except httpx.ConnectError as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Failed to connect to Ollama at {self.base_url} "
                f"(after {elapsed_ms:.0f}ms): {e}"
            )
            raise
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Ollama request failed after {elapsed_ms:.0f}ms: {e}")
            raise


# Initialise the global client instance
client = OllamaClient(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    timeout_s=OLLAMA_TIMEOUT_S,
)

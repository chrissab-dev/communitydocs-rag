from __future__ import annotations

import json
import time
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from communitydocs_rag.logging_setup import get_logger
from communitydocs_rag.llm.client import OllamaClient

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredGenerationError(RuntimeError):
    """
    Raised when we cannot produce valid JSON that conforms to the given Pydantic schema.
    Contains the last raw model output and the last validation/parsing error message.
    """

    def __init__(
        self,
        message: str,
        last_raw_output: str,
        last_error: str,
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.last_raw_output = last_raw_output
        self.last_error = last_error
        self.attempts = attempts


def _schema_hint(schema: Type[T]) -> str:
    """
    Produce a compact JSON-schema-ish hint to show the model the required keys/types.
    We don't dump the entire JSON schema (can be long); we use Pydantic's JSON schema
    and keep only the properties + required keys.
    """
    full = schema.model_json_schema()
    props = full.get("properties", {})
    required = full.get("required", [])

    # Build a compact representation the model can follow.
    hint: dict[str, Any] = {
        "type": "object",
        "required_keys": required,
        "properties": {},
    }

    for k, v in props.items():
        # Keep only type-ish info to reduce prompt length
        hint["properties"][k] = {
            "type": v.get("type"),
            "description": v.get("description"),
        }

    return json.dumps(hint, ensure_ascii=False, indent=2)


def _build_prompt(
    user_task: str,
    schema: Type[T],
    *,
    previous_invalid: Optional[str] = None,
    repair_mode: bool = False,
) -> str:
    """
    Build a prompt that strongly pushes the model to output strict JSON only.
    If repair_mode is True, the prompt focuses on fixing invalid output.
    """
    schema_hint = _schema_hint(schema)

    rules = (
        "RULES:\n"
        "1) Output ONLY valid JSON. No markdown. No explanation. No code fences.\n"
        "2) The JSON keys MUST match the schema exactly (no extra keys, no missing keys).\n"
        "3) Strings MUST be in double quotes.\n"
        "4) No trailing commas.\n"
        "5) If there is a confidence field, it MUST be a number between 0 and 1.\n"
    )

    if not repair_mode:
        prompt = (
            f"{rules}\n"
            "SCHEMA (follow this):\n"
            f"{schema_hint}\n\n"
            "TASK:\n"
            f"{user_task}\n"
        )
        return prompt

    # Repair mode: focus on fixing the previous output into correct JSON
    prompt = (
        f"{rules}\n"
        "SCHEMA (follow this):\n"
        f"{schema_hint}\n\n"
        "Your previous output was invalid JSON or did not match the schema.\n"
        "Repair it and output ONLY corrected JSON that matches the schema.\n\n"
        "INVALID OUTPUT:\n"
        f"{previous_invalid or ''}\n"
    )
    return prompt


def generate_structured(
    *,
    client: OllamaClient,
    schema: Type[T],
    user_task: str,
    max_retries: int = 2,
) -> T:
    """
    Call the LLM to complete user_task and return a validated instance of `schema`.

    Retries:
      - Attempt 1: normal prompt
      - Attempt 2: repair prompt
      - Attempt 3: (optional) repair prompt including invalid output (we already include it)

    max_retries=2 means total attempts = 1 + max_retries = 3 attempts max.
    """
    attempts_total = 1 + max_retries
    last_raw: str = ""
    last_err: str = ""

    for attempt in range(1, attempts_total + 1):
        start = time.time()

        if attempt == 1:
            prompt = _build_prompt(user_task=user_task, schema=schema)
        else:
            # Repair mode: include the last raw output so the model can fix it
            prompt = _build_prompt(
                user_task=user_task,
                schema=schema,
                previous_invalid=last_raw,
                repair_mode=True,
            )

        logger.info("LLM structured generation attempt %d/%d", attempt, attempts_total)

        try:
            raw = client.generate(prompt)
            last_raw = raw
            elapsed_ms = int((time.time() - start) * 1000)
            logger.info("LLM response received in %dms (attempt %d)", elapsed_ms, attempt)

            # 1) Parse JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                last_err = f"JSON parse error: {e}"
                logger.warning("Attempt %d failed: %s", attempt, last_err)
                continue

            # 2) Validate schema
            try:
                obj = schema.model_validate(data)
            except ValidationError as e:
                last_err = f"Schema validation error: {e}"
                logger.warning("Attempt %d failed: %s", attempt, last_err)
                continue

            # Success
            logger.info("Structured generation succeeded on attempt %d", attempt)
            return obj

        except Exception as e:
            # Network errors, timeouts, unexpected client errors
            last_err = f"Client/LLM error: {e}"
            logger.exception("Attempt %d failed with exception", attempt)
            continue

    # If we get here, all attempts failed
    raise StructuredGenerationError(
        message=f"Failed to generate valid structured output after {attempts_total} attempts",
        last_raw_output=last_raw,
        last_error=last_err or "Unknown error",
        attempts=attempts_total,
    )
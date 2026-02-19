"""
LLM output validation schemas.

This module defines Pydantic models to validate and parse LLM responses.
SimpleResult is a toy schema for testing the generate + parse pipeline.
TODO Future: Replace with ReviewAnswer, CitedClaimList, etc.
"""

from pydantic import BaseModel, Field


class SimpleResult(BaseModel):
    """
    Toy schema for validation testing.
    
    This is NOT the final schema. Test harness to validate:
    - LLM can produce structured JSON
    - Validation pipeline works
    - Retry logic catches invalid outputs
    
    To replace with domain-specific schemas
    like ReviewAnswer or CitedClaimList.
    """

    title: str = Field(
        ...,
        description="A short title or summary (e.g., 'Café noise assessment')",
        min_length=1,
        max_length=200,
    )

    items: list[str] = Field(
        ...,
        description="A list of supporting points or observations",
        min_items=1,
        max_items=10,
    )

    confidence: float = Field(
        ...,
        description="Confidence level in the result (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )

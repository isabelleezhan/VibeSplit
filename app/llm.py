"""Shared Gemini call helper"""

import asyncio

from fastapi import HTTPException
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.config import get_settings

MODEL = "gemini-3.5-flash"
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 3

REQUEST_TIMEOUT_SECONDS = 45


async def generate_structured(
    system_prompt: str,
    contents: str,
    schema: type[BaseModel],
) -> BaseModel:
    """Calls Gemini with the given prompt, constrained to `schema`'s JSON
    shape, and returns the parsed Pydantic object."""
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key or None)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_json_schema=schema.model_json_schema(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_error: genai_errors.ServerError | TimeoutError | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(model=MODEL, contents=contents, config=config),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return schema.model_validate_json(response.text)
        except (genai_errors.ServerError, TimeoutError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BASE_DELAY_SECONDS * (2 ** (attempt - 1)))

    reason = (
        f"timed out (no response within {REQUEST_TIMEOUT_SECONDS}s)"
        if isinstance(last_error, TimeoutError)
        else "is currently unavailable (high demand on the free tier)"
    )
    raise HTTPException(
        status_code=502,
        detail=f"Gemini API {reason} after {MAX_RETRIES} attempts. Please try again in a minute. ({last_error})",
    )

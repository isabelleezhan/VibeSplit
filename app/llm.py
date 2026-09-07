"""Shared Groq call helper"""

import asyncio
import json
import re

from fastapi import HTTPException
from groq import APIStatusError, AsyncGroq, RateLimitError
from pydantic import BaseModel

from app.config import get_settings


MODEL = "openai/gpt-oss-20b"
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 3

# Bounds how long a single call can run before giving up.
REQUEST_TIMEOUT_SECONDS = 120

MAX_COMPLETION_TOKENS = 20000

_RESET_DURATION_RE = re.compile(r"(?:(\d+)m)?(\d+(?:\.\d+)?)s")


def _parse_reset_seconds(value: str) -> float | None:
    """Parses Groq's rate-limit reset headers, e.g. '5.662s' or
    '10m4.8s', into a plain number of seconds. None if unparseable."""
    match = _RESET_DURATION_RE.match(value)
    if not match:
        return None
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2))
    return minutes * 60 + seconds


async def generate_structured(
    system_prompt: str,
    contents: str,
    schema: type[BaseModel],
) -> BaseModel:
    """Calls Groq with the given prompt, asks for JSON matching `schema`'s
    shape, and returns the parsed Pydantic object. Retries transient rate
    limits, tokens-per-minute limits, and server errors."""
    settings = get_settings()
    client = AsyncGroq(api_key=settings.groq_api_key)

    schema_prompt = (
        f"{system_prompt}\n\n"
        "Respond with ONLY a single JSON object matching this JSON Schema exactly — "
        f"no prose, no markdown code fences:\n{json.dumps(schema.model_json_schema())}"
    )

    last_error: RateLimitError | APIStatusError | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": schema_prompt},
                        {"role": "user", "content": contents},
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                ),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return schema.model_validate_json(response.choices[0].message.content or "")
        except TimeoutError as e:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Groq API timed out (no response within {REQUEST_TIMEOUT_SECONDS}s). "
                    "This wasn't retried, to avoid spending another request on one that's "
                    "unlikely to succeed on an identical retry."
                ),
            ) from e

        except RateLimitError as e:
            last_error = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
        except APIStatusError as e:
            if e.status_code == 413:
  
                last_error = e
                if attempt < MAX_RETRIES:
                    reset_header = e.response.headers.get("x-ratelimit-reset-tokens")
                    wait_seconds = _parse_reset_seconds(reset_header) if reset_header else None
                    if wait_seconds is None:
                        wait_seconds = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                    await asyncio.sleep(wait_seconds + 0.5)  # small buffer past the exact reset instant
                continue
            if e.status_code < 500:
                raise  # a genuinely bad request — retrying wouldn't fix it
            last_error = e
            if attempt < MAX_RETRIES:
                await asyncio.sleep(BASE_DELAY_SECONDS * (2 ** (attempt - 1)))

    if isinstance(last_error, RateLimitError):
        reason = "hit the free tier's rate limit"
    elif isinstance(last_error, APIStatusError) and last_error.status_code == 413:
        reason = "hit the free tier's tokens-per-minute limit"
    else:
        reason = "is currently unavailable (server error)"
    raise HTTPException(
        status_code=502,
        detail=f"Groq API {reason} after {MAX_RETRIES} attempts. Please try again in a minute. ({last_error})",
    )

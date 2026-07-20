import json
import math
import time
from typing import Any, Literal, cast

import httpx
from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    Groq,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from app.config import (
    GROQ_MODEL,
    OPENROUTER_MODEL,
    get_groq_api_key,
    get_openrouter_api_key,
)


LLMProvider = Literal["groq", "openrouter"]
SUPPORTED_LLM_PROVIDERS = {"groq", "openrouter"}
OPENROUTER_API_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)
OPENROUTER_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
STRUCTURED_OUTPUT_TOKEN_LIMIT = 2048
STRUCTURED_FALLBACK_ATTEMPTS = 2
MAX_RATE_LIMIT_WAIT_SECONDS = 90

_groq_client: Groq | None = None


class LLMServiceError(RuntimeError):
    """Raised when an LLM provider cannot produce a usable response."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        retry_after_seconds: int | None = None,
        provider: LLMProvider = "groq",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after_seconds = retry_after_seconds
        self.provider = provider


def normalize_provider(provider: str) -> LLMProvider:
    """Validate and normalize an LLM provider name."""

    if not isinstance(provider, str):
        raise TypeError("provider must be a string.")

    normalized = provider.strip().lower()

    if normalized not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            "provider must be groq or openrouter."
        )

    return cast(LLMProvider, normalized)


def get_groq_client() -> Groq:
    """Create and reuse the Groq API client."""

    global _groq_client

    if _groq_client is None:
        _groq_client = Groq(api_key=get_groq_api_key())

    return _groq_client


def parse_retry_after(value: str | None) -> int | None:
    if not value:
        return None

    try:
        return max(1, math.ceil(float(value)))
    except ValueError:
        return None


def retry_after_seconds(error: RateLimitError) -> int | None:
    """Read Groq's Retry-After response header."""

    return parse_retry_after(
        error.response.headers.get("retry-after")
    )


def create_groq_chat_completion(**kwargs):
    """Create a Groq completion and honor short rate-limit resets."""

    for attempt in range(2):
        try:
            return (
                get_groq_client()
                .chat.completions.create(**kwargs)
            )
        except RateLimitError as error:
            wait_seconds = retry_after_seconds(error)

            if (
                attempt == 0
                and wait_seconds is not None
                and wait_seconds <= MAX_RATE_LIMIT_WAIT_SECONDS
            ):
                time.sleep(wait_seconds + 0.25)
                continue

            raise

    raise RuntimeError("Unreachable rate-limit retry state.")


def wrap_groq_api_error(
    error: APIError,
    message: str,
) -> LLMServiceError:
    """Convert Groq exceptions into safe application errors."""

    wait_seconds = None

    if isinstance(error, APITimeoutError):
        code = "timeout"
    elif isinstance(error, APIConnectionError):
        code = "connection"
    elif isinstance(error, RateLimitError):
        code = "rate_limit"
        wait_seconds = retry_after_seconds(error)
    elif (
        isinstance(error, BadRequestError)
        and is_schema_generation_error(error)
    ):
        code = "structured_output"
    else:
        code = "provider_error"

    return LLMServiceError(
        message,
        code=code,
        retry_after_seconds=wait_seconds,
        provider="groq",
    )


def openrouter_error(response: httpx.Response) -> LLMServiceError:
    """Convert an unsuccessful OpenRouter response into a safe error."""

    if response.status_code == 429:
        code = "rate_limit"
    elif response.status_code in {408, 504}:
        code = "timeout"
    elif response.status_code == 400:
        code = "structured_output"
    else:
        code = "provider_error"

    return LLMServiceError(
        "OpenRouter could not complete the request.",
        code=code,
        retry_after_seconds=parse_retry_after(
            response.headers.get("retry-after")
        ),
        provider="openrouter",
    )


def create_openrouter_completion(
    *,
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> str:
    """Create one non-streaming OpenRouter completion."""

    payload: dict[str, Any] = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "reasoning": {
            "effort": "low",
            "exclude": True,
        },
    }

    if response_format is not None:
        payload["response_format"] = response_format

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    try:
        api_key = get_openrouter_api_key()
    except RuntimeError as error:
        raise LLMServiceError(
            "OpenRouter API key is not configured.",
            code="configuration",
            provider="openrouter",
        ) from error

    last_error: Exception | None = None

    for attempt in range(3):
        try:
            response = httpx.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Knowledge Graph RAG",
                },
                json=payload,
                timeout=OPENROUTER_TIMEOUT,
            )
        except httpx.TimeoutException as error:
            raise LLMServiceError(
                "OpenRouter took too long to respond.",
                code="timeout",
                provider="openrouter",
            ) from error
        except httpx.RequestError as error:
            raise LLMServiceError(
                "OpenRouter could not be reached.",
                code="connection",
                provider="openrouter",
            ) from error

        if response.is_error:
            raise openrouter_error(response)

        try:
            data = response.json()
        except ValueError as error:
            last_error = error
            if attempt < 2:
                continue
            break

        choices = (
            data.get("choices")
            if isinstance(data, dict)
            else None
        )

        if not isinstance(choices, list) or not choices:
            details = (
                data.get("error")
                if isinstance(data, dict)
                else None
            )
            error_code = (
                details.get("code")
                if isinstance(details, dict)
                else None
            )

            if str(error_code) == "429":
                raise LLMServiceError(
                    "OpenRouter's request limit was reached.",
                    code="rate_limit",
                    provider="openrouter",
                )

            last_error = RuntimeError(
                "OpenRouter response contained no choices."
            )
            if attempt < 2:
                continue
            break

        try:
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            last_error = error
            if attempt < 2:
                continue
            break

        if isinstance(content, str) and content.strip():
            return content.strip()

        last_error = RuntimeError(
            "OpenRouter response content was empty."
        )
        if attempt < 2:
            continue

    raise LLMServiceError(
        "OpenRouter returned no usable completion.",
        code="empty_response",
        provider="openrouter",
    ) from last_error


def validate_prompts(
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, str]:
    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string.")

    if not isinstance(user_prompt, str):
        raise TypeError("user_prompt must be a string.")

    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()

    if not system_prompt:
        raise ValueError("system_prompt cannot be empty.")

    if not user_prompt:
        raise ValueError("user_prompt cannot be empty.")

    return system_prompt, user_prompt


def generate_text(
    system_prompt: str,
    user_prompt: str,
    provider: str = "groq",
) -> str:
    """Generate a text response using the selected provider."""

    system_prompt, user_prompt = validate_prompts(
        system_prompt,
        user_prompt,
    )
    selected_provider = normalize_provider(provider)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if selected_provider == "openrouter":
        return create_openrouter_completion(messages=messages)

    try:
        completion = create_groq_chat_completion(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            reasoning_effort="low",
        )
    except APIError as error:
        raise wrap_groq_api_error(
            error,
            "Groq could not generate an answer.",
        ) from error

    if not completion.choices:
        raise LLMServiceError(
            "Groq returned no completion choices.",
            code="empty_response",
            provider="groq",
        )

    content = completion.choices[0].message.content

    if not content or not content.strip():
        raise LLMServiceError(
            "Groq returned an empty response.",
            code="empty_response",
            provider="groq",
        )

    return content.strip()


def is_schema_generation_error(
    error: BadRequestError,
) -> bool:
    """Return whether Groq rejected a generated JSON-schema result."""

    body = error.body

    if not isinstance(body, dict):
        return False

    details = body.get("error")

    return (
        isinstance(details, dict)
        and details.get("code") == "json_validate_failed"
    )


def clean_json_content(content: str) -> str:
    """Remove an optional Markdown fence around a JSON object."""

    cleaned = content.strip()

    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

    return cleaned.strip()


def validate_structured_content(
    content: str | None,
    response_model: type[BaseModel],
    *,
    provider: LLMProvider = "groq",
) -> BaseModel:
    """Parse and validate one structured model response."""

    if not content or not content.strip():
        raise LLMServiceError(
            "The model returned an empty structured response.",
            code="empty_response",
            provider=provider,
        )

    try:
        parsed_data = json.loads(clean_json_content(content))
        return response_model.model_validate(parsed_data)
    except (json.JSONDecodeError, ValidationError) as error:
        raise LLMServiceError(
            "The model returned invalid structured data.",
            code="structured_output",
            provider=provider,
        ) from error


def generate_groq_json_fallback(
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
) -> BaseModel:
    """Retry a Groq schema failure with JSON-object output."""

    last_error: Exception | None = None

    for _ in range(STRUCTURED_FALLBACK_ATTEMPTS):
        try:
            completion = create_groq_chat_completion(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0,
                reasoning_effort="low",
                max_completion_tokens=STRUCTURED_OUTPUT_TOKEN_LIMIT,
                response_format={"type": "json_object"},
            )
        except BadRequestError as error:
            if is_schema_generation_error(error):
                last_error = error
                continue

            raise wrap_groq_api_error(
                error,
                "Groq rejected the structured request.",
            ) from error
        except APIError as error:
            raise wrap_groq_api_error(
                error,
                "Groq could not generate structured data.",
            ) from error

        if not completion.choices:
            last_error = LLMServiceError(
                "Groq returned no completion choices.",
                code="empty_response",
                provider="groq",
            )
            continue

        try:
            return validate_structured_content(
                completion.choices[0].message.content,
                response_model,
                provider="groq",
            )
        except LLMServiceError as error:
            last_error = error

    raise LLMServiceError(
        "Groq could not produce valid structured data.",
        code="structured_output",
        provider="groq",
    ) from last_error


def generate_openrouter_structured_data(
    *,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict[str, Any],
    response_model: type[BaseModel],
) -> BaseModel:
    """Generate and validate JSON using OpenRouter's free router."""

    schema_text = json.dumps(
        json_schema,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                "Return only one valid JSON object matching this "
                "schema exactly. Do not use Markdown fences:\n"
                f"{schema_text}"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    last_error: Exception | None = None

    for attempt in range(STRUCTURED_FALLBACK_ATTEMPTS + 1):
        try:
            content = create_openrouter_completion(
                messages=messages,
                response_format=(
                    {"type": "json_object"}
                    if attempt == 0
                    else None
                ),
                max_tokens=STRUCTURED_OUTPUT_TOKEN_LIMIT,
            )
        except LLMServiceError as error:
            if error.code != "structured_output":
                raise
            last_error = error
            continue

        try:
            return validate_structured_content(
                content,
                response_model,
                provider="openrouter",
            )
        except LLMServiceError as error:
            last_error = error
            messages = [
                messages[0],
                messages[1],
                {
                    "role": "assistant",
                    "content": content[:6000],
                },
                {
                    "role": "user",
                    "content": (
                        "Correct the previous response so it matches "
                        "the required JSON schema. Include every "
                        "required field, remove unsupported fields, "
                        "and return only the corrected JSON object."
                    ),
                },
            ]

    raise LLMServiceError(
        "OpenRouter could not produce valid structured data.",
        code="structured_output",
        provider="openrouter",
    ) from last_error


def generate_structured_data(
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    json_schema: dict[str, Any],
    response_model: type[BaseModel],
    provider: str = "groq",
) -> BaseModel:
    """Generate and validate structured output."""

    system_prompt, user_prompt = validate_prompts(
        system_prompt,
        user_prompt,
    )

    if not isinstance(schema_name, str):
        raise TypeError("schema_name must be a string.")

    schema_name = schema_name.strip()

    if not schema_name:
        raise ValueError("schema_name cannot be empty.")

    selected_provider = normalize_provider(provider)

    if selected_provider == "openrouter":
        return generate_openrouter_structured_data(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            response_model=response_model,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = create_groq_chat_completion(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
            reasoning_effort="low",
            max_completion_tokens=STRUCTURED_OUTPUT_TOKEN_LIMIT,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
        )
    except BadRequestError as error:
        if not is_schema_generation_error(error):
            raise wrap_groq_api_error(
                error,
                "Groq rejected the structured request.",
            ) from error

        schema_text = json.dumps(
            json_schema,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fallback_messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "Return only one valid JSON object that "
                    "matches this schema exactly:\n"
                    f"{schema_text}"
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        return generate_groq_json_fallback(
            fallback_messages,
            response_model,
        )
    except APIError as error:
        raise wrap_groq_api_error(
            error,
            "Groq could not generate structured data.",
        ) from error

    if not completion.choices:
        raise LLMServiceError(
            "Groq returned no completion choices.",
            code="empty_response",
            provider="groq",
        )

    return validate_structured_content(
        completion.choices[0].message.content,
        response_model,
        provider="groq",
    )

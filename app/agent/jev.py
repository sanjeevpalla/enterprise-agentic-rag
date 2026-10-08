"""TypeSafe's Jev decision model, via the official SDK (``typesafe-sdk``).

Jev is a "System One" model: given ``state`` and typed questions, it returns typed answers
with calibrated probabilities in one pass, without generating text. That makes it fast and
cheap for routing. See https://docs.typesafe.ai/introduction/quickstart.

    client = TypeSafeClient()                     # reads TYPESAFE_API_KEY
    response = client.system_one(state=..., questions={"route": Choice(instructions=..., criteria={...})})
    response.answers["route"].choice              # e.g. "technical"
"""

from __future__ import annotations

from typesafe_sdk import RetryPolicy, TypeSafeClient

from app.config import Settings, get_settings


def build_typesafe_client(settings: Settings | None = None) -> TypeSafeClient:
    """TypeSafe client from settings (TYPESAFE_API_KEY, JEV_MODEL, ...).

    The SDK retries rate limits (429), overload (529), other 5xx errors, timeouts and
    connection errors with backoff, honouring the server's Retry-After header.
    """
    settings = settings or get_settings()
    if settings.typesafe_api_key is None:
        raise ValueError(
            "PLANNER_PROVIDER=jev needs TYPESAFE_API_KEY (environment or .env); "
            "get one at console.typesafe.ai, or set PLANNER_PROVIDER=llm"
        )
    return TypeSafeClient(
        api_key=settings.typesafe_api_key.get_secret_value(),
        model=settings.jev_model,
        base_url=settings.typesafe_base_url,
        timeout=settings.jev_timeout,
        retry=RetryPolicy(max_retries=settings.jev_max_retries),
    )
